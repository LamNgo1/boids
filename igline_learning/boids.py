
import base64
import math
import pickle
import time
from copy import deepcopy
from logging import debug, info, warning
from typing import Dict, Optional, Tuple

import gpytorch
import numpy as np
import torch
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.optimize import minimize
from pymoo.termination import get_termination

from baxus_embedding.baxus_configuration import BaxusBehavior
from baxus_embedding.data_utils import join_data
from baxus_embedding.projections import AxUS, ProjectionModel
from baxus_embedding.utils import (from_1_around_origin,
                                   one_around_origin_latin_hypercube)
from igline_learning.multiobjective_acquisition import \
    IncumbentGuidedMOAcquisitionFunction as MOAF
from igline_learning.multiobjective_acquisition import get_acquisition_function
from igline_learning.swarm import Swarm
from surrogate.gp import train_gp
from surrogate.gp_configuration import GPBehaviour
from test_functions.benchmark_function import Benchmark


class BOIDS():
    """
    Main class of the BOIDS algorithm.

    """

    def __init__(
            self,
            f: Benchmark,
            target_dim: int,
            n_init: int,
            max_evals: int,
            acq: str = 'ts',
            behavior: BaxusBehavior = BaxusBehavior(),
            max_cholesky_size=2000,
            dtype="float64",
            **kwargs,
    ):
        # Basic settings
        self.f = f
        self.lb = f.lb
        self.ub = f.ub
        self.n_init = n_init
        self.n_evals = 0
        self.max_evals = max_evals
        self.acq = acq
        self.max_cholesky_size = max_cholesky_size
        self.dtype = torch.float32 if dtype == "float32" else torch.float64

        # BAxUS settings
        self.baxus_behavior = behavior
        self._previous_restarts = []  # track the iterations of the full restart
        self._dim_in_iterations = {}  # track the iteration in which target dim updates

        # Set input dim and target dim
        self.__target_dim = target_dim
        self.__input_dim = f.dim
        self._init_target_dim = target_dim

        # Adjust the initial target dim if necessary (from BAxUS)
        if self.baxus_behavior.adjust_initial_target_dim:
            target_dim = self._adjust_init_target_dim()
            self._init_target_dim = target_dim
            self.__target_dim = target_dim

        # Construct BAxUS projector
        if self.input_dim != self.target_dim:
            self.projector = AxUS(self.input_dim, self.target_dim)
        else:
            self.projector = False

        # Termination factor K to control budget for each subpsace
        self.__K_min = 0
        self.__K_max = 7
        self.__K_init = 1
        self.__succtol = 3
        self.__success_decision_factor = 0.001
        self.K = self.K_init

        # Initialize parameters
        self._restart()

        # PSO swarm
        w = kwargs.get('w', 0.729)
        self.swarm_param = {
            'n_particles': kwargs.get('popsize', 20),
            'c1': kwargs.get('c1', w*2.05),
            'c2': kwargs.get('c2', w*2.05),
            'w': kwargs.get('w', w),
        }
        self.swarm: Swarm = None

        # Save the full history
        self.X = np.zeros((0, self.input_dim))
        self.fX = np.zeros((0, 1))

        # logging data
        self.boids_data = []
        self.projector_data = []
        self.output_file = kwargs.get('output_file', None)

        self._restart()

    @property
    def K_min(self) -> float:
        """
        The minimum termination factor to terminate the current subspace.

        Returns: The minimum termination factor to terminate the current subspace.

        """
        return self.__K_min

    @property
    def K_max(self) -> float:
        """
        The maximum termination factor to terminate the current subspace.

        Returns: The maximum termination factor to terminate the current subspace.

        """
        return self.__K_max

    @property
    def K_init(self) -> float:
        """
        The initial termination factor to terminate the current subspace.

        Returns: The initial termination factor to terminate the current subspace.

        """
        return self.__K_init

    @property
    def succtol(self) -> int:
        """
        The success tolerance. See TuRBO and BAxUS for more details.

        Returns: The success tolerance.

        """
        return self.__succtol

    @property
    def input_dim(self) -> int:
        """
        The input dimensionality

        Returns: the input dimensionality

        """
        return self.__input_dim

    @property
    def success_decision_factor(self) -> float:
        """
        The success decision factor of the function values.

        Returns: The success decision factor of the function values.

        """
        return self.__success_decision_factor

    @property
    def input_dim(self) -> int:
        """
        The input dimensionality

        Returns: the input dimensionality

        """
        return self.__input_dim

    @property
    def target_dim(self) -> int:
        """
        The target dimensionality.

        Returns: the target dimensionality

        """
        return self.__target_dim

    @target_dim.setter
    def target_dim(self, target_dim: int) -> None:
        """
        Setter for the target dimensionality

        Args:
            target_dim:  the new target dimensionality

        Returns:

        """
        self._dim_in_iterations[self.n_evals] = target_dim
        self.__target_dim = target_dim

    @property
    def n_cand(self) -> int:
        """
        The number of candidates for the discrete Thompson sampling, also used for the MOAF.

        Returns: the number of candidates for the discrete Thompson sampling

        """
        return min(100 * self.target_dim, 5000)

    @property
    def _dimension_importances(self) -> np.ndarray:
        """
        The (inverse) dimension importances. This just returns the lengthscales of the GP ARD kernel.

        Returns: The (inverse) dimension importances. This just returns the lengthscales of the GP ARD kernel.

        """
        return np.array(self.lengthscales)

    @property
    def _init_dim_in_restart(self) -> int:
        """
        The dim with which the current restart started.

        Returns: The dim with which the current restart started.

        """
        dim_in_iterations = self._dim_in_iterations
        if len(dim_in_iterations) == 0:
            # target dim was not yet adjusted
            return self._init_target_dim
        else:
            eval_when_tr_started = 0 if len(
                self._previous_restarts) == 0 else self._previous_restarts[-1]
            tr_adjust_iters = np.array(list(dim_in_iterations.keys()))
            min_iter = min(
                tr_adjust_iters[tr_adjust_iters >= eval_when_tr_started])
            return self._dim_in_iterations[min_iter]

    @property
    def _budget_lost_in_previous_restarts(self) -> int:
        """
        The number of function evaluations used in previous restarts.

        Returns: The number of function evaluations used in previous restarts.

        """
        return self.n_init if len(self._previous_restarts) == 0 else self._previous_restarts[-1]

    def _adjust_init_target_dim(self) -> int:
        """
        Adjust the initial target dimension such that the final target dimension
        is as close to the ambient dimensionality as possible given a fixed b. See BAxUS for more details.

        Returns: int: the adjusted initial target dimension.

        """

        def ndiff(b, d0):
            psi = 1
            desired_final_dim = self.input_dim
            initial_target_dim = d0

            base = psi * b + 1
            n = round(math.log(desired_final_dim / initial_target_dim, base))
            df_br = round(base ** n * initial_target_dim)
            res = np.abs(df_br - desired_final_dim)
            return res, n

        i_b, i_d0 = self.baxus_behavior.n_new_bins, self._init_target_dim

        def _fmin(d0):
            return ndiff(b=i_b, d0=d0)[0]

        bounds = (2, i_b + 1)

        x_best = 1
        y_best = _fmin(x_best)
        for j_d0 in range(bounds[0], bounds[1]):
            if _fmin(j_d0) < y_best:
                x_best = j_d0
                y_best = _fmin(j_d0)

        return x_best

    def _restart(self, K: float = None) -> None:
        """
        Reset observations for the current target subspace, reset counter, reset base length

        Args:
            length: new base length after resetting, if not set, length_init will be used.

        """
        self._X = np.empty((0, self.target_dim))
        self._fX = np.empty((0, 1))

        self.failcount = 0
        self.succcount = 0
        if K is None:
            self.K = self.K_init
        else:
            self.K = K

    @property
    def failtol(self) -> float:
        """
        The fail tolerance adapted from the BAxUS algorithm.
        Is computed dynamically depending on the split we are in as the fail tolerance is dependent on the
        current target dimensionality.

        Returns: the fail tolerance

        """
        ft_max = np.max([4.0, self.target_dim])
        if self.target_dim == self.input_dim:
            return ft_max

        desired_final_dim = self.input_dim
        evaluation_budget = self.max_evals if self.baxus_behavior.budget_until_input_dim == 0 else self.baxus_behavior.budget_until_input_dim
        evaluation_budget = evaluation_budget - self._budget_lost_in_previous_restarts

        psi = 1
        new_bins_on_split = self.baxus_behavior.n_new_bins
        _log_base = psi * new_bins_on_split + 1
        n = round(math.log(desired_final_dim /
                  self._init_dim_in_restart, _log_base))  # splits

        def _budget(dim):

            return (evaluation_budget * dim * (1 - _log_base)) / (self._init_dim_in_restart * (1 - _log_base ** (n + 1)))

        budget = _budget(self.target_dim)

        del (
            psi,
            new_bins_on_split,
            evaluation_budget,
        )

        length_init = 0.8       # see TuRBO and BAxUS for more details
        length_min = 0.5 ** -7  # see TuRBO and BAxUS for more details

        gamma = 2 * math.log(length_min / length_init, 0.5)
        if gamma == 0:
            return ft_max
        ft = math.ceil(budget / gamma)
        failtol = max(1, min(ft, ft_max))

        return failtol

    def _adjust_termination_factor(self, fX_next) -> None:
        """
        Adjust the termination factor K of the current subspace depending on the outcome of the next evaluation.
        If the next evaluation is better than the current, increase success count and potentially increase K.
        Otherwise, increase fail count and potentially decrease K.

        Args:
            fX_next: the function value of the next point

        """
        prev_data = self._fX

        if np.min(fX_next) < np.min(
                prev_data
        ) - self.success_decision_factor * math.fabs(np.min(prev_data)):
            self.succcount += 1
            self.failcount = 0
        else:
            self.succcount = 0
            self.failcount += 1
        if self.succcount >= self.succtol:
            debug(f"eval {self.n_evals}: reducing termination factor K")
            self.K = max([self.K - 1, self.K_min])
            self.succcount = 0
        elif self.failcount >= self.failtol:
            debug(f"eval {self.n_evals}: increasing termination factor K")
            self.K = self.K + 1
            self.failcount = 0

    def _choose_splitting_dim(
            self,
            projector: AxUS,
    ) -> Dict[int, int]:
        """
        Choose a new splitting dim based on our defined behavior. See BAxUS for more details.

        Args:
            projector: the projection model used

        Returns: the new splitting dim or -1 if none could be found


        """

        n_dims_to_split = self.target_dim

        n_new_bins = self.baxus_behavior.n_new_bins
        n_new_bins = (n_new_bins + 1) * n_dims_to_split
        assert n_new_bins >= 2 * n_dims_to_split, (
            "Number of new bins has "
            "to be at least 2 times"
            "the number of dimensions"
            "to split"
        )
        weights = self._dimension_importances
        indices_with_lengthscales = {i: weights[i]
                                     for i in range(self.target_dim)}
        indices_sorted_by_lengthscales = sorted(
            [i for i in indices_with_lengthscales.keys()],
            key=lambda i: indices_with_lengthscales[i],
        )
        splittable_idxs = np.array(
            [
                i
                for i in indices_sorted_by_lengthscales
                if len(projector.contributing_dimensions(i)) > 1
            ]
        )
        n_dims_to_split = min(len(splittable_idxs), n_dims_to_split)
        if n_dims_to_split == 0:
            return {}
        n_bins_per_dim = n_new_bins // n_dims_to_split
        bins_per_dim = np.array(
            [
                min(n_bins_per_dim, len(projector.contributing_dimensions(i)))
                for i in splittable_idxs
            ]
        )
        cum_sum = np.cumsum(bins_per_dim)
        dims_to_split = np.sum(cum_sum <= n_new_bins)
        dims_and_bins = {
            splittable_idxs[i]: bins_per_dim[i] for i in range(dims_to_split)
        }

        return dims_and_bins

    def _resample_and_restart(self, n_points: int, K: float = None) -> None:
        """
        Resample new initial points and reset algorithm.

        Args:
            n_points: number of new initial points
            K: new termination factor after resetting

        Returns: None

        """
        # Initialize parameters
        self._restart(K=K)

        # Generate and evaluate initial design points
        n_pts = min(self.max_evals - self.n_evals, n_points)
        X_init = one_around_origin_latin_hypercube(n_pts, self.target_dim)

        X_init_up = from_1_around_origin(
            self.projector.project_up(X_init.T).T, self.lb, self.ub
        )
        fX_init = np.array([[self.f(x)] for x in X_init_up])
        # Update budget and set as initial data for this restart
        self.n_evals += n_pts
        self._X = deepcopy(X_init)
        self._fX = deepcopy(fX_init)

        # Append data to the global history
        self.X = np.vstack((self.X, deepcopy(X_init_up)))
        self.fX = np.vstack((self.fX, deepcopy(fX_init)))

    @staticmethod
    def _projector_as_base64(projector: ProjectionModel) -> str:
        """
        Return the current projection model as a Base64 string. For debugging purposes.
        Args:
            projector: the projector to return as base64.

        Returns: the current projection model as a Base64 string.

        """
        if isinstance(projector, AxUS):
            return base64.b64encode(pickle.dumps(projector)).decode("utf-8")
        return ""

    @staticmethod
    def _compute_t_range(_pso_vel_norm: np.ndarray, _x: np.ndarray, _lb: np.ndarray, _ub: np.ndarray):
        """
        Compute the parameter t for each parameterized lines (x = x0 + v*t) such that x is in the bounds [_lb, _ub].
        Args:
            _pso_vel_norm: the normalized direction
            _x: the current position
            _lb: the lower bound
            _ub: the upper bound
        """
        original_ndim = _pso_vel_norm.ndim
        if _pso_vel_norm.ndim == _x.ndim == 1:
            _pso_vel_norm = _pso_vel_norm.reshape(1, -1)
            _x = _x.reshape(1, -1)
        if _lb.ndim == _ub.ndim == 1:
            _lb = np.repeat(_lb.reshape(1, -1), _x.shape[0], axis=0)
            _ub = np.repeat(_ub.reshape(1, -1), _x.shape[0], axis=0)
        assert _pso_vel_norm.shape == _x.shape == _lb.shape == _ub.shape
        # test
        temp_min = np.ones(_x.shape) * -999
        temp_max = np.ones(_x.shape) * 999
        mask_pos = _pso_vel_norm > 0
        mask_neg = _pso_vel_norm < 0
        temp_min[mask_pos] = (_lb[mask_pos] - _x[mask_pos]
                              ) / _pso_vel_norm[mask_pos]
        temp_min[mask_neg] = (_ub[mask_neg] - _x[mask_neg]
                              ) / _pso_vel_norm[mask_neg]
        temp_max[mask_pos] = (_ub[mask_pos] - _x[mask_pos]
                              ) / _pso_vel_norm[mask_pos]
        temp_max[mask_neg] = (_lb[mask_neg] - _x[mask_neg]
                              ) / _pso_vel_norm[mask_neg]

        t_min = np.max(temp_min, axis=1)
        t_max = np.min(temp_max, axis=1)
        if original_ndim == 1:
            return t_min[0], t_max[0]
        return t_min, t_max

    def _gen_candidates(self, X: np.ndarray, fX: np.ndarray) -> Tuple[np.ndarray, int]:
        """
        Generate candidates assuming X has been scaled to [-1,1]^d.

        Args:
            X: the local observed x-values
            fX: the local observed y-values
        Returns:
            a tuple (X_next, id_particle) of the next candidate for evaluation and the particle (line) index
        """

        fX = fX.copy() * (-1)
        # Standardize local function values.
        observed_mu, observed_sigma = np.mean(fX), fX.std(ddof=1)
        observed_sigma = 1.0 if observed_sigma < 1e-6 else observed_sigma
        fX = (deepcopy(fX) - observed_mu) / observed_sigma

        # Train GP
        device, dtype = torch.device("cpu"), self.dtype
        # We use CG + Lanczos for training if we have enough data
        with gpytorch.settings.max_cholesky_size(self.max_cholesky_size):
            X_torch = torch.tensor(X).to(device=device, dtype=dtype)
            y_torch = torch.tensor(fX).to(device=device, dtype=dtype)
            gp, hyper = train_gp(
                train_x=X_torch,
                train_y=y_torch,
            )

        # Save GP lengthscales for splitting
        weights = gp.lengthscales
        weights = weights / weights.mean()  # This will make the next line more stable
        weights = weights / np.prod(
            np.power(weights, 1.0 / len(weights))
        )
        self.lengthscales = weights

        # We may have to move the GP to a new device
        gp = gp.to(dtype=dtype, device=device)

        f_alpha = get_acquisition_function(
            self.acq, gp, observed_mu, observed_sigma, dtype, device, self.max_cholesky_size)

        lb = -np.ones(self.target_dim)
        ub = np.ones(self.target_dim)

        # Adaptive Line Selection - MAB Thompson Sampling
        pso_vel_batch = self.swarm.compute_pso_vel_batch()
        x_all_particles = deepcopy(self.swarm.positions)
        v_all_particles = deepcopy(pso_vel_batch)
        v_norm_all_particles = v_all_particles / \
            np.linalg.norm(v_all_particles, axis=1).reshape(-1, 1)
        n_cand_each_line = min(5000//self.swarm.n_particles, self.n_cand)

        # Parameterize the lines x = x0 + v*t such that x is in the bounds [lb, ub]
        t_mins, t_maxs = BOIDS._compute_t_range(
            v_norm_all_particles, x_all_particles, lb, ub)
        assert np.all(t_mins <= t_maxs)
        t_ranges = np.random.uniform(t_mins, t_maxs, size=(
            n_cand_each_line, self.swarm.n_particles)).T

        # Generate data points on each line
        x_cand_batch = x_all_particles[:, None, :] + \
            t_ranges[:, :, None] * v_norm_all_particles[:, None, :]
        if np.any(x_cand_batch - lb < 0) or np.any(ub - x_cand_batch < 0):
            info(
                f'x_cand_batch.min()={x_cand_batch.min()}, x_cand_batch.max()={x_cand_batch.max()}')
            x_cand_batch = np.clip(x_cand_batch, lb, ub)
        x_cand = x_cand_batch.reshape(-1, self.target_dim)

        # Evaluate the reward - Thompson Sampling values
        with torch.no_grad(), gpytorch.settings.max_cholesky_size(self.max_cholesky_size):
            x_cand_torch = torch.tensor(x_cand).to(dtype=dtype)
            y_cand_ = gp.likelihood(gp(x_cand_torch)).sample().t().numpy()
        y_cand = (observed_mu + observed_sigma * y_cand_) * (-1)
        top1 = int(np.argmin(y_cand))
        idx_particle = top1 // n_cand_each_line  # index of chosen particle/line

        # Retrieve the information related to the chosen particle/line
        t_min = t_mins[idx_particle]
        t_max = t_maxs[idx_particle]
        x_current = x_all_particles[idx_particle]
        pso_vel_norm = v_norm_all_particles[idx_particle]
        gbest_x = deepcopy(self.swarm.gbest_pos.reshape(1, -1))
        pbest_x = deepcopy(self.swarm.pbest_pos[idx_particle].reshape(1, -1))

        # Initial conditions for MOAF
        t_range = np.random.uniform(t_min, t_max, size=(self.n_cand, 1))
        x_init_moaf = x_current.reshape(
            1, -1) + t_range * pso_vel_norm.reshape(1, -1)

        # Optimize the multi-objective acquisition function using NSGA-II
        pymoo_algorithm = NSGA2(sampling=x_init_moaf)
        pymoo_term = get_termination("n_gen", 100)
        pymoo_problem = MOAF(
            acq_dim=self.target_dim,
            acq_lb=lb,
            acq_ub=ub,
            alpha=f_alpha,
            x_global_incumbent=gbest_x,
            x_local_incumbent=pbest_x,
        )
        res = minimize(
            problem=pymoo_problem,
            algorithm=pymoo_algorithm,
            termination=pymoo_term,
            seed=np.random.randint(1e6),
        )
        pareto_set = res.X
        pareto_front = res.F

        # Optimize w.r.t f_alpha
        idx_best = np.argmin(pareto_front[:, 0])
        X_next = pareto_set[idx_best, :].reshape(-1, self.target_dim)

        # Remove the torch variables
        del X_torch, y_torch, gp
        return X_next, idx_particle

    def _inner_optimization_step(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, int]:
        """
        Generate next data point, project up point, evaluate point

        Returns: next point in target space, next point in input space, function value of the next point, partilce/line index

        """
        # Warp inputs
        X = self._X
        fX = deepcopy(self._fX).ravel()

        # Generate next data points in target subspace
        is_cands = self._gen_candidates(X, fX)
        X_next, idx_particle = is_cands

        # Project X_next to input space
        X_next_up = from_1_around_origin(
            self.projector.project_up(
                X_next.T).T if self.projector else X_next,
            self.lb,
            self.ub,
        )

        # Evaluate batch
        fX_next = np.array([[self.f(x)] for x in X_next_up])

        # Update termination factor
        self._adjust_termination_factor(fX_next)

        # Update budget and append data
        self.n_evals += 1
        self._X = np.vstack((self._X, X_next))
        self._fX = np.vstack((self._fX, fX_next))
        self.swarm.update_pos(
            idx=idx_particle, new_pos=X_next, new_cost=fX_next.flatten())

        # Append data to the global history
        self.X = np.vstack((self.X, deepcopy(X_next_up)))
        self.fX = np.vstack((self.fX, deepcopy(fX_next)))

        return X_next, X_next_up, fX_next, idx_particle

    def optimize(self) -> None:
        """
        Run the optimization. Simplify from BAxUS code.

        Returns: None

        """
        while self.n_evals < self.max_evals:
            n_pts = min(self.max_evals - self.n_evals, self.n_init)
            # only executed if we already gathered data, i.e., not in the first run
            if len(self._fX) > 1:
                # target dim increase
                n_evals, fbest = self.n_evals, self._fX.min()
                info(f"Restarting with fbest = {fbest:.4}")

                # Split target dimension, will be used if we made progress and if not -1
                dims_and_bins = self._choose_splitting_dim(self.projector)
                if dims_and_bins:  # if we have a remaining-splitting dim
                    splitting_dims = list(dims_and_bins.keys())
                    n_new_bins = sum(list(dims_and_bins.values()))
                    # self._split_points.append(self.n_evals)
                    for splitting_dim, n_bins in dims_and_bins.items():
                        info(
                            f"Splitting dimension {splitting_dim + 1} into {n_bins} new "
                            f"bins with lengthscale: {self.lengthscales[splitting_dim]:.4} and contributing input "
                            f"dimensions {sorted(self.projector.contributing_dimensions(splitting_dim))}"
                        )
                    self.projector.increase_target_dimensionality(
                        dims_and_bins)
                    self.target_dim += n_new_bins - len(dims_and_bins)
                    self._dim_in_iterations[self.n_evals] = self.target_dim
                    info(f"New target dim = {self.target_dim}")
                    self.K = self.K_init

                    self._X = join_data(self._X, dims_and_bins)

                    self.swarm.positions = join_data(
                        self.swarm.positions, dims_and_bins)
                    self.swarm.gbest_pos = join_data(
                        self.swarm.gbest_pos.reshape(1, -1), dims_and_bins).flatten()
                    self.swarm.pbest_pos = join_data(
                        self.swarm.pbest_pos, dims_and_bins)
                    self.swarm.velocities = join_data(
                        self.swarm.velocities, dims_and_bins)
                    self.swarm.dimensions = self.target_dim

                    self.save_boids_data(idx_particle=-1)
                    self.save_projector_data()

                else:
                    warning(
                        f"BOIDS iteration {self.n_evals}: "
                        f"Re-starting with new HeSBO embedding and new subspace."
                    )
                    self.projector = AxUS(
                        input_dim=self.input_dim,
                        target_dim=self.target_dim,
                    )
                    self._resample_and_restart(
                        n_points=self.n_init, K=self.K_init)
                    self._previous_restarts.append(self.n_evals)
                    self._dim_in_iterations[self.n_evals] = self.target_dim

                    # initialize swarm
                    self.swarm = Swarm(
                        dimensions=self.target_dim, **self.swarm_param)
                    n_data_points = len(self._fX)
                    top1 = int(np.argmin(self._fX.flatten()))
                    topr = list(range(n_data_points))
                    topr.remove(top1)
                    topk = np.array([top1] + topr)[:self.swarm.n_particles]
                    np.random.shuffle(topk)
                    self.swarm.init_pos_vel(
                        init_pos=self._X[topk, :],
                        init_cost=self._fX[topk, :].flatten(),
                        clamp=(-1., 1.)
                    )
                    self.save_boids_data(idx_particle=-1)
                    self.save_projector_data()
                    self.particle_counter = [0] * self.swarm.n_particles

                self.failcount = 0
                self.succcount = 0
            else:
                self._resample_and_restart(self.n_init, self.K_init)
                fbest = self._fX.min()
                info(
                    f"BOIDS iteration {self.n_evals}: starting from fbest = {fbest:.4}")
                # initialize swarm
                self.swarm = Swarm(
                    dimensions=self.target_dim, **self.swarm_param)
                n_data_points = len(self._fX)
                top1 = int(np.argmin(self._fX.flatten()))
                topr = list(range(n_data_points))
                topr.remove(top1)
                topk = np.array([top1] + topr)[:self.swarm.n_particles]
                np.random.shuffle(topk)
                self.swarm.init_pos_vel(
                    init_pos=self._X[topk, :],
                    init_cost=self._fX[topk, :].flatten(),
                    clamp=(-1., 1.)
                )
                self.particle_counter = [0] * self.swarm.n_particles
                self.save_boids_data(idx_particle=-1)
                self.save_projector_data()
                ...

            # Thompson sample to get next suggestions
            while self.n_evals < self.max_evals and self.K <= self.K_max:
                start = time.time()
                X_next, X_next_up, fX_next, idx_particle = self._inner_optimization_step()

                end = time.time()
                self.particle_counter[idx_particle] += 1
                info(
                    f"BOIDS iteration {self.n_evals} ({self.f.name}): fx={fX_next.min():.4f}; fbest={self._fX.min():.4f}/{self.fX.min():.4f}; " +
                    f"td={self.target_dim}; K={self.K}; failcount={self.failcount}/{self.failtol}; succcount={self.succcount}/{self.succtol}; " +
                    f"iter_time={end-start:.2f}s"
                )
                self.save_boids_data(idx_particle=idx_particle)

    def optimization_results_raw(
            self,
    ) -> Tuple[Optional[np.ndarray], np.ndarray]:
        """
        The observations in the input space and their function values.

        Returns: The observations in the input space and their function values.

        """
        return self.X, self.fX.squeeze()

    def dump_to_pickle(self):
        """
        Save the current state of the algorithm to a pickle file.
        """
        results = dict()
        results['x'] = deepcopy(self.X)
        results['fx'] = deepcopy(self.fX)
        results['boids_data'] = deepcopy(self.boids_data)
        results['projector_data'] = deepcopy(self.projector_data)
        results['f_name'] = self.f.name
        results['dim'] = self.input_dim
        with open(self.output_file + '.pkl', 'wb') as file:
            pickle.dump(results, file, pickle.HIGHEST_PROTOCOL)

    def save_boids_data(self, idx_particle: int):
        """
        Save the current state of the algorithm to a pickle file.
        """
        if self.output_file is None:
            return
        try:
            self.boids_data.append({
                'eval': self.n_evals,
                'x_embedd': deepcopy(self._X[-1]),
                'x': deepcopy(self.X[-1]),
                'fx': self._fX[-1],
                'idx_particle': idx_particle,
                'K': self.K,
                'failcount': self.failcount,
                'succcount': self.succcount,
            })
            self.dump_to_pickle()
        except Exception as e:
            info(f"Error saving boids data: {e}")

    def save_projector_data(self):
        if self.output_file is None:
            return
        try:
            self.projector_data.append({
                'eval': self.n_evals,
                'projector': deepcopy(self.projector.S),
            })
            self.dump_to_pickle()
        except Exception as e:
            info(f"Error saving projector data: {e}")
