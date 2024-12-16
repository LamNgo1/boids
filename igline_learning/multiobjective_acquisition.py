
import gpytorch
import numpy as np
import torch
from botorch.acquisition.analytic import (ExpectedImprovement,
                                          LogExpectedImprovement,
                                          UpperConfidenceBound)
from botorch.models import SingleTaskGP
from pymoo.core.problem import Problem


class IncumbentGuidedMOAcquisitionFunction(Problem):
    def __init__(self, acq_dim, acq_lb, acq_ub, alpha, **kwargs):
        super().__init__(n_var=acq_dim, n_obj=3, xl=acq_lb, xu=acq_ub)
        self.gbest_x = kwargs['x_global_incumbent']
        self.pbest_x = kwargs['x_local_incumbent']
        self.alpha = alpha

    def _evaluate(self, x, out, *args, **kwargs):
        acq_val = self.alpha(x)
        dist_gbest = np.linalg.norm(x - self.gbest_x, axis=1)
        dist_pbest = np.linalg.norm(x - self.pbest_x, axis=1)
        out["F"] = [acq_val, dist_gbest, dist_pbest]


def get_acquisition_function(acq, gp: SingleTaskGP, observed_mu, observed_sigma, dtype, device, max_cholesky_size):
    '''
    Get the core acquisition function f_alpha(.) = alpha(.) for the multiobjective optimization.
    '''
    if acq == 'ts':
        random_seed_ts = np.random.randint(1e6)

        def ts_for_minimize(x: np.ndarray) -> np.ndarray:
            with torch.no_grad(), gpytorch.settings.max_cholesky_size(max_cholesky_size):
                x_torch = torch.tensor(x).to(dtype=dtype, device=device)
                pred = gp.likelihood(gp(x_torch))
                mean = pred.mean.cpu().detach().numpy()
                stddev = pred.stddev.cpu().detach().numpy()
                val = np.random.RandomState(
                    random_seed_ts).normal(loc=mean, scale=stddev)
            acq_val = (observed_mu + observed_sigma * val) * (-1)
            return acq_val
        f_alpha = ts_for_minimize
    elif acq == 'ei':
        ei = ExpectedImprovement(gp, best_f=gp.train_targets.max())

        def ei_for_minimize(x: np.ndarray) -> np.ndarray:
            x_torch = torch.tensor(x).to(
                dtype=dtype, device=device).unsqueeze(1)
            return -ei(x_torch).cpu().detach().numpy()
        f_alpha = ei_for_minimize
    elif acq == 'logei':
        logei = LogExpectedImprovement(gp, best_f=gp.train_targets.max())

        def logei_for_minimize(x: np.ndarray) -> np.ndarray:
            x_torch = torch.tensor(x).to(
                dtype=dtype, device=device).unsqueeze(1)
            return -logei(x_torch).cpu().detach().numpy()
        f_alpha = logei_for_minimize
    elif acq == 'ucb':
        ucb = UpperConfidenceBound(gp, beta=3.0)

        def ucb_for_minimize(x: np.ndarray) -> np.ndarray:
            x_torch = torch.tensor(x).to(
                dtype=dtype, device=device).unsqueeze(1)
            return -ucb(x_torch).cpu().detach().numpy()
        f_alpha = ucb_for_minimize
    else:
        raise NotImplementedError(
            f"Acquisition function {acq} not implemented")
    return f_alpha
