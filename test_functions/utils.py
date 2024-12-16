import argparse
import random

import botorch
import numpy as np
import torch

from .benchmark_function import Benchmark
from .function_realworld_bo.functions_realworld_bo import *
from .functions_bo import *
from .highdim_functions import *
from .lasso_benchmark import *


def get_arguments():
    default_w = 0.729
    default_c1 = 2.05*default_w
    default_c2 = 2.05*default_w
    parser = argparse.ArgumentParser(description='Process inputs')
    parser.add_argument('-f', '--func_name', help='specify the test function')
    parser.add_argument('-d', '--dim', type=int, help='specify the problem dimensions', default=10)
    parser.add_argument('-n', '--maxevals', type=int, help='specify the maxium number of evaluations to collect in the search')
    parser.add_argument('-a', '--acq', type=str, help='Acquisition function', default='ts')
    parser.add_argument('--output', type=str, help='Output folder', default='output')
    parser.add_argument('-m', '--popsize', type=int, help='Number of particles', default=20)
    parser.add_argument('--w', '-w', type=float, help='PSO inertia parameter', default=default_w)
    parser.add_argument('--c1', '-c1', type=float, help='PSO cognitive parameter', default=default_c1)
    parser.add_argument('--c2', '-c2', type=float, help='PSO social parameter', default=default_c2)
    parser.add_argument('--seed', type=int, help='Random seed', default=None)


    args, _ = parser.parse_known_args()
    

    dim = args.dim
    func_name = args.func_name.lower()
    if func_name == 'ackley':
        func_core = ackley(dim)
    elif func_name == 'rastrigin':
        func_core = rastrigin(dim)
    elif func_name == 'ellipsoid':
        func_core = ellipsoid(dim)
    elif func_name == 'levy':
        func_core = Levy(dim)
    elif func_name == 'alpine':
        func_core = alpine(dim)
    elif func_name == 'rosenbrock':
        func_core = rosenbrock(dim)
    elif func_name == 'powell':
        func_core = powell(dim)
    elif func_name == 'lasso-simple':
        func_core = LassoSimpleBenchmark()
    elif func_name == 'lasso-medium':
        func_core = LassoMediumBenchmark()
    elif func_name == 'lasso-high':
        func_core = LassoHighBenchmark()
    elif func_name == 'lasso-hard':
        func_core = LassoHardBenchmark()
    elif func_name == 'lasso-diabete':
        func_core = LassoDiabetesBenchmark()
    elif func_name == 'lasso-dna':
        func_core = LassoDNABenchmark()
    elif func_name == 'hartmann500':
        func_core = Hartmann500D()
    elif func_name == 'branin500':
        func_core = Branin500D()
    elif func_name == 'schaffer100':
        func_core = Schaffer100()
    elif func_name == 'bohachevsky100':
        func_core = Bohachevsky100()
    elif func_name == 'mopta':
        func_core = MoptaSoftConstraints()
    elif func_name == 'hopper':
        from .function_realworld_bo.functions_mujoco import Hopper
        func_core = Hopper()
    elif func_name == 'walker2d':
        from .function_realworld_bo.functions_mujoco import Walker2d
        func_core = Walker2d()
    elif func_name == 'half-cheetah':
        from .function_realworld_bo.functions_mujoco import HalfCheetah
        func_core = HalfCheetah()
    elif func_name == 'humanoid':
        from .function_realworld_bo.functions_mujoco import Humanoid
        func_core = Humanoid()
    elif func_name == 'ant':
        from .function_realworld_bo.functions_mujoco import Ant
        func_core = Ant()
    elif func_name == 'swimmer':
        from .function_realworld_bo.functions_mujoco import Swimmer
        func_core = Swimmer()
    else:
        raise NotImplementedError(f'Objective function {func_name} is not supported')
    
    class CustomFunction(Benchmark):
        def __init__(self, base_function):
            self.base_function = base_function
            lb = np.array(self.base_function.bounds)[:, 0]
            ub = np.array(self.base_function.bounds)[:, 1]
            input_dim = base_function.input_dim
            name = base_function.name if hasattr(base_function, 'name') else 'name_not_set'
            super().__init__(dim=input_dim, ub=ub, lb=lb, name=name)

        def __call__(self, x: Union[np.ndarray, List[float], List[List[float]]]):
            return float(self.base_function.func(x))
    
    f = CustomFunction(func_core)
    
    dict = {
        'func_name': func_name,
        "f": f,
        'max_evals': args.maxevals,
        'seed': args.seed,
        'output': args.output,
        'popsize': args.popsize,
        'w': args.w,
        'c1': args.c1,
        'c2': args.c2,
        'acq': args.acq
    }
    return dict

    
def set_seed(seed=1234):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    botorch.manual_seed(seed)


def to_unit_cube(x, lb, ub):
    """Project to [0, 1]^d from hypercube with bounds lb and ub"""
    assert np.all(lb < ub) and lb.ndim == 1 and ub.ndim == 1 and x.ndim == 2
    xx = (x - lb) / (ub - lb)
    return xx


def from_unit_cube(x, lb, ub):
    """Project from [0, 1]^d to hypercube with bounds lb and ub"""
    assert np.all(lb < ub) and lb.ndim == 1 and ub.ndim == 1 and x.ndim == 2
    xx = x * (ub - lb) + lb
    return xx


def latin_hypercube(n_pts, dim):
    """Basic Latin hypercube implementation with center perturbation."""
    X = np.zeros((n_pts, dim))
    centers = (1.0 + 2.0 * np.arange(0.0, n_pts)) / float(2 * n_pts)
    for i in range(dim):  # Shuffle the center locataions for each dimension.
        X[:, i] = centers[np.random.permutation(n_pts)]

    # Add some perturbations within each box
    pert = np.random.uniform(-1.0, 1.0, (n_pts, dim)) / float(2 * n_pts)
    X += pert
    return X
