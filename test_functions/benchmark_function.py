from typing import List, Union

import numpy as np


class Benchmark:
    """
    Abstract benchmark function.

    Args:
        dim: dimensionality of the objective function
        ub: the upper bound, the object will have the attribute ub_vec which is an np array of length dim filled with ub
        lb: the lower bound, the object will have the attribute lb_vec which is an np array of length dim filled with lb
        benchmark_func: the benchmark function, should inherit from SyntheticTestFunction
    """

    def __init__(self, dim: int, ub: np.ndarray, lb: np.ndarray, **kwargs):

        lb = np.array(lb)
        ub = np.array(ub)
        if (
                not lb.shape == ub.shape
                or not lb.ndim == 1
                or not ub.ndim == 1
                or not dim == len(lb) == len(ub)
        ):
            raise ValueError("lb and ub must be of shape (dim,)")
        if not np.all(lb < ub):
            raise ValueError("Lower bounds must be smaller than upper bounds")
        self.__dim = dim
        self.__lb = lb.astype(np.float32)
        self.__ub = ub.astype(np.float32)
        self.__name = kwargs.get("name", 'name_not_set')

    @property
    def dim(self) -> int:
        """
        The benchmark dimensionality

        Returns: the benchmark dimensionality

        """
        return self.__dim

    @property
    def lb(self) -> np.ndarray:
        """
        The lower bound of the search space of this benchmark (length = benchmark dim)

        Returns: The lower bound of the search space of this benchmark (length = benchmark dim)

        """
        return self.__lb

    @property
    def ub(self) -> np.ndarray:
        """
        The upper bound of the search space of this benchmark (length = benchmark dim)

        Returns: The upper bound of the search space of this benchmark (length = benchmark dim)

        """
        return self.__ub

    @property
    def name(self) -> str:
        """
        The name of the benchmark function

        Returns: The name of the benchmark function

        """
        return self.__name

    def __call__(self, x: Union[np.ndarray, List[float], List[List[float]]]):
        raise NotImplementedError()
