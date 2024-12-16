# -*- coding: utf-8 -*-
"""
"""
import os
import stat
import subprocess
import sys
import tempfile
import urllib
from collections import OrderedDict
from logging import info, warning
from platform import machine
from typing import List, Optional, Union

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVR

# from .bipedal_walker import BipedalWalker, heuristic_bipedal
# from .lunar_lander import LunarLander, heuristic_turbo
# from .push_function import PushReward


class MoptaSoftConstraints:
    """
    Mopta08 benchmark with soft constraints as described in https://arxiv.org/pdf/2103.00349.pdf
    Supports i386, x86_84, armv7l

    Args:
        temp_dir: Optional[str]: directory to which to write the input and output files (if not specified, a temporary directory will be created automatically)
        binary_path: Optional[str]: path to the binary, if not specified, the default path will be used
    """

    def __init__(
            self,
            temp_dir: Optional[str] = None,
            binary_path: Optional[str] = None,
            noise_std: Optional[float] = 0,
            **kwargs,
    ):
        # super().__init__(124, np.ones(124), np.zeros(124), noise_std=noise_std)
        lb = np.zeros(124)
        ub = np.ones(124)
        self.noise_std = noise_std
        self._dim = 124
        self._lb_vec = lb.astype(np.float32)
        self._ub_vec = ub.astype(np.float32)
        if binary_path is None:
            self.sysarch = 64 if sys.maxsize > 2 ** 32 else 32
            self.machine = machine().lower()
            if self.machine == "armv7l":
                assert self.sysarch == 32, "Not supported"
                self._mopta_exectutable = "mopta08_armhf.bin"
            elif self.machine == "x86_64":
                assert self.sysarch == 64, "Not supported"
                self._mopta_exectutable = "mopta08_elf64.bin"
            elif self.machine == "i386":
                assert self.sysarch == 32, "Not supported"
                self._mopta_exectutable = "mopta08_elf32.bin"
            elif self.machine == "amd64":
                assert self.sysarch == 64, "Not supported"
                self._mopta_exectutable = "mopta08_amd64.exe"
            else:
                raise RuntimeError(
                    "Machine with this architecture is not supported")
            dir_path = os.path.dirname(os.path.realpath(__file__))
            self._mopta_exectutable = os.path.join(
                dir_path, "mopta08", self._mopta_exectutable
            )

            if not os.path.exists(self._mopta_exectutable):
                basename = os.path.basename(self._mopta_exectutable)
                print(
                    f"Mopta08 executable for this architecture not locally available. Downloading '{basename}'...")
                urllib.request.urlretrieve(
                    f"https://mopta.papenmeier.io/{os.path.basename(self._mopta_exectutable)}",
                    self._mopta_exectutable)
                os.chmod(self._mopta_exectutable, stat.S_IXUSR)

        else:
            self._mopta_exectutable = binary_path
        if temp_dir is None:
            self.directory_file_descriptor = tempfile.TemporaryDirectory()
            self.directory_name = self.directory_file_descriptor.name
        else:
            if not os.path.exists(temp_dir):
                warning(
                    f"Given directory '{temp_dir}' does not exist. Creating...")
                os.mkdir(temp_dir)
            self.directory_name = temp_dir

        # custom param
        self.input_dim = self._dim
        self.name = 'mopta124'
        self.bounds = [(0., 1.)]*self._dim

    def __call__(self, x):
        # super(MoptaSoftConstraints, self).__call__(x)
        x = np.array(x)
        if x.ndim == 0:
            x = np.expand_dims(x, 0)
        if x.ndim == 1:
            x = np.expand_dims(x, 0)
        assert x.ndim == 2
        # create tmp dir for mopta binary

        vals = np.array([self._call(y) for y in x]).squeeze()
        return vals + np.random.normal(
            np.zeros_like(vals), np.ones_like(
                vals) * self.noise_std, vals.shape
        )

    def _call(self, x: np.ndarray):
        """
        Evaluate Mopta08 benchmark for one point

        Args:
            x: one input configuration

        Returns:value with soft constraints

        """
        assert x.ndim == 1
        # write input to file in dir
        with open(os.path.join(self.directory_name, "input.txt"), "w+") as tmp_file:
            for _x in x:
                tmp_file.write(f"{_x}\n")
        # pass directory as working directory to process
        popen = subprocess.Popen(
            self._mopta_exectutable,
            stdout=subprocess.PIPE,
            cwd=self.directory_name,
        )
        popen.wait()
        # read and parse output file
        output = (
            open(os.path.join(self.directory_name, "output.txt"), "r")
            .read()
            .split("\n")
        )
        output = [x.strip() for x in output]
        output = np.array([float(x) for x in output if len(x) > 0])
        value = output[0]
        constraints = output[1:]
        # see https://arxiv.org/pdf/2103.00349.pdf E.7
        return value + 10 * np.sum(np.clip(constraints, a_min=0, a_max=None))

    def func(self, x):
        return self.__call__(x)
