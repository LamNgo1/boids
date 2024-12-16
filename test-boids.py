'''Driver code for BOIDS'''
import datetime
import logging
import os
import time
import warnings

import numpy as np
from botorch.exceptions import BotorchWarning

from baxus_embedding.baxus_configuration import BaxusBehavior
from igline_learning.boids import BOIDS
from test_functions.utils import get_arguments, set_seed

warnings.simplefilter("ignore", BotorchWarning)
np.set_printoptions(linewidth=np.inf)


# Testing parameters
input_dict = get_arguments()
objective = input_dict['f']
MAX_EVALS = input_dict['max_evals']
seed = input_dict['seed']
output_folder = input_dict['output']
popsize = input_dict['popsize']
w = input_dict['w']
c1 = input_dict['c1']
c2 = input_dict['c2']
acq = input_dict['acq']
n_init = 20
# does not matter if BaxusBehavior.adjust_initial_target_dim is True
initital_target_dim = 1

os.makedirs(output_folder, exist_ok=True)
logging.basicConfig(level=logging.INFO)
logging.info(
    f'Start BOIDS run: {objective.name}, seed={seed}, max_evals={MAX_EVALS}')
logging.info(
    f'Swarm settings: popsize={popsize}, w={w:.4f}, c1={c1:.4f}, c2={c2:.4f}')
logging.info(f'Start time: {datetime.datetime.now()}')

if seed is not None:
    set_seed(seed=seed)
filename = os.path.join(output_folder, f'boids_{objective.name}_{seed}')
kwargs = {'output_file': filename}

stamp1 = time.time()
boids = BOIDS(
    max_evals=MAX_EVALS,
    n_init=n_init,
    f=objective,
    target_dim=initital_target_dim,
    behavior=BaxusBehavior(),
    popsize=popsize,
    acq=acq,
    w=w,
    c1=c1,
    c2=c2,
    **kwargs,
)

boids.optimize()
stamp2 = time.time()

x_raw, y_raw = boids.optimization_results_raw()

logging.info(
    f'***FINISHED: Elapsed time: {str(datetime.timedelta(seconds=stamp2 - stamp1))}')
