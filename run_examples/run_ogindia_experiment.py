'''
Example script for setting policy and running OG-India
'''

# import modules
import multiprocessing
from dask.distributed import Client
import time
import numpy as np

from taxcalc import Calculator
from ogindia import postprocess
from ogindia.execute import runner
from ogindia.utils import REFORM_DIR, BASELINE_DIR


def run_micro_macro(user_params):
    # Define parameters to use for multiprocessing
    client = Client(processes=False)
    num_workers = min(multiprocessing.cpu_count(), 7)
    print('Number of workers = ', num_workers)
    run_start_time = time.time()

    # Proposed changes:
    # 1) Increase government spending - increase by 2pp of GDP, starting in 2020 ending after 2022
    # 2) Cut corporate income tax rate to 25%, permanently
    # 3) Increase the standard deduction from 40,000 to 100,000, starting in 2020

    # Specify direct tax refrom
    dt_reform = {2020: {'_std_deduction': [100000]}}

    # Set some model parameters
    # See parameters.py for description of these parameters
    OG_params = {'alpha_G': [0.112, 0.132, 0.132, 0.132, 0.112],  # specify policy from 2019 through 2023- assuming constant thereafter
                 'tau_b': [0.27, 0.25 * 0.27/0.34]}  # note the rate is the effective tax rate

    '''
    ------------------------------------------------------------------------
    Run baseline policy first
    ------------------------------------------------------------------------
    '''
    output_base = BASELINE_DIR
    kwargs = {'output_base': output_base, 'baseline_dir': BASELINE_DIR,
              'test': False, 'time_path': True, 'baseline': True,
              'guid': '_TPRU_19232019', 'run_micro': True,
              'data': 'pitSmallData.csv',
              'client': client, 'num_workers': num_workers}

    start_time = time.time()
    runner(**kwargs)
    print('run time = ', time.time()-start_time)

    '''
    ------------------------------------------------------------------------
    Run reform policy
    ------------------------------------------------------------------------
    '''
    output_base = REFORM_DIR
    kwargs = {'output_base': output_base, 'baseline_dir': BASELINE_DIR,
              'test': False, 'time_path': True, 'baseline': False,
              'user_params': OG_params, 'guid': '_TPRU_19232019_policy',
              'reform': dt_reform, 'run_micro': True,
              'data': 'pitSmallData.csv',
              'client': client, 'num_workers': num_workers}

    start_time = time.time()
    runner(**kwargs)
    print('run time = ', time.time()-start_time)

    # return ans - the percentage changes in macro aggregates and prices
    # due to policy changes from the baseline to the reform
    ans = postprocess.create_diff(
        baseline_dir=BASELINE_DIR, policy_dir=REFORM_DIR)

    print("total time was ", (time.time() - run_start_time))
    print('Percentage changes in aggregates:', ans)


if __name__ == "__main__":
    run_micro_macro(user_params={})
