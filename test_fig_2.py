import random
import matplotlib.pyplot as plt
import numpy as np
import model

import pandas as pd

import pickle

from stats import *
from rw import *
from concurrent.futures import ThreadPoolExecutor


if __name__ == "__main__":


    step_size = 0.1          
    max_time = int(10 / step_size)
    init_num = (20, 2)
    prob_fugal2 = 1

    
    data = {}
    param_list = [ (round(b, 2), round(-round(l, 2) + round(b, 2), 2)) for b in np.arange(0.1, 0.21, 0.01) for l in np.arange(-0.10, 0.11, 0.01) ]
    print(len(param_list))

    for j, key in enumerate(param_list):
            rb, ra = key
            print(j, ' out of ', len(param_list))
            data[key] = run_multiple_trials(rb, ra, max_time=max_time, prob_fugal = prob_fugal2, step_size = step_size, init_num = init_num)
        
            

    with open('sim_data_fig_1.pkl', 'wb') as f:
        pickle.dump(data, f)

    print('done!')
