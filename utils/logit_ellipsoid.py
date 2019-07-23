# Learn ellipsoid epsilon space and find intersection for logistic model
import sys 
import numpy as np
import os
from tqdm import tqdm
import argparse
from scipy.optimize import linear_sum_assignment

from keras.models import Sequential, load_model
from keras.layers import Dense, Activation, Flatten
from keras.utils import to_categorical 
import keras

from .data_utils import load_data, flatten
from .model_utils import *
from .epsilon_space import compute_fisher

from .ellipsoid_sampler import *


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

def fine_tune(gems_model, Xval_flat, yval_flat, Xts_flat, yts_flat, exp, holdout_size):
    """ Run finetuning experiment 
    """
    orig_gems_weights = gems_model.get_weights()
    indices = np.random.choice(len(yval_flat), holdout_size, replace = False)
    holdout_x, holdout_y = Xval_flat[indices], yval_flat[indices]
    gems_model.fit(holdout_x, holdout_y, validation_data=(Xts_flat, yts_flat), epochs=5, verbose=0)
    loss, acc = gems_model.evaluate(Xts_flat, yts_flat, verbose =0)
    gems_model.set_weights(orig_gems_weights)
    return loss, acc  

def train_from_scratch(Xval_flat, yval_flat, Xts_flat, yts_flat, exp, holdout_size):
    indices = np.random.choice(len(yval_flat), holdout_size, replace = False)
    holdout_x, holdout_y = Xval_flat[indices], yval_flat[indices]
    m = learn_logit(holdout_x, holdout_y, Xts_flat, yts_flat, exp, epochs=5, verbose=0)
    loss, acc = m.evaluate(Xts_flat, yts_flat, verbose =0)
    return loss, acc 

def run_logit_ellipsoid(exp, Xtr, ytr, Xval, yval, Xts, yts, models):
    fisher_samples = exp['fisher_count']
    num_samples = exp['num_samples'] # number of weight sampels to draw 
    epsilon = exp['final_epsilon']
    max_radius = exp['max_radius']
    num_tasks = exp['num_nodes']

    # Get flattened test data 
    Xts_flat, yts_flat = flatten(Xts, yts)
    Xval_flat, yval_flat = flatten(Xval, yval)

    # Calculate fisher information matrix for each model
    fisher_arr = []
    for t in range(num_tasks):
        print("Computing fisher information matrix for node-%d" % t)
        fisher_arr.append(compute_fisher(models[t], Xval[t], fisher_samples))
    
    # calculate epsilon spaces
    spaces = []
    for t in range(num_tasks):
        print("Computing good-enough model space for node-%d" % t)
        s = EllipsoidSampler(models[t], Xval[t], yval[t], fisher_arr[t], exp)
        s.sample_weight_space(num_samples, max_radius = max_radius, verbose=False)
        print("Node-%d. Radius: %f" % (t, s.ep_radius))     
        spaces.append(s)
    
    # Find intersection between epsilon spaces 
    intersected_weights, intersection = ellipsoid_intersection(spaces)
    gems_model = create_logit(intersected_weights, Xts_flat, yts_flat, exp)
    loss, acc = gems_model.evaluate(Xts_flat, yts_flat, verbose=0)
    print("GEMS accuracy: %f" % acc)
    results = {
        'loss': loss, 
        'acc': acc,
        'intersection': intersection,
    }
    
    # Get average model 
    _, avg_model = average_models(Xts_flat, yts_flat, models, exp)

    # Fine tunining on holdout validation data 
    holdout_sizes = exp['holdout_sizes']
    holdout_trials = exp['holdout_trials']
    gems_acc_scores = []
    raw_acc_scores = []
    avg_acc_scores = []
    loc_acc_scores = {}
    for t in range(num_tasks):
        loc_acc_scores[t] = []
    
    for holdout_size in holdout_sizes:
        gems_trial_acc = []
        raw_trial_acc = []
        avg_trial_acc = []
        loc_trial_acc = {}
        for t in range(num_tasks):
            loc_trial_acc[t] = []
        for _ in range(holdout_trials):
            _, gems_acc = fine_tune(gems_model, Xval_flat, yval_flat, Xts_flat, yts_flat, exp, holdout_size)
            gems_trial_acc.append(gems_acc)

            _, raw_acc = train_from_scratch(Xval_flat, yval_flat, Xts_flat, yts_flat, exp, holdout_size)
            raw_trial_acc.append(raw_acc)

            _, avg_acc = fine_tune(avg_model, Xval_flat, yval_flat, Xts_flat, yts_flat, exp, holdout_size)
            avg_trial_acc.append(avg_acc)

            for t in range(num_tasks):
                _, loc_acc = fine_tune(models[t], Xval_flat, yval_flat, Xts_flat, yts_flat, exp, holdout_size)
                loc_trial_acc[t].append(loc_acc)

        print("Holdout Size: %d." % (holdout_size))
        print("\tRaw: %f." % np.mean(raw_trial_acc))
        print("\tGems (Tuned): %f." % np.mean(gems_trial_acc))
        print("\tAverage (Tuned): %f." % np.mean(avg_trial_acc))
        for t in range(num_tasks):
            print("\t Node-%d Loc Model (Tuned): %f" % (t, np.mean(loc_trial_acc[t])))
            loc_acc_scores[t].append(loc_trial_acc[t])

        gems_acc_scores.append(gems_trial_acc)
        raw_acc_scores.append(raw_trial_acc) 
        avg_acc_scores.append(avg_trial_acc)
    
    holdout_results = {
        'holdout_sizes': holdout_sizes, 
        'gems_acc': gems_acc_scores,
        'raw_acc': raw_acc_scores,
        'avg_acc': avg_acc_scores
    } 
    for t in range(num_tasks):
        key = 'loc_%d' % t
        holdout_results[key] = loc_acc_scores[t]

    results['holdout'] = holdout_results
    return gems_model, results

def parse_arguments(parser):
    parser.add_argument('-e', '--exp_id',
                        help='path to config file for experiment',
                        type=str,
                        default="test")
    parser.add_argument('-d', '--dataset',
                        help='name of dataset',
                        type=str,
                        default=0)
    parser.add_argument('-n', '--num_tasks',
                        help='number of tasks',
                        type=int,
                        default=2)   
    parser.add_argument('--verbose', help='Print more data', 
                        action='store_true')                        
    return parser.parse_args()

    