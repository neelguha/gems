# Perform GEMS for a deep neural network with two layers 

import sys 
import numpy as np
import os
from tqdm import tqdm_notebook
import argparse
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import KMeans

from keras.models import Sequential, load_model
from keras.layers import Dense, Activation, Flatten
from keras.utils import to_categorical 
import keras

from .data_utils import load_data, flatten
from .model_utils import *
from .epsilon_space import compute_fisher, SingleSphere, single_sphere_intersection, multiple_sphere_intersection, check_intersection

from .neuron_act_sampling import *
from .convex_sampling import *


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

def train_from_scratch(Xval_flat, yval_flat, Xts_flat, yts_flat, exp, holdout_size):
    indices = np.random.choice(len(yval_flat), holdout_size, replace = False)
    holdout_x, holdout_y = Xval_flat[indices], yval_flat[indices]
    m = learn_dnn2(holdout_x, holdout_y, Xts_flat, yts_flat, exp, epochs=exp['global_epochs'], verbose=0)
    loss, acc = m.evaluate(Xts_flat, yts_flat, verbose =0)
    return loss, acc 


def run_dnn_gems(exp, Xtr, ytr, Xval, yval, Xts, yts, models):

    # Experimental parameters
    fisher_samples = exp['fisher_count']
    num_samples = exp['num_samples'] 
    epsilon = exp['final_epsilon']
    num_hidden = exp['hidden']
    max_radius = exp['max_radius']
    num_tasks = exp['num_nodes']
    k = exp['k']

    Xts_flat, yts_flat = flatten(Xts, yts)


    # Calculate fisher information matrix for each model
    fisher_arrays = []
    for t in range(num_tasks):
        fisher_arrays.append(compute_fisher(models[t], Xval[t], num_samples=fisher_samples))

    ## First layer intersection 
    slack = exp['hidden_slack']
    X = []
    spaces = []
    print("Calculating good-enough spaces.")
    for i in tqdm_notebook(range(num_hidden)):
        for t in range(num_tasks):
            n = ConcenActNeuronSphereSampler(
                models[t], Xval[t], yval[t], fisher_arrays[t], 
                layer_index = 0, neuron_index = i, slack = slack, exp = exp)
            n.sample_weight_space(num_samples, verbose=False)
            spaces.append(n)
            X.append(np.concatenate([x.ravel() for x in n.orig_weights]))

    kmeans = KMeans(n_clusters=k).fit(X)
            

    # iterate through each neuron
    assignments = kmeans.labels_
    gems_w = []
    gems_b = []
    intersection_count = []
    for cluster in range(1,k+1):
        indices = np.arange(num_hidden*num_tasks)[assignments == cluster]
        ep_spaces = []
        # Check whether intersection exists 
        for i in indices:
            intersection = True
            for j in ep_spaces:
                if not check_intersection(spaces[i], spaces[j]):
                    intersection = False
            if intersection:
                ep_spaces.append(i)
            else:
                # append to gems_w, gems_b
                w = spaces[i].orig_weights[0]
                b = spaces[i].orig_weights[1]
                intersection_count.append(1)
                gems_w.append(w)
                gems_b.append(b)
        if len(ep_spaces) > 1:
            gems_neuron, _ = multiple_sphere_intersection([spaces[l] for l in ep_spaces])
            w = gems_neuron[0]
            b = gems_neuron[1]
            intersection_count.append(len(ep_spaces))
            gems_w.append(w)
            gems_b.append(b)
        elif len(ep_spaces) == 1:
            w = spaces[ep_spaces[0]].orig_weights[0]
            b = spaces[ep_spaces[0]].orig_weights[1]
            intersection_count.append(1)
            gems_w.append(w)
            gems_b.append(b)
        
    gems_w = np.array(gems_w).transpose()
    gems_b = np.array(gems_b)
    num_hidden = gems_w.shape[1]
    print("Number of hidden neurons: %d" % num_hidden )

    # Transform data based on constructed hidden layer
    Xtr_transform = generate_new_data(gems_w, gems_b, Xtr)
    Xval_transform = generate_new_data(gems_w, gems_b, Xval)
    Xts_transform = generate_new_data(gems_w, gems_b, Xts)
    Xts_flat_transform, yts_flat = flatten(Xts_transform, yts)

    # Train new logit models on transformed datasets 
    print("Training new logistic models")
    epochs = exp['secondary_epochs']
    logit_models = []
    for t in range(num_tasks):
        logit_models.append(learn_logit(Xtr_transform[t], ytr[t], Xts_transform[t], yts[t], exp, epochs = epochs, verbose = 0))

    # Calculate fisher information matrix for each model
    fisher_arrays = []
    for t in range(num_tasks):
        fisher_arrays.append(compute_fisher(logit_models[t], Xval_transform[t], num_samples=fisher_samples))


    final_spaces = []
    for t in range(num_tasks):
        s = ConcenSphereSampler(logit_models[t], Xval_transform[t], yval[t], fisher_arrays[t], epsilon, exp)
        s.sample_weight_space(num_samples, max_radius = max_radius, verbose=False)
        final_spaces.append(s)     

    # Find intersection between epsilon spaces 
    intersected_weights, intersection = multiple_sphere_intersection(final_spaces)
    gems_model = create_logit(intersected_weights, Xts_flat_transform, yts_flat, exp)
    loss, acc = gems_model.evaluate(Xts_flat_transform, yts_flat, verbose=0)

    print("Pure GEMS accuracy: %f" % acc)
    results = {
        'loss': loss, 
        'acc': acc,
        'intersection': intersection,
        'w': gems_w,
        'b': gems_b,
    }

    
    # Flatten data 
    Xval_flat_transform, yval_flat = flatten(Xval_transform, yval)
    Xts_flat_transform, yts_flat = flatten(Xts_transform, yts)
    Xval_flat, yval_flat = flatten(Xval, yval)

    # Get average model 
    _, avg_model = average_models(Xts_flat, yts_flat, models, exp)
    
    # Fine tuning on holdout validation data 
    holdout_sizes = exp['holdout_sizes']
    holdout_trials = exp['holdout_trials']
    
    gems_acc_scores = []
    raw_acc_scores = []
    avg_raw_scores = []
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
            _, gems_acc = fine_tune(gems_model, Xval_flat_transform, yval_flat, Xts_flat_transform, yts_flat, exp, holdout_size)
            gems_trial_acc.append(gems_acc)

            _, avg_acc = fine_tune(avg_model, Xval_flat, yval_flat, Xts_flat, yts_flat, exp, holdout_size)
            avg_trial_acc.append(avg_acc)

            _, raw_acc = train_from_scratch(Xval_flat, yval_flat, Xts_flat, yts_flat, exp, holdout_size)
            raw_trial_acc.append(raw_acc)

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
        avg_raw_scores.append(avg_trial_acc)
    
    holdout_results = {
        'holdout_sizes': holdout_sizes, 
        'gems_acc': gems_acc_scores,
        'raw_acc': raw_acc_scores,
        'avg_acc': avg_raw_scores
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