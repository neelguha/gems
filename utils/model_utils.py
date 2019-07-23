import numpy as np
import os
import timeit
import shutil
import subprocess as sp

from .data_utils import *

from sklearn.metrics import accuracy_score, roc_auc_score, make_scorer
from sklearn.model_selection import GridSearchCV

from keras.models import Sequential, load_model
from keras.layers import Dense, Activation, Flatten, Dropout
from keras import regularizers
from keras.wrappers.scikit_learn import KerasClassifier
from keras import backend as K
from keras.optimizers import RMSprop, Adam
from keras import optimizers
from keras.callbacks import EarlyStopping


#######################################
# Finetuning functions 
#######################################
def fine_tune(model, Xval_flat, yval_flat, Xts_flat, yts_flat, exp, holdout_size):
    """ Run finetuning on holdout dataset.   
    """
    # store original weights 
    orig_weights = model.get_weights()

    # Select holdout data to train on 
    indices = np.random.choice(len(yval_flat), holdout_size, replace = False)
    holdout_x, holdout_y = Xval_flat[indices], yval_flat[indices]

    # Perform finetuning
    ft_epochs = 5
    if 'finetuning_epochs' in exp.keys():
        ft_epochs = exp['finetuning_epochs']
    model.fit(holdout_x, holdout_y, validation_data=(Xts_flat, yts_flat), epochs=ft_epochs, verbose=0)
    loss, acc = model.evaluate(Xts_flat, yts_flat, verbose =0)

    # Reset model.
    model.set_weights(orig_weights)
    return loss, acc  

#######################################
# Model training functions 
#######################################

def learn_logit(Xtr, ytr, Xts, yts, exp, epochs, verbose=0):
    """ Learns logistic regression model on passed train data 

    # Arguments: 
        Xtr: train data for specific agent
        ytr: train labels for specific agent 
        Xts: flattened test data over all agents
        yts: flattened test data over all agents 
        secondary: whether this logit model corresponds to the second layer of a DNN
        epochs: number of epochs to train for 
        verbose: level of verbosity
    # Returns: 
        trained model
    """
    input_dim = Xtr.shape[1]
    output_dim = 1 if len(ytr.shape) == 1 else ytr.shape[1]
    optimizer_name = exp['optimizer']
    loss = exp['loss']

    regularization = False 
    if 'regularize' in exp.keys():
        regularization = exp['regularize']

    model = Sequential()
    if regularization:
        print("REGULARIZING")
        model.add(Dense(units=output_dim, input_dim=input_dim, 
                    activation='softmax', kernel_initializer='normal', 
                    kernel_regularizer=regularizers.l2(0.005),
                    use_bias=True))
    else:
        model.add(Dense(units=output_dim, input_dim=input_dim, 
                    activation='softmax', kernel_initializer='normal', 
                    use_bias=True))
    es = EarlyStopping(monitor='acc', mode='max', verbose=0, patience=5)
    if optimizer_name == 'sgd':
        optimizer = optimizers.SGD(lr = 0.001)
    elif optimizer_name == 'adam':
        optimizer = optimizers.Adam(lr = 0.001)
    else: 
        raise NotImplementedError(optimizer_name)

    model.compile(optimizer='adam', loss=loss, metrics=['accuracy'])
    
    model.fit(Xtr, ytr, epochs=epochs, validation_data=(Xts, yts), verbose=verbose, callbacks=[es])
    return model

def create_logit(vars, Xts, yts, exp, verbose = 0):
    """ Creates model with weights in vars. Uses Xts, yts to set size. Returns model.
    """
    
    input_dim = Xts.shape[1]
    output_dim = 1 if len(yts.shape) == 1 else yts.shape[1]
    optimizer_name = exp['optimizer']
    loss = exp['loss']

    model = Sequential()
    model.add(Dense(units=output_dim, input_dim=input_dim, activation='softmax', kernel_initializer='normal', use_bias=True))

    if optimizer_name == 'sgd':
        optimizer = optimizers.SGD(lr = 0.01)
    elif optimizer_name == 'adam':
        optimizer = optimizers.Adam(lr = 0.01)
    else: 
        raise NotImplementedError(optimizer_name)

    model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])

    model.evaluate(Xts, yts, verbose=0)
    model.set_weights(vars)
    
    return model

def learn_dnn2(Xtr, ytr, Xts, yts, exp, epochs, verbose=0):
    """Learns two layer neural network on passed train data 

    # Arguments: 
        Xtr: train data for specific agent
        ytr: train labels for specific agent 
        Xts: flattened test data over all agents
        yts: flattened test data over all agents 
        epochs: number of epochs to train for 
        verbose: level of verbosity
    # Returns: 
        trained model
    """
    input_dim = len(Xtr[0])
    output_dim = 1 if len(ytr.shape) == 1 else ytr.shape[1]
    input_dim = Xtr.shape[1]
    output_dim = 1 if len(ytr.shape) == 1 else ytr.shape[1]
    optimizer_name = exp['optimizer']
    loss = exp['loss']
    hidden = exp['hidden']
    dropout = exp['dropout'] # Note - dropout is not supposed to do anything when equal to 0

    model = Sequential()
    model.add(Dense(hidden, activation='relu', input_shape=(input_dim,)))
    model.add(Dropout(dropout))
    model.add(Dense(output_dim, activation='softmax'))
    es = EarlyStopping(monitor='acc', mode='max', verbose=0, patience=3)
    model.compile(loss=loss, optimizer='adam', metrics=['accuracy'])
    
    model.fit(Xtr, ytr, epochs=epochs, validation_data=(Xts, yts), verbose=verbose, callbacks=[es])
    return model

def create_dnn2(vars, Xts, yts, exp):
    """
        Creates model with weights in vars. Uses Xts, yts to set size. Returns model.
    """
    
    input_dim = Xts.shape[1]
    output_dim = 1 if len(yts.shape) == 1 else yts.shape[1]
    optimizer_name = exp['optimizer']
    loss = exp['loss']
    hidden = exp['hidden']
    dropout = exp['dropout'] # Note - dropout is not supposed to do anything when equal to 0

    model = Sequential()
    model.add(Dense(hidden, activation='relu', input_shape=(input_dim,)))
    model.add(Dropout(dropout))
    model.add(Dense(output_dim, activation='softmax'))

    if optimizer_name == 'sgd':
        optimizer = optimizers.SGD(lr = 0.01)
    elif optimizer_name == 'adam':
        optimizer = optimizers.Adam(lr = 0.01)
    else: 
        raise NotImplementedError(optimizer_name)
    model.layers[0].trainable = False
    model.compile(loss=loss, optimizer=optimizer, metrics=['accuracy'])   

    model.evaluate(Xts, yts, verbose=0)
    model.set_weights(vars)
    return model

def get_model_func(exp):
    if exp['model'] == 'logit':
        return learn_logit
    elif exp['model'] == 'logitInterp':
        return learn_logit
    elif exp['model'] == 'dnn2':
        return learn_dnn2
    else:
        raise NotImplementedError("Unknown model type")


def generate_new_data(w, b, X):
    """Apply ReLU + linear transformation to each data sample. Specifically:   
            
    # Returns: 
        maximum(0, Xw^T + b)
    """
    num_tasks = len(X)
    X_transform = np.empty(shape=(num_tasks), dtype=object)
    for i in range(num_tasks):
        X_transform[i] = np.maximum(np.dot(X[i], w) + b, 0)
    return X_transform


def average_models(Xts_flat, yts_flat, models, exp):
    """ Return score of with averaged weights
    """
    
    mvars = models[0].get_weights()
    for model in models[1:]:
        weights = model.get_weights()
        for i in range(len(mvars)):
            mvars[i] += weights[i]
    
    for i in range(len(mvars)):
        mvars[i] = mvars[i] / len(models)

    model_type = exp['model']
    if model_type == 'dnn2':
        avg_model = create_dnn2(mvars, Xts_flat, yts_flat, exp)
    elif model_type == 'logit':
        avg_model = create_logit(mvars, Xts_flat, yts_flat, exp)
    
    loss, acc = avg_model.evaluate(Xts_flat, yts_flat, verbose=0)
    avg_results = {
        "avg_loss": loss, 
        "avg_acc": acc,
    }
    return avg_results , avg_model

def get_ensemble_scores(Xts_flat, yts_flat, models):
    preds = models[0].predict(Xts_flat)
    for model in models[1:]:
        preds = preds + np.round(model.predict(Xts_flat))
    
    # add some small noise to break ties when taking argmax
    preds = np.random.rand(*preds.shape)*0.1 - 0.05 + preds
    acc = np.mean(np.argmax(preds, axis=1) == np.argmax(yts_flat, axis=1))
    results = {
        'ensemble_acc': acc
    }
    return results


def train_local_model(Xtr, ytr, Xts, yts, t,  exp, verbose = 0):
    """Trains model on {Xtr[t], ytr[t]}, and evaluates on all of {Xts, yts}.
        # Arguments
    
            Xtr: train X, partitioned by agent  
            ytr: train y, partitioned by agent  
            Xt:  test X, partitioned by agent   
            yts: test y, partitioned by agent  
            t: agent index   
            exp: experiment config dictionary, contains params for traininng local model   
        
        # Returns
            model: trained keras model object 
            results: dictionary containinng local and global loss/accuracy for model 
    """
    
    Xts_flt, yts_flt = flatten(Xts, yts)
    
    model_func = get_model_func(exp)
    epochs = exp['local_epochs']
    model = model_func(Xtr[t], ytr[t], Xts[t], yts[t],  exp, epochs = epochs, verbose= verbose)
    local_loss, local_accuracy = model.evaluate(Xts[t], yts[t], verbose =0)
    glob_loss, glob_accuracy = model.evaluate(Xts_flt, yts_flt, verbose= 0)
    results = {}
    results['local_accuracy'] = local_accuracy
    results['local_loss'] = local_loss
    results['global_accuracy'] = glob_accuracy
    results['global_loss'] = glob_loss
    return model, results
