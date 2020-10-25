#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 25 20:27:09 2020
Here are our own helper functions
Including functions used in data process and cross validation

@author: jiaanzhu, leiwang, qinyuezheng
"""
import numpy as np
import matplotlib.pyplot as plt


'''
Helper functions
'''
def standardize(x):
    """Standardize the original data set."""
    mean_x = np.mean(x, axis=0)
    x = x - mean_x
    std_x = np.std(x, axis=0)
    x = x / std_x
    return x

def normalize(x):
    """Normalize the original data set."""
    x = (x - np.amin(x, axis=0)) / (np.amax(x, axis=0) - np.min(x, axis=0))
    return x

def iqr(tx, y, ind):
    '''
    IQR method, remove outliers
    '''
    tx_new = tx # Make a copy

    q25, q75 = np.nanpercentile(tx_new, 25, axis=0), np.nanpercentile(tx_new, 75, axis=0)
    iqr = q75 - q25
    cut_off = iqr * ind # Remove extreme value 
    lower, upper = q25 - cut_off, q75 + cut_off

    # Compare each row with lower & upper, for those < lower or > upper, record in outlier_index
    outlier_index = []
    for i in range(tx_new.shape[0]):
        if np.any(tx_new[i, :] < lower) or np.any(tx_new[i, :] > upper):
            outlier_index.append(i)

    # Remove outlier
    tx = np.delete(tx, outlier_index, axis = 0)
    y = np.delete(y, outlier_index, axis = 0)
    return tx, y

def build_poly(x, degree):
    """polynomial basis functions for input data x, for j=0 up to j=degree."""
    # this function return the matrix formed
    # by applying the polynomial basis to the input data
    x2 = np.transpose(np.matrix(x))
    x_poly = np.ones((x2.shape[0], 1))
    for i in range(1, degree+1):
        x_poly = np.append(x_poly, np.power(x2, i),  axis=1)
    return x_poly

def split_data(x, y, ratio, seed=1):
    """
    split the dataset based on the split ratio. If ratio is 0.8 
    you will have 80% of your data set dedicated to training 
    and the rest dedicated to testing
    """
    # set seed
    np.random.seed(seed)
    # split the data based on the given ratio
    num = int(len(x) * ratio)
    indices = np.random.permutation(len(x))
    training_idx, test_idx = indices[:num], indices[num:]
    
    x1, y1 = x[training_idx], y[training_idx]
    x2, y2 = x[test_idx], y[test_idx]
    
    return x1, y1, x2, y2

'''
Cross validation for determining ind in IQR
'''
def build_k_indices(y, k_fold, seed):
    """build k indices for k-fold."""
    num_row = y.shape[0]
    interval = int(num_row / k_fold)
    np.random.seed(seed)
    indices = np.random.permutation(num_row)
    k_indices = [indices[k * interval: (k + 1) * interval]
                 for k in range(k_fold)]
    return np.array(k_indices)

def check_ind_visualization(inds, te, tr):
    """visualization the curves of te and mse_tr."""
    plt.plot(inds, te, marker=".", color='b', label='Accuracy')
    plt.plot(inds, tr, marker=".", color='r', label='Accuracy')
    plt.xlabel("iqr index")
    plt.ylabel("Accuracy")
    plt.title("cross validation for iqr")
    plt.legend(loc=2)
    plt.grid(True)
    plt.savefig("cross_validation")
    
def cross_validation_idx(y, x, k_indices, k, lambda_, degree, ind, threshold = 0):
    """Cross validation for ind."""
    # Split train set into new train set and test set
    # get k'th subgroup in test, others in train:
    x_te = x[k_indices[k], :]
    x_tr = x[np.squeeze(np.array(np.delete(k_indices, k, 0))).flatten(), :]
    y_te = y[k_indices[k]]
    y_tr = y[np.squeeze(np.array(np.delete(k_indices, k, 0))).flatten()]
    # Get the number of sample in test set 
    size = y_te.shape[0]
    
    # Remove outliers in train set
    x_tr, y_tr = iqr(x_tr, y_tr, ind)
    # Get the number of sample in train set 
    size2 = y_tr.shape[0]
    # form data with polynomial degree             
    x_tr_poly = build_poly(np.transpose(x_tr), degree)
    x_te_poly = build_poly(np.transpose(x_te), degree)
    
    # ridge regression
    mse, weights = ridge_regression(np.transpose(np.matrix(y_tr)), x_tr_poly, lambda_)
    
    # Predict test set
    y_pred = predict_labels(weights, x_te_poly, threshold)
    # Predict train set
    y_pred2 = predict_labels(weights, x_tr_poly, threshold)
    
    y_te = np.expand_dims(y_te, axis=1)
    y_tr = np.expand_dims(y_tr, axis=1)
    # return accuracy of test set and accuracy of train set
    return sum(y_pred == y_te)/size, sum(y_pred2 == y_tr)/size2

def check_ind_demo(x, y):
    # set seed
    seeds = range(100)
    # here degree is a constant
    degree = 1
    # here lambda is a constant
    lambda_ = 0
    # set inds
    inds = range(1, 33, 2)
    # For data storage
    mean_accu = np.empty((len(seeds), len(inds)))
    mean_accu2 = np.empty((len(seeds), len(inds)))
    
    for seed in seeds:
        k_fold = 4
        k_indices = build_k_indices(y, k_fold, seed)
        for index_ind, ind in enumerate(inds):
            accu = []
            accu2 = []
            for k in range(k_fold):            
                te, tr = cross_validation_idx(y, x, k_indices, k, lambda_, degree, ind, 0)
                accu.append(te)
                accu2.append(tr)
            mean_accu[seed, index_ind] = np.mean(accu)
            mean_accu2[seed, index_ind] = np.mean(accu2)
    # calculate average of each column
    mean_accu_av = np.mean(mean_accu, axis = 0)
    mean_accu2_av = np.mean(mean_accu2, axis = 0)
    # Visualization
    check_ind_visualization(inds, mean_accu_av, mean_accu2_av)
    # Looking for the minimum difference bewteen two set
    dist = mean_accu2_av - mean_accu_av
    mini = min(dist)
    print(mini, np.where(dist == mini))
    
'''
Cross validation for determining degree in data augment
'''
def check_degree_visualization(degrees, te, tr):
    """visualization the curves of tr and te."""
    plt.plot(degrees, te, marker=".", color='b', label='Accuracy')
    plt.plot(degrees, tr, marker=".", color='r', label='Accuracy')
    plt.xlabel("degree")
    plt.ylabel("Accuracy")
    plt.title("cross validation for degree")
    plt.legend(loc=2)
    plt.grid(True)
    plt.savefig("cross_validation")

    
def cross_validation_degree(y, x, k_indices, k, lambda_, degree, ind, threshold = 0):
    """Cross validation for degree."""
    # Split train set into new train set and test set
    # get k'th subgroup in test, others in train:
    x_te = x[k_indices[k], :]
    x_tr = x[np.squeeze(np.array(np.delete(k_indices, k, 0))).flatten(), :]
    y_te = y[k_indices[k]]
    y_tr = y[np.squeeze(np.array(np.delete(k_indices, k, 0))).flatten()]
    # Get the number of sample in test set 
    size = y_te.shape[0]
    # Remove outliers in train set
    x_tr, y_tr = iqr(x_tr, y_tr, ind)
    # Get the number of sample in train set 
    size2 = y_tr.shape[0]
    # form data with polynomial degree                   
    x_tr_poly = build_poly(np.transpose(x_tr), degree)
    x_te_poly = build_poly(np.transpose(x_te), degree)
    
    # ridge regression
    mse, weights = ridge_regression(np.transpose(np.matrix(y_tr)), x_tr_poly, lambda_)
    # Predict test set
    y_pred = predict_labels(weights, x_te_poly, threshold)
    # Predict train set
    y_pred2 = predict_labels(weights, x_tr_poly, threshold)
    
    y_te = np.expand_dims(y_te, axis=1)
    y_tr = np.expand_dims(y_tr, axis=1)
    # return accuracy of test set and accuracy of train set
    return sum(y_pred == y_te)/size, sum(y_pred2 == y_tr)/size2

def check_degree_demo(x, y, ind):
    '''here ind is from our check_ind_demo result'''
    # set seed
    seeds = range(100)
    # here lambda is a constant
    lambda_ = 0
    # set k_fold
    k_fold = 4
    # set degree
    degrees = range(1, 12)
    # For data storage
    mean_accu = np.empty((len(seeds), len(degrees)))
    mean_accu2 = np.empty((len(seeds), len(degrees)))
    
    for seed in seeds:
        # Generate indexs for four groups
        k_indices = build_k_indices(y, k_fold, seed)
        for degree in degrees:
            accu = []
            accu2 = []
            for k in range(k_fold):            
                te, tr = cross_validation_degree(y, x, k_indices, k, lambda_, degree, ind, 0)
                accu.append(te)
                accu2.append(tr)
            mean_accu[seed, degree-1] = np.mean(accu)
            mean_accu2[seed, degree-1] = np.mean(accu2)
    # calculate average of each column
    mean_accu_av = np.mean(mean_accu, axis = 0)
    mean_accu2_av = np.mean(mean_accu2, axis = 0)
    # Visualization
    check_degree_visualization(degrees, mean_accu_av, mean_accu2_av)
    # show maixmum accuracy in test set and its index
    maxi = max(mean_accu_av)
    print(maxi, np.where(mean_accu_av == maxi))

'''
Cross validation for determining lambda in data augment
'''
def check_lambda_visualization(lambds, te, tr):
    """visualization the curves of te and tr."""
    plt.semilogx(lambds, te, marker=".", color='b', label='Accuracy')
    plt.semilogx(lambds, tr, marker=".", color='r', label='Accuracy')
    plt.xlabel("lambda")
    plt.ylabel("Accuracy")
    plt.title("cross validation of lambda")
    plt.legend(loc=2)
    plt.grid(True)
    plt.savefig("cross_validation")

def cross_lambda_validation(y, x, k_indices, k, lambda_, degree, ind, threshold = 0):
    """Cross validation for lambda."""
    # Split train set into new train set and test set
    # get k'th subgroup in test, others in train:
    x_te = x[k_indices[k], :]
    x_tr = x[np.squeeze(np.array(np.delete(k_indices, k, 0))).flatten(), :]
    y_te = y[k_indices[k]]
    y_tr = y[np.squeeze(np.array(np.delete(k_indices, k, 0))).flatten()]
    # Get the number of sample in test set
    size = y_te.shape[0]
    # Remove outliers in train set
    x_tr, y_tr = iqr(x_tr, y_tr, ind)
    # Get the number of sample in train set
    size2 = y_tr.shape[0] 
    
    # form data with polynomial degree                
    x_tr_poly = build_poly(np.transpose(x_tr), degree)
    x_te_poly = build_poly(np.transpose(x_te), degree)
    # ridge regression
    mse, weights = ridge_regression(np.transpose(np.matrix(y_tr)), x_tr_poly, lambda_)      
    # Predict test set
    y_pred = predict_labels(weights, x_te_poly, threshold)
    # Predict train set
    y_pred2 = predict_labels(weights, x_tr_poly, threshold)
    
    y_te = np.expand_dims(y_te, axis=1)
    y_tr = np.expand_dims(y_tr, axis=1)
    # return accuracy of test set and accuracy of train set
    return sum(y_pred == y_te)/size, sum(y_pred2 == y_tr)/size2

def check_lambda_demo(x, y, ind, degree):
    '''here ind is from our check_ind_demo result, here degree is from our check_degree_demo'''
    # set seed
    seeds = range(100)
    # set lambda
    lambdas = np.logspace(-10, 5, 16)
    # For data storage
    mean_accu = np.empty((len(seeds), len(lambdas)))
    mean_accu2 = np.empty((len(seeds), len(lambdas)))
    
    for seed in seeds:
        # Generate indexs for four groups
        k_fold = 4
        k_indices = build_k_indices(y, k_fold, seed)
        
        for index_lambda, lambda_ in enumerate(lambdas):
            accu = []
            accu2 = []
            for k in range(k_fold):            
                accuracy = cross_lambda_validation(y, x, k_indices, k, lambda_, degree, ind, 0)
                accu.append(accuracy)
                accu2.append(accuracy)
            mean_accu[seed, index_lambda] = np.mean(accu)
            mean_accu2[seed, index_lambda] = np.mean(accu2)
    # calculate average of each column
    mean_accu_av = np.mean(mean_accu, axis = 0)
    mean_accu2_av = np.mean(mean_accu2, axis = 0)
    # Visualization
    check_lambda_visualization(lambdas, mean_accu_av, mean_accu2_av)
    # show maixmum accuracy in test set and its index
    maxi = max(mean_accu_av)
    print(maxi, np.where(mean_accu_av == maxi))

