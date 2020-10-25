#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 27 14:14:49 2020
Here are 6 requried functions

@author: jiaanzhu, leiwang, qinyuezheng
"""
import numpy as np
import matplotlib.pyplot as plt

'''
Gradient descent
'''
def compute_loss(y, tx, w):
    """Calculate the loss.

    You can calculate the loss using mse or mae.
    """
    # ***************************************************
    # For MSE
    return (1/(2 * tx.shape[0]))*np.dot(y - np.dot(tx, w), (y - np.dot(tx, w)))
    # For MAE
    # return (1/tx.shape[0])*np.sum(np.abs((y - np.dot(tx, w))))

def compute_mse(y, tx, w):
    """Calculate the loss.

    You can calculate the loss using mse or mae.
    """
    # For MSE
    e = y - np.dot(tx, w)
    return (1/(2 * tx.shape[0]))*np.dot(np.transpose(e), e)

def compute_gradient(y, tx, w):
    """Compute the gradient."""
    # ***************************************************
    # For MSE
    return (-1/tx.shape[0]) * np.dot(np.transpose(tx), (y - np.dot(tx, w)))
    # For MAE
    # return 1/tx.shape[0] * np.dot(np.transpose(tx), np.sign(np.dot(tx, w) - y))
    # ***************************************************
    
def gradient_descent(y, tx, initial_w, max_iters, gamma):
    """Gradient descent algorithm."""
    # Define parameters to store w and loss
    ws = [initial_w]
    losses = []
    w = initial_w
    for n_iter in range(max_iters):      
        # compute gradient and loss
        loss = compute_loss(y, tx, w)
        gradient = compute_gradient(y, tx, w)
        # update w by gradient
        w = w - gamma*gradient
        # store w and loss
        ws.append(w)
        losses.append(loss)
    loss = losses[-1]
    weight = ws[-1]
    return loss, weight

'''
Stochastic Gradient Descent
'''
def compute_stoch_gradient(y, tx, w):
    """Compute a stochastic gradient from just few examples n and their corresponding y_n labels."""
    for minibatch_y, minibatch_tx in batch_iter(y, tx, 1):
        
        return (-1/minibatch_tx.shape[0]) * np.dot(np.transpose(minibatch_tx), (minibatch_y - np.dot(minibatch_tx, w)))


def stochastic_gradient_descent(
        y, tx, initial_w, max_iters, gamma):
    """Stochastic gradient descent algorithm."""
    # implement stochastic gradient descent.
    ws = [initial_w]
    losses = []
    w = initial_w
    for n_iter in range(max_iters):
        # compute gradient and loss
        loss = compute_loss(y, tx, w)
        gradient = compute_stoch_gradient(y, tx, w)
        # update w by gradient
        w = w - gamma*gradient
        # store w and loss
        ws.append(w)
        losses.append(loss)        
    loss = losses[-1]
    weight = ws[-1]
    return loss, weight

def batch_iter(y, tx, batch_size, num_batches=1, shuffle=True):
    """
    Generate a minibatch iterator for a dataset.
    Takes as input two iterables (here the output desired values 'y' and the input data 'tx')
    Outputs an iterator which gives mini-batches of `batch_size` matching elements from `y` and `tx`.
    Data can be randomly shuffled to avoid ordering in the original data messing with the randomness of the minibatches.
    Example of use :
    for minibatch_y, minibatch_tx in batch_iter(y, tx, 32):
        <DO-SOMETHING>
    """
    data_size = len(y)

    if shuffle:
        shuffle_indices = np.random.permutation(np.arange(data_size))
        shuffled_y = y[shuffle_indices]
        shuffled_tx = tx[shuffle_indices]
    else:
        shuffled_y = y
        shuffled_tx = tx
    for batch_num in range(num_batches):
        start_index = batch_num * batch_size
        end_index = min((batch_num + 1) * batch_size, data_size)
        if start_index != end_index:
            yield shuffled_y[start_index:end_index], shuffled_tx[start_index:end_index]

'''
Least regression
'''        
def least_squares(y, tx):
    """calculate the least squares solution."""
    a = np.dot(np.transpose(tx), tx)
    b = np.dot(np.transpose(tx), y)
    w = np.linalg.solve(a, b)
    e = y - np.dot(tx, w)
    mse = 1/(2*tx.shape[0]) * np.dot(np.transpose(e), e)
    # returns mse, and optimal weights
    return mse, w


def build_poly(x, degree):
    """polynomial basis functions for input data x, for j=0 up to j=degree."""
    # this function return the matrix formed by applying the polynomial basis to the input data
    x2 = np.transpose(np.matrix(x))
    x_poly = np.ones((x2.shape[0], 1))
    for i in range(1, degree+1):
        x_poly = np.append(x_poly, np.power(x2, i),  axis=1)
    return x_poly

'''
Ridge regression
'''            
    
def ridge_regression(y, tx, lambda_):
    """implement ridge regression."""
    a = np.dot(np.transpose(tx), tx) + 2 * tx.shape[0] * lambda_ * np.identity(tx.shape[1])
    b = np.dot(np.transpose(tx), y)
    w = np.linalg.solve(a, b)
    e = y - np.dot(tx, w)
    mse = 1/(2*tx.shape[0]) * np.dot(np.transpose(e), e)
    return mse, w

    
'''
Logistic regression
'''
def sigmoid(t):
    """apply the sigmoid function on t."""
    return 1/(1 + np.exp(-t))

def calculate_loss(y, tx, w):
    """compute the loss: negative log likelihood."""
    # ***************************************************
    pred = sigmoid(tx.dot(w))
    loss = y.T.dot(np.log(pred)) + (1 - y).T.dot(np.log(1 - pred))
    return np.squeeze(- loss)
    
def calculate_gradient(y, tx, w):
    """compute the gradient of loss."""
    return np.dot(np.transpose(tx), sigmoid(np.dot(tx, w)) - np.transpose(np.matrix(y)))

def logistic_regression(y, tx, initial_w, max_iters, gamma):
    """Gradient descent algorithm."""
    # Define parameters to store w and loss
    ws = [initial_w]
    losses = []
    w = initial_w
    for n_iter in range(max_iters):
        # compute gradient and loss
        loss = calculate_loss(y, tx, w)
        gradient = calculate_gradient(y, tx, w)
        # update w by gradient
        w = w - gamma*gradient
        # store w and loss
        ws.append(w)
        losses.append(loss)
    loss = losses[-1]
    weight = ws[-1]
    return loss, weight



'''
Regularized Logistic regression
'''
def calculate_hessian(y, tx, w):
    """return the Hessian of the loss function."""
    S = np.zeros((tx.shape[0], tx.shape[0]))
    for i in range(tx.shape[0]):
        value = sigmoid(np.dot(tx[i, :], w)) * (1 - sigmoid(np.dot(tx[i, :], w)))
        S[i, i] = value[0]
    return np.dot(np.dot(np.transpose(tx), S), tx)


def penalized_logistic_regression(y, tx, w, lambda_):
    """return the loss, gradient, and Hessian."""
    loss = calculate_loss(y, tx, w) + lambda_ / (2 * tx.shape[0]) * np.linalg.norm(w) ** 2
    gradient = calculate_gradient(y, tx, w) + lambda_ * w 
    hessian = calculate_hessian(y, tx, w) + lambda_ * np.eye(tx.shape[1])
    # return loss, gradient, and Hessian: TODO
    return loss, gradient, hessian
    
def reg_logistic_regression(y, tx, w, initial_w, gamma, lambda_):
    """
    Using the regularized logistic regression.
    Return the loss and updated w.
    """
    losses = []
    w = initial_w
    for n_iter in range(max_iters):
        # compute gradient and loss
        loss, gradient, hessian = penalized_logistic_regression(y, tx, w, lambda_)
        # update w by gradient
        w = w - np.dot(np.linalg.inv(hessian), gradient)
        # store w and loss
        ws.append(w)
        losses.append(loss)
    loss = losses[-1]
    weight = ws[-1]
    return loss, weight











