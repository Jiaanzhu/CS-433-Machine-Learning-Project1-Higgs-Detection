#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 27 14:14:49 2020

@author: jiaanzhu
"""
import numpy as np

def standardize(x):
    """Standardize the original data set."""
    x = (x - np.mean(x, axis=0)) / np.std(x, axis=0)
    return x

def normalize(x):
    """Standardize the original data set."""
    x = (x - np.amin(x, axis=0)) / (np.amax(x, axis=0) - np.min(x, axis=0))
    return x

def compute_loss(y, tx, w):
    """Calculate the loss.

    You can calculate the loss using mse or mae.
    """
    # ***************************************************
    # INSERT YOUR CODE HERE
    # For MSE
    return (1/(2 * tx.shape[0]))*np.dot(y - np.dot(tx, w), (y - np.dot(tx, w)))
    # For MAE
    # return (1/tx.shape[0])*np.sum(np.abs((y - np.dot(tx, w))))
    # TODO: compute loss by MSE
    # ***************************************************   
    
def compute_gradient(y, tx, w):
    """Compute the gradient."""
    # ***************************************************
    # INSERT YOUR CODE HERE
    # TODO: compute gradient and error vector
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
        # ***************************************************
        # INSERT YOUR CODE HERE
        # TODO: compute gradient and loss
        loss = compute_loss(y, tx, w)
        gradient = compute_gradient(y, tx, w)
        # ***************************************************
        # ***************************************************
        # INSERT YOUR CODE HERE
        # TODO: update w by gradient
        w = w - gamma*gradient
        # ***************************************************
        # store w and loss
        ws.append(w)
        losses.append(loss)
        #print("Gradient Descent({bi}/{ti}): loss={l}, w0={w0}, w1={w1}".format(
        #      bi=n_iter, ti=max_iters - 1, l=loss, w0=w[0], w1=w[1]))

    return losses, ws

def compute_stoch_gradient(y, tx, w):
    """Compute a stochastic gradient from just few examples n and their corresponding y_n labels."""
    # ***************************************************
    # INSERT YOUR CODE HERE
    # TODO: implement stochastic gradient computation.It's same as the gradient descent.
    for minibatch_y, minibatch_tx in batch_iter(y, tx, 1):
        
        return (-1/minibatch_tx.shape[0]) * np.dot(np.transpose(minibatch_tx), (minibatch_y - np.dot(minibatch_tx, w)))
    # ***************************************************


def stochastic_gradient_descent(
        y, tx, initial_w, batch_size, max_iters, gamma):
    """Stochastic gradient descent algorithm."""
    # ***************************************************
    # INSERT YOUR CODE HERE
    # TODO: implement stochastic gradient descent.
    ws = [initial_w]
    losses = []
    w = initial_w
    for n_iter in range(max_iters):
        # ***************************************************
        # INSERT YOUR CODE HERE
        # TODO: compute gradient and loss
        loss = compute_loss(y, tx, w)
        gradient = compute_stoch_gradient(y, tx, w)
        # ***************************************************
        # ***************************************************
        # INSERT YOUR CODE HERE
        # TODO: update w by gradient
        w = w - gamma*gradient
        # ***************************************************
        # store w and loss
        ws.append(w)
        losses.append(loss)   
    # ***************************************************
    return losses, ws

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
            
def least_squares(y, tx):
    """calculate the least squares solution."""
    # ***************************************************
    # INSERT YOUR CODE HERE
    a = np.dot(np.transpose(tx), tx)
    b = np.dot(np.transpose(tx), y)
    w = np.linalg.solve(a, b)
    e = y - np.dot(tx, w)
    mse = 1/(2*tx.shape[0]) * np.dot(np.transpose(e), e)
    # returns mse, and optimal weights
    return mse, w
    # ***************************************************

def build_poly(x, degree):
    """polynomial basis functions for input data x, for j=0 up to j=degree."""
    # ***************************************************
    # INSERT YOUR CODE HERE
    x2 = np.transpose(np.matrix(x))
    x_poly = np.ones((x2.shape[0], 1))
    for i in range(1, degree+1):
        x_poly = np.append(x_poly, np.power(x2, i),  axis=1)
    return x_poly
    # this function should return the matrix formed
    # by applying the polynomial basis to the input data
    # ***************************************************

def ridge_regression(y, tx, lambda_):
    """implement ridge regression."""
    # ***************************************************
    # INSERT YOUR CODE HERE
    a = np.dot(np.transpose(tx), tx) + 2 * tx.shape[0] * lambda_ * np.identity(tx.shape[1])
    # 2 * tx.shape[0]
    b = np.dot(np.transpose(tx), y)
    w = np.linalg.solve(a, b)
    e = y - np.dot(tx, w)
    mse = 1/(2*tx.shape[0]) * np.dot(np.transpose(e), e)
    return mse, w
    # ***************************************************

