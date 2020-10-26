#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 25 12:53:42 2020
In data preprocessing part, we split data into 8 parts and augment each part by the degree we get in test.
In ridge regression part, we train each group with lambdas we get in test
In output part, we combine the results of each group
 
@author: Jiaan Zhu, Qinyue Zheng, Lei Wang
"""

import numpy as np
import matplotlib.pyplot as plt
import datetime

from helpers import *
from proj1_helpers import *
from implementations import *
# load train set
DATA_TRAIN_PATH = 'train.csv' 
y, tX, ids = load_csv_data(DATA_TRAIN_PATH)
# load test set
DATA_TEST_PATH = 'test.csv'
_, tX_test, ids_test = load_csv_data(DATA_TEST_PATH)
    
tx = tX.copy() # Make a copy of train data
tx_test = tX_test.copy() # Make a copy of test data

# Useful starting lines
def demo():
    # Split train set and test set into 4 groups
    # Split y
    y_0 = y[tx[:,22] == 0]
    y_1 = y[tx[:,22] == 1]
    y_2 = y[tx[:,22] == 2]
    y_3 = y[tx[:,22] == 3]
    # Split tx and tx_test in to 4 groups
    tx_0 = tx[tx[:,22] == 0]
    tx_1 = tx[tx[:,22] == 1]
    tx_2 = tx[tx[:,22] == 2]
    tx_3 = tx[tx[:,22] == 3]
    tx_test_0 = tx_test[tx_test[:,22] == 0]
    tx_test_1 = tx_test[tx_test[:,22] == 1]
    tx_test_2 = tx_test[tx_test[:,22] == 2]
    tx_test_3 = tx_test[tx_test[:,22] == 3]
    # Remove column 22
    tx_01 = np.delete(tx_0, 22, 1)
    tx_11 = np.delete(tx_1, 22, 1)
    tx_21 = np.delete(tx_2, 22, 1)
    tx_31 = np.delete(tx_3, 22, 1)
    tx_test_01 = np.delete(tx_test_0, 22, 1)
    tx_test_11 = np.delete(tx_test_1, 22, 1)
    tx_test_21 = np.delete(tx_test_2, 22, 1)
    tx_test_31 = np.delete(tx_test_3, 22, 1)
    # Remove column with all -999 
    tx_01 = np.delete(tx_01, [4,5,6,12,22,23,24,25,26,27,28], 1)
    tx_11 = np.delete(tx_11, [4,5,6,12,25,26,27], 1)
    tx_test_01 = np.delete(tx_test_01, [4,5,6,12,22,23,24,25,26,27,28], 1)
    tx_test_11 = np.delete(tx_test_11, [4,5,6,12,25,26,27], 1)
    
    # Split into 8 groups
    # Split tx
    tx_01_1 = tx_01[tx_01[:,0] == -999]
    tx_01_2 = tx_01[tx_01[:,0] != -999]
    tx_11_1 = tx_11[tx_11[:,0] == -999]
    tx_11_2 = tx_11[tx_11[:,0] != -999]
    tx_21_1 = tx_21[tx_21[:,0] == -999]
    tx_21_2 = tx_21[tx_21[:,0] != -999]
    tx_31_1 = tx_31[tx_31[:,0] == -999]
    tx_31_2 = tx_31[tx_31[:,0] != -999]
    tx_test_01_1 = tx_test_01[tx_test_01[:,0] == -999]
    tx_test_01_2 = tx_test_01[tx_test_01[:,0] != -999]
    tx_test_11_1 = tx_test_11[tx_test_11[:,0] == -999]
    tx_test_11_2 = tx_test_11[tx_test_11[:,0] != -999]
    tx_test_21_1 = tx_test_21[tx_test_21[:,0] == -999]
    tx_test_21_2 = tx_test_21[tx_test_21[:,0] != -999]
    tx_test_31_1 = tx_test_31[tx_test_31[:,0] == -999]
    tx_test_31_2 = tx_test_31[tx_test_31[:,0] != -999]
    # Remove column with all -999
    tx_01_1 = np.delete(tx_01_1, 0, 1)
    tx_11_1 = np.delete(tx_11_1, 0, 1)
    tx_21_1 = np.delete(tx_21_1, 0, 1)
    tx_31_1 = np.delete(tx_31_1, 0, 1)
    tx_test_01_1 = np.delete(tx_test_01_1, 0, 1)
    tx_test_11_1 = np.delete(tx_test_11_1, 0, 1)
    tx_test_21_1 = np.delete(tx_test_21_1, 0, 1)
    tx_test_31_1 = np.delete(tx_test_31_1, 0, 1)
    # Split y
    y_0_1 = y_0[tx_01[:,0] == -999]
    y_0_2 = y_0[tx_01[:,0] != -999]
    y_1_1 = y_1[tx_11[:,0] == -999]
    y_1_2 = y_1[tx_11[:,0] != -999]
    y_2_1 = y_2[tx_21[:,0] == -999]
    y_2_2 = y_2[tx_21[:,0] != -999]
    y_3_1 = y_3[tx_31[:,0] == -999]
    y_3_2 = y_3[tx_31[:,0] != -999]
    # Shift column 15 of tx_01_2
    tx_01_2[:, 15] = tx_01_2[:, 15] + 25
    tx_test_01_2[:, 15] = tx_test_01_2[:, 15] + 25
    
    # Remove outliers in train set by IQR method
    tx_01_1, y_0_1 = iqr(tx_01_1, y_0_1, 11)
    tx_01_2, y_0_2 = iqr(tx_01_2, y_0_2, 7)
    tx_11_1, y_1_1 = iqr(tx_11_1, y_1_1, 11)
    tx_11_2, y_1_2 = iqr(tx_11_2, y_1_2, 5)
    tx_21_1, y_2_1 = iqr(tx_21_1, y_2_1, 17)
    tx_21_2, y_2_2 = iqr(tx_21_2, y_2_2, 9)
    tx_31_1, y_3_1 = iqr(tx_31_1, y_3_1, 19)
    tx_31_2, y_3_2 = iqr(tx_31_2, y_3_2, 15)
    
    # Augment data
    degree1_1 = 7
    degree1_2 = 8
    degree2_1 = 4
    degree2_2 = 9
    degree3_1 = 2
    degree3_2 = 8
    degree4_1 = 1
    degree4_2 = 10
    tx_01_1 = build_poly(np.transpose(tx_01_1), degree1_1)
    tx_01_2 = build_poly(np.transpose(tx_01_2), degree1_2)
    tx_11_1 = build_poly(np.transpose(tx_11_1), degree2_1)
    tx_11_2 = build_poly(np.transpose(tx_11_2), degree2_2)
    tx_21_1 = build_poly(np.transpose(tx_21_1), degree3_1)
    tx_21_2 = build_poly(np.transpose(tx_21_2), degree3_2)
    tx_31_1 = build_poly(np.transpose(tx_31_1), degree4_1)
    tx_31_2 = build_poly(np.transpose(tx_31_2), degree4_2)
    tx_test_01_1 = build_poly(np.transpose(tx_test_01_1), degree1_1)
    tx_test_01_2 = build_poly(np.transpose(tx_test_01_2), degree1_2)
    tx_test_11_1 = build_poly(np.transpose(tx_test_11_1), degree2_1)
    tx_test_11_2 = build_poly(np.transpose(tx_test_11_2), degree2_2)
    tx_test_21_1 = build_poly(np.transpose(tx_test_21_1), degree3_1)
    tx_test_21_2 = build_poly(np.transpose(tx_test_21_2), degree3_2)
    tx_test_31_1 = build_poly(np.transpose(tx_test_31_1), degree4_1)
    tx_test_31_2 = build_poly(np.transpose(tx_test_31_2), degree4_2)
    
    
    # Running ridge regression algorithm
    lambda_0_1 = 0
    lambda_0_2 = 0
    lambda_1_1 = 0
    lambda_1_2 = 0
    lambda_2_1 = 0
    lambda_2_2 = 0
    lambda_3_1 = 0
    lambda_3_2 = 0
    mse_0_1, weights_rg_0_1 = ridge_regression(np.transpose(np.matrix(y_0_1)), tx_01_1, lambda_0_1)
    mse_0_2, weights_rg_0_2 = ridge_regression(np.transpose(np.matrix(y_0_2)), tx_01_2, lambda_0_2)
    mse_1_1, weights_rg_1_1 = ridge_regression(np.transpose(np.matrix(y_1_1)), tx_11_1, lambda_1_1)
    mse_1_2, weights_rg_1_2 = ridge_regression(np.transpose(np.matrix(y_1_2)), tx_11_2, lambda_1_2)
    mse_2_1, weights_rg_2_1 = ridge_regression(np.transpose(np.matrix(y_2_1)), tx_21_1, lambda_2_1)
    mse_2_2, weights_rg_2_2 = ridge_regression(np.transpose(np.matrix(y_2_2)), tx_21_2, lambda_2_2)
    mse_3_1, weights_rg_3_1 = ridge_regression(np.transpose(np.matrix(y_3_1)), tx_31_1, lambda_3_1)
    mse_3_2, weights_rg_3_2 = ridge_regression(np.transpose(np.matrix(y_3_2)), tx_31_2, lambda_3_2)
    
    # Predict each subgroup
    y_pred_0_1 = predict_labels(weights_rg_0_1, tx_test_01_1)
    y_pred_0_2 = predict_labels(weights_rg_0_2, tx_test_01_2)
    y_pred_1_1 = predict_labels(weights_rg_1_1, tx_test_11_1)
    y_pred_1_2 = predict_labels(weights_rg_1_2, tx_test_11_2)
    y_pred_2_1 = predict_labels(weights_rg_2_1, tx_test_21_1)
    y_pred_2_2 = predict_labels(weights_rg_2_2, tx_test_21_2)
    y_pred_3_1 = predict_labels(weights_rg_3_1, tx_test_31_1)
    y_pred_3_2 = predict_labels(weights_rg_3_2, tx_test_31_2)
    # Combine the results of each subgroup
    y_pred_0 = np.zeros([tx_test_01.shape[0],1])
    y_pred_1 = np.zeros([tx_test_11.shape[0],1])
    y_pred_2 = np.zeros([tx_test_21.shape[0],1])
    y_pred_3 = np.zeros([tx_test_31.shape[0],1])
    y_pred_0[np.where(tx_test_01[:,0] == -999)] = y_pred_0_1
    y_pred_0[np.where(tx_test_01[:,0] != -999)] = y_pred_0_2
    y_pred_1[np.where(tx_test_11[:,0] == -999)] = y_pred_1_1
    y_pred_1[np.where(tx_test_11[:,0] != -999)] = y_pred_1_2
    y_pred_2[np.where(tx_test_21[:,0] == -999)] = y_pred_2_1
    y_pred_2[np.where(tx_test_21[:,0] != -999)] = y_pred_2_2
    y_pred_3[np.where(tx_test_31[:,0] == -999)] = y_pred_3_1
    y_pred_3[np.where(tx_test_31[:,0] != -999)] = y_pred_3_2
    y_pred = np.zeros([tx_test.shape[0],1])
    y_pred[np.where(tx_test[:,22] == 0)] = y_pred_0
    y_pred[np.where(tx_test[:,22] == 1)] = y_pred_1
    y_pred[np.where(tx_test[:,22] == 2)] = y_pred_2
    y_pred[np.where(tx_test[:,22] == 3)] = y_pred_3
    # Create submission file
    create_csv_submission(ids_test, y_pred, 'to_submit.csv')

# Run demo
demo()
    



    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    