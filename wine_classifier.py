#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Skeleton code for CW2 submission.
We suggest you implement your code in the provided functions
Make sure to use our print_features() and print_predictions() functions
to print your results
"""

from __future__ import print_function

import argparse
import numpy as np
import matplotlib.pyplot as plt
from utilities import load_data, print_features, print_predictions
from sklearn.decomposition import PCA

# you may use these colours to produce the scatter plots
CLASS_1_C = r'#3366ff'
CLASS_2_C = r'#cc3300'
CLASS_3_C = r'#ffc34d'

MODES = ['feature_sel', 'knn', 'alt', 'knn_3d', 'knn_pca']

train_set, train_labels, test_set, test_labels = load_data()

def getColours(train_labels):
    colours = []
    for l in train_labels:
        if l == 1:
            colours.append(CLASS_1_C)
        elif l == 2:
            colours.append(CLASS_2_C)
        elif l == 3:
            colours.append(CLASS_3_C)
    return colours

def feature_selection(train_set, train_labels, **kwargs):
    # write your code here and make sure you return the features at the end of
    # the function
    n_features = train_set.shape[1]
    # fig, ax = plt.subplots(n_features, n_features)      # 2 features
    fig, ax = plt.subplots(1, 1)      # To plot chosen feature
    # plt.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.01, wspace=0.2, hspace=0.4)

    # for r in range(n_features):
    #     for c in range(n_features):
    #         ax[r, c].scatter(train_set[:, r], train_set[:, c], c = getColours(train_labels), s = 1)
    plt.scatter(train_set[:, 10], train_set[:, 12], c = getColours(train_labels), s = 8)
    # plt.show()
    best_feature = [10, 12]
    return best_feature

# Reducing the data sets for the best feature
def reduce_feature(test_set, train_set, feature):
    red_test = test_set[:, feature]
    red_train = train_set[:, feature]
    return red_test, red_train

# Calculating Euclidean Distance between 2 points
def euclidean_dist(x, y):
    return np.sqrt(np.sum(np.power((x - y), 2)))

# Function to calculate the accuracy
def calc_accuracy(gt_labels, pred_labels):
    same = 0
    for i in range(len(gt_labels)):
        if gt_labels[i] == pred_labels[i]:
            same += 1
    accuracy = (same/len(gt_labels)) * 100
    return accuracy

# Function to calculate the ratio
def countElem(gt_labels, pred_labels, element, predicted_element):
    count = 0
    predCount = 0
    for i in range(len(gt_labels)):
        if gt_labels[i] == element:
            count += 1
            if pred_labels[i] == predicted_element:
                predCount += 1
    actual = predCount/count
    return actual

# Function to calculate the confusion matrix
def calculate_confusion_matrix(gt_labels, pred_labels):
    uniqueLabels = np.unique(pred_labels)
    dim = len(uniqueLabels)
    CM = np.empty((dim, dim))
    for i in range(dim):
        for j in range(dim):
            CM[i][j] = (countElem(gt_labels, pred_labels, uniqueLabels[i], uniqueLabels[j]))

    return CM

def knn(train_set, train_labels, test_set, k, **kwargs):
    # write your code here and make sure you return the predictions at the end of
    # the function
    red_test, red_train = reduce_feature(test_set, train_set, [10, 12])

    calc_distance = lambda x : [euclidean_dist(x, b) for b in red_train]
    predictions = []

    for i in red_test:
        count1 = 0      # keep track of count when classified as respective label
        count2 = 0
        count3 = 0
        dist = np.column_stack((calc_distance(i), train_labels))    # storing the calculated distance and corresponding label
        sorted_dist_indices = np.argsort(dist[:, 0])
        dist = dist[sorted_dist_indices]
        knn = dist[:k]      # storing the first k distances
        knn = knn[:, 1]     # storing only the train labels

        for j in knn:
            if j == 1:
                count1 += 1
            if j == 2:
                count2 += 1
            if j == 3:
                count3 += 1

        counts = [count1, count2, count3]
        predictions.append(np.argmax(counts) + 1)

    accuracy = calc_accuracy(test_labels, predictions)
    # print(accuracy)
    CM = calculate_confusion_matrix(test_labels, predictions)
    return predictions

# Function to calculate the prior probabilities of each class
def prior_prob(train_labels):
    count1 = list(train_labels).count(1)
    count2 = list(train_labels).count(2)
    count3 = list(train_labels).count(3)

    prob1 = count1/len(train_labels)
    prob2 = count2/len(train_labels)
    prob3 = count3/len(train_labels)

    priors = [prob1, prob2, prob3]

    return priors

# Function to find the mean of each class in the training set (row-wise)
def mean_of_features(train_set, train_labels):
    sorted_set = []
    means = []

    set1 = []
    set2 = []
    set3 = []

    final = []

    for i in range(len(train_labels)):
        if train_labels[i] == 1:
            set1.append(train_set[i])
        elif train_labels[i] == 2:
            set2.append(train_set[i])
        else:
            set3.append(train_set[i])

    sorted_set = [set1, set2, set3]

    for j in range(len(sorted_set)):
        x = np.column_stack(sorted_set[j])
        final.append(x)

    for k in range(len(final)):
        mean_feat_1 = np.mean(final[k][0])
        mean_feat_2 = np.mean(final[k][1])

        means.append(mean_feat_1)
        means.append(mean_feat_2)

    return means

# Function to find the variance of each class in the training set
def stddev_of_features(train_set, train_labels):
    sorted_set = []
    stddevs = []

    set1 = []
    set2 = []
    set3 = []

    final = []

    for i in range(len(train_labels)):
        if train_labels[i] == 1:
            set1.append(train_set[i])
        elif train_labels[i] == 2:
            set2.append(train_set[i])
        else:
            set3.append(train_set[i])

    sorted_set = [set1, set2, set3]

    for j in range(len(sorted_set)):
        x = np.column_stack(sorted_set[j])
        final.append(x)

    for k in range(len(final)):
        stddev_feat_1 = np.std(final[k][0])
        stddev_feat_2 = np.std(final[k][1])

        stddevs.append(stddev_feat_1)
        stddevs.append(stddev_feat_2)

    return stddevs

# Function to calculate the probability density function
def posterior_prob(t, mean, stddev):
    exponent = np.exp( -(np.power(t - mean, 2) / (2 * np.power(stddev, 2))))
    return (1 / (np.sqrt(2 * np.pi) * stddev)) * exponent

# Function to calculate the final conditional probability
def calc_final_conditional(prob1, prob2, prob3, train_labels):
    priors = prior_prob(train_labels)

    p1 = prob1 * priors[0]
    p2 = prob2 * priors[1]
    p3 = prob3 * priors[2]

    total = p1 + p2 + p3

    ans1 = p1/total
    ans2 = p2/total
    ans3 = p3/total

    probs = [ans1, ans2, ans3]

    return probs

def alternative_classifier(train_set, train_labels, test_set, **kwargs):
    # write your code here and make sure you return the predictions at the end of
    # the function
    red_test, red_train = reduce_feature(test_set, train_set, [10, 12])

    predictions = []
    means = mean_of_features(red_train, train_labels)
    stddevs = stddev_of_features(red_train, train_labels)

    for t in red_test:
        prob1 = posterior_prob(t[0], means[0], stddevs[0])
        _prob1 = posterior_prob(t[1], means[1], stddevs[1])
        prob_1_tot = prob1 * _prob1

        prob2 = posterior_prob(t[0], means[2], stddevs[2])
        _prob2 = posterior_prob(t[1], means[3], stddevs[3])
        prob_2_tot = prob2 * _prob2

        prob3 = posterior_prob(t[0], means[4], stddevs[4])
        _prob3 = posterior_prob(t[1], means[5], stddevs[5])
        prob_3_tot = prob3 * _prob3

        conditionals = calc_final_conditional(prob_1_tot, prob_2_tot, prob_3_tot, train_labels)

        best_prob = np.argmax(conditionals) + 1
        predictions.append(best_prob)

    accuracy = calc_accuracy(test_labels, predictions)
    # print(accuracy)
    CM = calculate_confusion_matrix(test_labels, predictions)
    return predictions

# Function to return the feature with the best spread. Choosing the resp feature
# after printing the required matrices
def choose_best_3d(train_set):
    n_features = train_set.shape[1]         # best feature is 10

    for i in range(n_features):
        red_train_3d = train_set[:, [i, 10, 12]]
        covariance_mat = np.cov(red_train_3d)

    best_3d_feature = [7, 10, 12]

    return best_3d_feature

def knn_three_features(train_set, train_labels, test_set, k, **kwargs):
    # write your code here and make sure you return the predictions at the end of
    # the function
    best_3d_feature = choose_best_3d(train_set)

    red_train_3d = train_set[:, best_3d_feature]
    red_test_3d = test_set[:, best_3d_feature]

    calc_distance = lambda x : [euclidean_dist(x, b) for b in red_train_3d]
    predictions = []

    for i in red_test_3d:
        count1 = 0      # keep track of count when classified as respective label
        count2 = 0
        count3 = 0
        dist = np.column_stack((calc_distance(i), train_labels))    # storing the calculated distance and corresponding label
        sorted_dist_indices = np.argsort(dist[:, 0])
        dist = dist[sorted_dist_indices]
        knn = dist[:k]      # storing the first k distances
        knn = knn[:, 1]     # storing only the train labels

        for j in knn:
            if j == 1:
                count1 += 1
            if j == 2:
                count2 += 1
            if j == 3:
                count3 += 1

        counts = [count1, count2, count3]
        predictions.append(np.argmax(counts) + 1)

    accuracy = calc_accuracy(test_labels, predictions)
    # print(accuracy)
    CM = calculate_confusion_matrix(test_labels, predictions)
    return predictions


def knn_pca(train_set, train_labels, test_set, k, n_components=2, **kwargs):
    # write your code here and make sure you return the predictions at the end of
    # the function

    # With scipy's PCA
    scipyPCA = PCA(n_components=2)                               # For 2 eigenvectors
    scipyPCA.fit(train_set)                                      # calculates W matrix
    scipy_train_set_transformed = scipyPCA.transform(train_set)  # transforming the train set
    scipy_test_set_transformed = scipyPCA.transform(test_set)    # transforming the test set

    calc_distance = lambda x : [euclidean_dist(x, b) for b in scipy_train_set_transformed]
    predictions = []

    for i in scipy_test_set_transformed:
        count1 = 0      # keep track of count when classified as respective label
        count2 = 0
        count3 = 0
        dist = np.column_stack((calc_distance(i), train_labels))    # storing the calculated distance and corresponding label
        sorted_dist_indices = np.argsort(dist[:, 0])
        dist = dist[sorted_dist_indices]
        knn = dist[:k]      # storing the first k distances
        knn = knn[:, 1]     # storing only the train labels

        for j in knn:
            if j == 1:
                count1 += 1
            if j == 2:
                count2 += 1
            if j == 3:
                count3 += 1

        counts = [count1, count2, count3]
        predictions.append(np.argmax(counts) + 1)

    accuracy = calc_accuracy(test_labels, predictions)
    # print(accuracy)
    CM = calculate_confusion_matrix(test_labels, predictions)
    return predictions


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', nargs=1, type=str, help='Running mode. Must be one of the following modes: {}'.format(MODES))
    parser.add_argument('--k', nargs='?', type=int, default=1, help='Number of neighbours for knn')
    parser.add_argument('--train_set_path', nargs='?', type=str, default='data/wine_train.csv', help='Path to the training set csv')
    parser.add_argument('--train_labels_path', nargs='?', type=str, default='data/wine_train_labels.csv', help='Path to training labels')
    parser.add_argument('--test_set_path', nargs='?', type=str, default='data/wine_test.csv', help='Path to the test set csv')
    parser.add_argument('--test_labels_path', nargs='?', type=str, default='data/wine_test_labels.csv', help='Path to the test labels csv')

    args = parser.parse_args()
    mode = args.mode[0]

    return args, mode


if __name__ == '__main__':
    args, mode = parse_args() # get argument from the command line

    # load the data
    train_set, train_labels, test_set, test_labels = load_data(train_set_path=args.train_set_path,
                                                                       train_labels_path=args.train_labels_path,
                                                                       test_set_path=args.test_set_path,
                                                                       test_labels_path=args.test_labels_path)
    if mode == 'feature_sel':
        selected_features = feature_selection(train_set, train_labels)
        print_features(selected_features)
    elif mode == 'knn':
        predictions = knn(train_set, train_labels, test_set, args.k)
        print_predictions(predictions)
    elif mode == 'alt':
        predictions = alternative_classifier(train_set, train_labels, test_set)
        print_predictions(predictions)
    elif mode == 'knn_3d':
        predictions = knn_three_features(train_set, train_labels, test_set, args.k)
        print_predictions(predictions)
    elif mode == 'knn_pca':
        prediction = knn_pca(train_set, train_labels, test_set, args.k)
        print_predictions(prediction)
    else:
        raise Exception('Unrecognised mode: {}. Possible modes are: {}'.format(mode, MODES))
