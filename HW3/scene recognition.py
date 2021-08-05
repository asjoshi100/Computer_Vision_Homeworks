# -*- coding: utf-8 -*-
"""
Created on Tue Mar 16 19:48:25 2021

@author: admin
"""


import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import KMeans
from sklearn.svm import LinearSVC
from scipy import stats
from pathlib import Path, PureWindowsPath
from sklearn.neighbors import KNeighborsClassifier


def extract_dataset_info(data_path):
    # extract information from train.txt
    f = open(os.path.join(data_path, "train.txt"), "r")
    contents_train = f.readlines()
    label_classes, label_train_list, img_train_list = [], [], []
    for sample in contents_train:
        sample = sample.split()
        label, img_path = sample[0], sample[1]
        if label not in label_classes:
            label_classes.append(label)
        label_train_list.append(sample[0])
        img_train_list.append(os.path.join(data_path, Path(PureWindowsPath(img_path))))
    print('Classes: {}'.format(label_classes))

    # extract information from test.txt
    f = open(os.path.join(data_path, "test.txt"), "r")
    contents_test = f.readlines()
    label_test_list, img_test_list = [], []
    for sample in contents_test:
        sample = sample.split()
        label, img_path = sample[0], sample[1]
        label_test_list.append(label)
        img_test_list.append(os.path.join(data_path, Path(PureWindowsPath(img_path))))  # you can directly use img_path if you run in Windows

    return label_classes, label_train_list, img_train_list, label_test_list, img_test_list


def sift(*args, **kwargs):
    return cv2.xfeatures2d.SIFT_create(*args, **kwargs)

def compute_dsift(img, stride, size):
    # To do
    keypoints = [cv2.KeyPoint(x, y, size)
                 for y in range(0, img.shape[0], stride)
                 for x in range(0, img.shape[1], stride)]
    dense_feature = sift().compute(img, keypoints)[1]
    dense_feature /= dense_feature.sum(axis=1).reshape(-1, 1)
    return dense_feature


def get_tiny_image(img, output_size):
    # To do
    img = cv2.resize(img, output_size)
    mean = np.mean(img)
    std = img.std()
    feature = (img - mean)/std
    return feature


def predict_knn(feature_train, label_train, feature_test, k):
    # To do
    classified = KNeighborsClassifier(n_neighbors=k)
    classified.fit(feature_train, label_train)
    label_test_pred = classified.predict(feature_test)
    return label_test_pred

def classify_function(label_classes, label_train_list, img_train_list, label_test_list, img_test_list, predict, feature_extract):
    if feature_extract == 'bow':
        all_dense_feature = extract_train_dense(label_classes, label_train_list, img_train_list)
        vocab = build_visual_dictionary(all_dense_feature, 200)
        size = 25
        stride = 25
        k = 10
    if feature_extract == 'tiny':
        output_size = (16,16)
        k = 10
    feature_train = []
    label_train = []
    for i in range(len(label_train_list)):
        img = cv2.imread(img_train_list[i], 0)
        if feature_extract == 'bow':
            feature = compute_dsift(img, stride, size)
            feature = compute_bow(feature, vocab)
        if feature_extract == 'tiny':
            feature = get_tiny_image(img, output_size)
            feature = feature.reshape(output_size[0]*output_size[1])
        feature_train.append(feature)
        lab_tra = label_classes.index(label_train_list[i])
        label_train.append(lab_tra)
    feature_train = np.array(feature_train)
    label_train = np.array(label_train)
    # print('Train done')
    
    feature_test = []
    label_test = []
    for i in range(len(label_test_list)):
        img = cv2.imread(img_test_list[i], 0)
        if feature_extract == 'bow':
            feature = compute_dsift(img, stride, size)
            feature = compute_bow(feature, vocab)
        if feature_extract == 'tiny':
            feature = get_tiny_image(img, output_size)
            feature = feature.reshape(output_size[0]*output_size[1])
        feature_test.append(feature)
        lab_test = label_classes.index(label_test_list[i])
        label_test.append(lab_test)
    feature_test = np.array(feature_test)
    label_test = np.array(label_test)
    # print('Test done')
    
    # Check accuracy and confusion matrix
    if predict == 'knn':
        label_test_pred = predict_knn(feature_train, label_train, feature_test, k)
    if predict == 'SVM':
        label_test_pred = predict_svm(feature_train, label_train, feature_test, len(label_classes))
    confusion = np.zeros((len(label_classes), len(label_classes)))
    for i in range(len(label_test_list)):
        true_label = label_test[i]
        pred_label = label_test_pred[i]
        confusion[true_label][pred_label] = confusion[true_label][pred_label] + 1
    confusion_sum = np.sum(confusion, axis=1)
    accuracy_mat = (confusion.T/confusion_sum).T
    accuracy = 0
    for i in range(len(label_classes)):
        accuracy = accuracy + accuracy_mat[i][i]
    accuracy = accuracy/len(label_classes)
    return confusion, accuracy, label_classes


def classify_knn_tiny(label_classes, label_train_list, img_train_list, label_test_list, img_test_list):
    
    confusion, accuracy, label_classes = classify_function(label_classes, label_train_list, img_train_list, label_test_list, img_test_list, 'knn', 'tiny')

    visualize_confusion_matrix(confusion, accuracy, label_classes)
    print('Classify knn tiny done')
    return confusion, accuracy


def extract_train_dense(label_classes, label_train_list, img_train_list):
    size = 25
    stride = 25
    all_dense_feature = np.empty((0, 128))
    for i in range(len(label_train_list)):
        img = cv2.imread(img_train_list[i], 0)
        dense_feature = compute_dsift(img, stride, size)
        all_dense_feature = np.append(all_dense_feature, dense_feature, axis=0)
    return all_dense_feature
        

def build_visual_dictionary(dense_feature_list, dic_size):
    # To do
    # Uncomment this if you want to create new train model (Vocab)
    mean = KMeans(n_clusters = dic_size, n_init=10, max_iter=300).fit(dense_feature_list)
    vocab = mean.cluster_centers_ 
    np.savetxt('vocab.txt', vocab)
    # Till here
    
    # If you want to use trained model then use following (Vocab)
    # vocab = np.loadtxt('vocab.txt')
    # Till here
    
    # print('Visual dictionary created')
    return vocab


def compute_bow(feature, vocab):
    # To do
    knn = NearestNeighbors(n_neighbors = 1)
    knn.fit(vocab)
    kneigh = knn.kneighbors(feature, return_distance=False)
    histo = np.histogram(kneigh, np.arange(vocab.shape[0]+1))[0]
    bow_feature = histo/np.linalg.norm(histo)
    return bow_feature


def classify_knn_bow(label_classes, label_train_list, img_train_list, label_test_list, img_test_list):
    # To do
    confusion, accuracy, label_classes = classify_function(label_classes, label_train_list, img_train_list, label_test_list, img_test_list, 'knn', 'bow')
    
    visualize_confusion_matrix(confusion, accuracy, label_classes)
    print('Classify knn bow done')
    return confusion, accuracy


def predict_svm(feature_train, label_train, feature_test, n_classes):
    # To do
    clf = LinearSVC(C = 1)
    all_conf = np.empty((label_train.shape[0], 0))
    all_train = np.empty((label_train.shape[0], 0))
    for i in range(n_classes):
        new_label_train = [(n_classes) if x != i else x for x in label_train]
        clf.fit(feature_train, new_label_train)
        conf_label_test = clf.decision_function(feature_test)
        conf_label_test = np.array([x if x >= 0 else x for x in conf_label_test]).reshape(conf_label_test.shape[0], 1)
        all_conf = np.hstack((all_conf, conf_label_test))
        
        new_label_train = np.array([(n_classes) if x != i else x for x in label_train]).reshape(conf_label_test.shape[0], 1)
        all_train = np.hstack((all_train, new_label_train))
    label_test_pred = np.argmin(all_conf, axis = 1)
    return label_test_pred


def classify_svm_bow(label_classes, label_train_list, img_train_list, label_test_list, img_test_list):
    # To do
    confusion, accuracy, label_classes = classify_function(label_classes, label_train_list, img_train_list, label_test_list, img_test_list, 'SVM', 'bow')
    
    visualize_confusion_matrix(confusion, accuracy, label_classes)
    print('Classify svm bow done')
    return confusion, accuracy


def visualize_confusion_matrix(confusion, accuracy, label_classes):
    plt.title("accuracy = {:.3f}".format(accuracy))
    plt.imshow(confusion)
    ax, fig = plt.gca(), plt.gcf()
    plt.xticks(np.arange(len(label_classes)), label_classes)
    plt.yticks(np.arange(len(label_classes)), label_classes)
    # set horizontal alignment mode (left, right or center) and rotation mode(anchor or default)
    plt.setp(ax.get_xticklabels(), rotation=-30, ha="center", rotation_mode="default")
    # avoid top and bottom part of heatmap been cut
    ax.set_xticks(np.arange(len(label_classes) + 1) - .5, minor=True)
    ax.set_yticks(np.arange(len(label_classes) + 1) - .5, minor=True)
    ax.tick_params(which="minor", bottom=False, left=False)
    fig.tight_layout()
    fig.set_size_inches(10, 10)
    plt.savefig('figure_{:.3f}.png'.format(accuracy), dpi=500)
    plt.show()


if __name__ == '__main__':
    # To do: replace with your dataset path
    label_classes, label_train_list, img_train_list, label_test_list, img_test_list = extract_dataset_info("./scene_classification_data")
    
    classify_knn_tiny(label_classes, label_train_list, img_train_list, label_test_list, img_test_list)

    classify_knn_bow(label_classes, label_train_list, img_train_list, label_test_list, img_test_list)
    
    classify_svm_bow(label_classes, label_train_list, img_train_list, label_test_list, img_test_list)