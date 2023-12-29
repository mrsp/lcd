import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import itertools


def read_dataset(filename):
    """
        Reads a dataset consisting of data with their corresponding labels.

        Args: 
            filename (string): Dataset file in csv format.
    
        Returns:
            dataset (Numpy array): Dataset in Numpy format.
    """
    df = pd.read_csv(filename, engine='python')
    dataset = df.values
    dataset = dataset.astype('float32')
    return dataset

def remove_outliers(dataset, labels, num_std=3):
    """
        Removes outlier data from a dataset that are num_std sigmas off their corresponding mean 
        value.

        Args:
            dataset (Numpy array): Dataset to be checked for outliers.
            labels (Numpy array): Corresponding dataset labels.
        
        Returns:
            dataset (Numpy array): Dataset with outliers removed.
            labels (Numpy array): Corresponding dataset labels.
    """
    feature_mean = []  # contains the mean value for every feature
    feature_std = []  # contains the standard deviation of every feature

    for i in range(dataset.shape[1]):
        feature_mean.append(np.mean(dataset[:, i]))
        feature_std.append(np.std(dataset[:, i]))

    # Identify outliers
    cut_off = []
    lower_bound = []
    upper_bound = []

    for i in range(dataset.shape[1]):
        cut_off.append(feature_std[i]*num_std)
        lower_bound.append(feature_mean[i]-cut_off[i])
        upper_bound.append(feature_mean[i]+cut_off[i])

    outliers = []    # stores the indexes where the outliers are
    for i in range(dataset.shape[0]):
        for j in range(dataset.shape[1]):
            if (dataset[i, j] <= lower_bound[j]) or (dataset[i, j] >= upper_bound[j]):
                outliers.append(i)
                continue

    # Delete the outliers from dataset
    dataset = np.delete(dataset, outliers, axis=0)
    labels = np.delete(labels, outliers, axis=0)

    return dataset, labels

def merge_slip_with_fly(labels):
    """
        Returns binary labels for stable and unstable contacts by merging no contact with slipping
        contact labels.

        Args:
            labels (Numpy array): Dataset labels.
        
        Returns:
            labels (Numpy array): Merged dataset labels.
    """
    for i in range(labels.shape[0]):
        if labels[i] == 2:
            labels[i] = 1
    return labels

def normalize(din, dmax):
    """
        Normalizes data in [0, dmax]

        Args:
            din (Numpy array): Data to be normalized.
            dmax (float): Maximum allowed value.
        
        Returns:
            dout (Numpy array): Normalized data.
    """
    if(dmax != 0):
        dout = np.abs(din/dmax)
    else:
        dout = np.zeros((np.size(din)))
    return dout

# Removes list of features (for point feet robots)
def remove_features(features_to_remove, dataset):
    """
        Removes a list of features in the dataset
        
        Args:
            features_to_remove (List): List of feature indices to be removed.
            dataset (Numpy array): Dataset from which the indicated features will be removed.
        
        Returns:
            dataset (Numpy array): Dataset with the corresponding features removed.
    """
    dataset = np.delete(dataset, features_to_remove, axis=1)
    return dataset

def add_noise(data, std):
    """
        Adds zero-mean Gaussian noise with standard deviation std to data.

        Args:
            data (Numpy array): Data to be perturbed with noise.
            std (float): Standard deviation of the zero-mean Gaussian distribution.
        
        Returns:
            data (Numpy array): Perturb by noise data.
    """
    mu = 0  # mean and standard deviation
    s = np.random.normal(mu, std, data.shape[0])
    for f in range(0,data.shape[1]):
        for i in range(0,data.shape[0]):
            data[i,f] = data[i,f] + s[i]
    return data

def plot_confusion_matrix(cm, class_names):
    """
        Creates a matplotlib figure containing the plotted confusion matrix.

        Args:
            cm (array, shape = [n, n]): a confusion matrix of integer classes.
            class_names (array, shape = [n]): String names of the integer classes.

        Returns:
            figure (matplotlib figure): Confusion matrix figure.
    """
    figure = plt.figure(figsize=(8, 8))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title("Confusion matrix")
    plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)

    # Compute the labels from the normalized confusion matrix.
    labels = np.around(cm.astype('float') / cm.sum(axis=1)[:, np.newaxis], decimals=2)

    # Use white text if squares are dark; otherwise black.
    threshold = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        color = "white" if cm[i, j] > threshold else "black"
        plt.text(j, i, labels[i, j], horizontalalignment="center", color=color)

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    return figure
