import numpy as np
import pandas as pd

# Reads a dataset consisting of data with their corresponding labels
def read_dataset(filename):
    df = pd.read_csv(filename, engine='python')
    dataset = df.values
    dataset = dataset.astype('float32')
    return dataset

# Outlier removal. (due to fall data spikes)
def remove_outliers(dataset, labels):
    feature_mean = []  # contains the mean value for every feature
    feature_std = []  # contains the standard deviation of every feature

    for i in range(dataset.shape[1]):
        feature_mean.append(np.mean(dataset[:, i]))
        feature_std.append(np.std(dataset[:, i]))

    # identify outliers
    cut_off = []
    lower_bound = []
    upper_bound = []
    num_std = 3       # how many sigmas

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

# Merges no_contact and slip labels -> UNSTABLE contact
def merge_slip_with_fly(labels):
    for i in range(labels.shape[0]):
        if labels[i] == 2:
            labels[i] = 1
    return labels

# Normalizes data in [-1, 1]
def normalize(din, dmax):
    if(dmax != 0):
        dout =  np.abs(din/dmax)
    else:
        dout =  np.zeros((np.size(din)))
    return dout

# Removes list of features ( for point feet robots)
def remove_features(features_to_remove,dataset):
  dataset = np.delete(dataset,features_to_remove,axis=1)
  return dataset

# Adds gaussian noise to data
def add_noise(data,std):
  mu = 0  # mean and standard deviation
  s = np.random.normal(mu, std, data.shape[0])
  for f in range(0,data.shape[1]):
    for i in range(0,data.shape[0]):
      data[i,f] = data[i,f] + s[i]
  return data
