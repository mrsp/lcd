import numpy as np

from torch.utils.data import Dataset
from utils import read_dataset, remove_outliers, merge_slip_with_fly, remove_features, normalize

class ContactDataSet(Dataset):
    def __init__(self, csv_file, root_dir, transform=None, point_feet = False):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
            on a sample.
        """
        dataset = read_dataset(root_dir + "/" + csv_file)

        labels = dataset[:, -1]         # delete labels
        dataset = np.delete(dataset, -1, axis=1)
        
        # Merge slip labels with no_contact labels = Unstable contact
        labels = merge_slip_with_fly(labels)

        dataset, labels = remove_outliers(dataset, labels)

        # Remove features and add noise in case of point-feet robot
        if (point_feet):
            dataset = remove_features([0,1,3,4,5],dataset)

        # Normalize data in [-1, 1]
        for i in range(dataset.shape[1]):
            dataset[:,i] = normalize(dataset[:,i],np.max(abs(dataset[:,i])))

        dataset, labels = remove_outliers(dataset, labels)
        self.data = dataset
        self.labels = labels
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.data[idx, :], int(self.labels[idx])
