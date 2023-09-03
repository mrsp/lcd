import numpy as np

from torch.utils.data import Dataset
from utils import read_dataset, remove_outliers, merge_slip_with_fly, remove_features, normalize

class ContactDataSet(Dataset):
    def __init__(self, csv_file, root_dir, transform = None, point_feet = False, add_noise = False):
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

        # Remove features and add noise in case of point-feet robot and optionally add noise to the
        # data
        if (point_feet):
            dataset = remove_features([0,1,3,4,5],dataset)
            # Remove features and add noise in case of point-feet robot
            if (add_noise):
                dataset[:,0:1] = add_noise(dataset[:,0:1],0.6325)       # Fz
                dataset[:,1:4] = add_noise(dataset[:,1:4],0.0078)       # ax ay az
                dataset[:,4:7] = add_noise(dataset[:,4:7],0.00523)      # wx wy wz
        else:
            if (add_noise):
                dataset[:,:3]  = add_noise(dataset[:,:3],0.6325)       # Fx Fy Fz
                dataset[:,3:6] = add_noise(dataset[:,3:6],0.03)        # Tx Ty Tz
                dataset[:,6:9] = add_noise(dataset[:,6:9],0.0078)      # ax ay az
                dataset[:,9:12] = add_noise(dataset[:,9:12],0.00523)   # wx wy wz
        
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
