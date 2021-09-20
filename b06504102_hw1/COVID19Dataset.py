import torch
import csv
import numpy as np
from torch.utils.data import Dataset, DataLoader

class COVID19Dataset(Dataset):
    ''' Dataset for loading and preprocessing the COVID19 dataset '''
    mean = 0
    std = 0

    def __init__(self,
                 path,
                 mode='train'):
        self.mode = mode

        # Read data into numpy arrays
        with open(path, 'r') as fp:
            data = list(csv.reader(fp))
            data = np.array(data[1:])[:, 1:].astype(float)
        
            '''
            cli: 40, 58, 76
            ili: 41, 59, 77 
            hh_cmnty_cli: 42, 60, 78
            nohh_cmnty_cli: 43, 61, 79
            shop: 47, 65, 83
            public_transit: 51, 69, 87
            worried_finance: 56, 74, 92
            tested_positive: 57, 75
            '''
        
        # # 4 症狀、陽性
        # feats = list(range(40)) + [40,41,42,43,57,58,59,60,61,75,76,77,78,79] 
        # # 4 症狀、worried_finance、陽性
        # feats = list(range(40)) + [40,41,42,43,56,57,58,59,60,61,74,75,76,77,78,79,92]
        # 4 症狀、public_transit、worried_finance、陽性
        # feats = list(range(40)) + [40,41,42,43,51,56,57,58,59,60,61,69,74,75,76,77,78,79,87,92]
        # 4 症狀、shop、public_transit、worried_finance、陽性
        feats = list(range(40)) + [40,41,42,43,47,51,56,57,58,59,60,61,65,69,74,75,76,77,78,79,83,87,92]



        if mode == 'test':
            # Testing data
            # data: 893 x 93 (40 states + day 1 (18) + day 2 (18) + day 3 (17))
            data = data[:, feats]
            self.data = torch.FloatTensor(data)
        else:
            # Training data (train/dev sets)
            # data: 2700 x 94 (40 states + day 1 (18) + day 2 (18) + day 3 (18))
            target = data[:, -1]
            data = data[:, feats]
            
            # Splitting training data into train & dev sets
            if mode == 'train':
                indices = [i for i in range(len(data)) if i % 10 != 0]
                # indices = [i for i in range(len(data))]
            elif mode == 'dev':
                indices = [i for i in range(len(data)) if i % 10 == 0]
                # indices = [i for i in range(len(data))]


            
            # Convert data into PyTorch tensors
            self.data = torch.FloatTensor(data[indices])
            self.target = torch.FloatTensor(target[indices])

            # if mode == 'train':
            #     COVID19Dataset.mean = self.data[:, 40:].mean(dim=0, keepdim=True)
            #     COVID19Dataset.std  = self.data[:, 40:].std(dim=0, keepdim=True)

        # Normalize features (you may remove this part to see what will happen)
        # self.data[:, 40:] = \
        #     (self.data[:, 40:] - self.data[:, 40:].mean(dim=0, keepdim=True)) \
        #     / self.data[:, 40:].std(dim=0, keepdim=True)

        # Normalize with train dataset
        # self.data[:, 40:] = (self.data[:, 40:] - COVID19Dataset.mean) / COVID19Dataset.std

        self.dim = self.data.shape[1]

        print('Finished reading the {} set of COVID19 Dataset ({} samples found, each dim = {})'
              .format(mode, len(self.data), self.dim))

    def __getitem__(self, index):
        # Returns one sample at a time
        if self.mode in ['train', 'dev']:
            # For training
            return self.data[index], self.target[index]
        else:
            # For testing (no target)
            return self.data[index]

    def __len__(self):
        # Returns the size of the dataset
        return len(self.data)

