__author__ = "Gorkem Can Ates"
__email__ = "gca45@miami.edu"

import os
import torch
import numpy as np
from torch.utils.data import Dataset
from transforms.transforms import Transforms
from sklearn.model_selection import train_test_split

class TOPODataset:
    def __init__(self,
                 data_path,
                 target_path,
                 train_transform,
                 val_transform,
                 total_data=10000,
                 init_iteration=5,
                 split_size=0.2,
                 shuffle=True,
                 debug=False):

        self.train_transform = train_transform
        self.val_transform = val_transform
        self.init_iteration = init_iteration

        if debug:
            train_ids, test_ids = train_test_split(np.arange(8),
                                                   test_size=split_size,
                                                   random_state=42,
                                                   shuffle=shuffle)

        else:
            samples = np.random.choice(np.arange(os.listdir(data_path).__len__()),
                                       size=(total_data,),
                                       replace=False)
            train_ids, test_ids = train_test_split(samples,
                                                   test_size=split_size,
                                                   random_state=42,
                                                   shuffle=shuffle)

        self.train_dataset = TOPO(data_dir=data_path,
                                  target_dir=target_path,
                                  indexes=train_ids,
                                  init_iteration=self.init_iteration,
                                  transform=self.train_transform
                                  )
        self.test_dataset = TOPO(data_dir=data_path,
                                 target_dir=target_path,
                                 indexes=test_ids,
                                 init_iteration=self.init_iteration,
                                 transform=self.val_transform
                                 )

        print('Data load completed.')

class TOPO(Dataset):
    def __init__(self,
                 data_dir,
                 target_dir,
                 indexes,
                 init_iteration,
                 transform=None,
                 ):
        self.data_dir = data_dir
        self.target_dir = target_dir
        self.transforms = transform
        self.init_iteration = init_iteration
        data_list = os.listdir(data_dir)
        target_list = os.listdir(target_dir)
        self.data_list = [data_list[i] for i in indexes]
        self.target_list = [target_list[i] for i in indexes]

    def __getitem__(self, item):
        img_path = os.path.join(self.data_dir, self.data_list[item])
        target_path = os.path.join(self.target_dir, self.target_list[item])
        im, target = np.load(img_path).astype(np.float32)[:, :, int(self.init_iteration - 1)], \
                   np.load(target_path).astype(np.float32)
        image = np.tile(im[:, :, None], [1, 1, 3])

        if self.transforms is not None:
            augmentations = self.transforms(image=image, mask=target)
            image = augmentations['image']
            target = augmentations['mask']
        return image, target


    def __len__(self):
        return len(self.data_list)
