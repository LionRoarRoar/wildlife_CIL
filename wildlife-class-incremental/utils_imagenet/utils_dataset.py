import argparse
import os
import shutil
import time
import numpy as np
import torch.utils.data as data
from torchvision.datasets.folder import default_loader

#split trainset.imgs
def split_images_labels(imgs):
    images = []
    labels = []
    for item in imgs:
        images.append(item[0])
        labels.append(item[1])

    return np.array(images), np.array(labels)

#merge into trainset.imgs
def merge_images_labels(images, labels):
    images = list(images)
    labels = list(labels)
    assert(len(images)==len(labels))
    imgs = []
    for i in range(len(images)):
        item = (images[i], labels[i])
        imgs.append(item)
    
    return imgs


class D3DatasetLoader(data.Dataset):
    def __init__(self, root, train, transform=None, target_transform=None):
        samples = []
        if train:
            train_data = open('d3_train.txt', 'r').readlines()
            for x in train_data:
                cls, image = x.strip().split(' ')
                item = (os.path.join(root, image), int(cls))
                samples.append(item)
        else:
            test_data = open('d3_test.txt', 'r').readlines()
            for x in test_data:
                cls, image = x.strip().split(' ')
                item = (os.path.join(root, image), int(cls))
                samples.append(item)

        self.root = root
        self.loader = default_loader
        self.imgs = self.samples = samples
        self.targets = [s[1] for s in samples]

        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        path, target = self.samples[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, target

    def __len__(self):
        return len(self.samples)