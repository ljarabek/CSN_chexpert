"""Edited to for SERVER run of CheXpert and MIMIC full"""

import os

import pandas as pd
import torch
from PIL import Image, ImageFile
from torch.utils.data import Dataset
import torch.utils.data as data
from torchvision import transforms
from tqdm import tqdm
import numpy as np

ImageFile.LOAD_TRUNCATED_IMAGES = True

disease_categories = {
    'Atelectasis': 0,
    'Cardiomegaly': 1,
    'Consolidation': 2,
    'Edema': 3,
    'Pleural Effusion': 4,
    'No Finding': 5,
    'Enlarged Cardiomediastinum': 6,
    'Lung Opacity': 7,
    'Lung Lesion': 8,
    'Pneumonia': 9,
    'Pneumothorax': 10,
    'Pleural Other': 11,
    'Fracture': 12,
    'Support Devices': 13,
}

validation_categories = {
    'Atelectasis': 0,
    'Cardiomegaly': 1,
    'Consolidation': 2,
    'Edema': 3,
    'Pleural Effusion': 4,
    'No Finding': 5,
}


class CXRDataset_binary(Dataset):
    """Base Dataset class"""

    def __init__(self, root_dir, dataset_source='/home/filip_relander/Medical/chexpert/data/2Label/',
                 dataset_type='2Ltrain_Chex&MIMIC_Shuffle_Frontal', transform=None, additional_ch_dir=None, tag = None):
        self.image_dir = root_dir
        self.tag = tag
        self.transform = transform
        self.index_dir = os.path.join(dataset_source, dataset_type + '.csv')

        self.classes = pd.read_csv(self.index_dir, header=None, nrows=1).iloc[0, :].values[1:]
        self.label_index = pd.read_csv(self.index_dir, header=0)
        self.add_ch_dir = None
        # this is for training based on segmentation
        if additional_ch_dir is not None:
            self.add_ch_dir = additional_ch_dir

    def __getitem__(self, idx):
        img_path = self.label_index.iloc[idx, 0]
        img_dir = os.path.join(self.image_dir, img_path)
        ##TODO image = Image.open(img_dir).convert('RGB')
        label = self.label_index.iloc[idx, 1:].values.astype('int')
        if self.add_ch_dir is not None:
            add_ch_dir = os.path.join(self.add_ch_dir, img_path)
            add_ch = Image.open(add_ch_dir).convert('RGB')  # 3-dim mode

        # argumentation
        assert self.transform is not None, 'Please specify the transform!'
        ##TODO image = self.transform(image)

        #if self.tag is not None:
        #    label = label[self.tag]

        if self.add_ch_dir is not None:
            add_ch = self.transform(add_ch)
            return None, label, img_path, add_ch ##TODO image

        return None, label, img_path ##TODO image

    def __len__(self):
        return int(len(self.label_index))

def sample_weights(dataset, tag,args):
    """
    :param dataset:
    :param tag: (int) tag index w.r.t. that is ranndomly sampled
    :return: size of new dataset (2*frequency), weights (~probabilities) per sample, such that when sampling with p, you get 50:50 pos:neg
    """
    try:
        labels = np.load("labels.npy") ### TODO: CSVs are BAAD!!!
    except:
        print("labels.npy was not found, generating")
        labels = list()
        for im, label, img_path in tqdm(dataset):
            labels.append(label)
        labels = np.array(labels)
        np.save("labels.npy", labels)

    dataset_len = labels.shape[0]
    label_count = np.sum(labels,axis = 0)
    label_count_pos = label_count[tag]
    label_count_neg = dataset_len - label_count_pos

    weight_pos = label_count_neg / dataset_len
    weight_neg = label_count_pos / dataset_len

    weights = list()
    for idx, example in enumerate(labels):
        if example[tag]==0:
            weights.append(weight_neg)
        else:
            weights.append(weight_pos)
        #print(example[tag], weights[idx])


    count = int(label_count_pos*2)


    return count, weights

def balanced_dataloader(dataset, tag, args):
    """
        :param dataset:
        :param tag: (int) tag index w.r.t. that is ranndomly sampled
        :return: dataloader with balanced sampling w.r.t. tag
    """
    count, weights = sample_weights(dataset,tag,args)

    sampler = data.WeightedRandomSampler(weights=weights, num_samples=count, replacement=True)

    dataloader = data.DataLoader(dataset, batch_size=args.batch_size, sampler = sampler,num_workers=15, drop_last=True)

    return dataloader



if __name__ == '__main__':
    # --- Data Debug Code ---
    trans = transforms.Compose([
        transforms.Resize(512),
        transforms.RandomCrop(size=512),
        transforms.ToTensor()])

    ds = CXRDataset_binary(dataset_source="./2Label/", root_dir='/raid/Medical/DX/',
                           transform=trans)  # '/home/filip/DeployedProjects/CSN/data/2Label/'
    train_loader = torch.utils.data.DataLoader(ds, batch_size=5, shuffle=False, num_workers=32,
                                               pin_memory=True, drop_last=False)
    labels = np.load("labels.npy")

    sample_weights(10)

    #print(np.sum(labels,axis=0))

    # for i, batch in enumerate(train_loader):
    #    img, label, name = batch
    #    print(i, label, name)
    print('Done!')
