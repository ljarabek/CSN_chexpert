"""Edited to for SERVER run of CheXpert and MIMIC full"""

import os

import pandas as pd
import torch
from PIL import Image, ImageFile
from torch.utils.data import Dataset
from torchvision import transforms

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


class CXRDataset(Dataset):
    """Base Dataset class"""

    def __init__(self, root_dir, dataset_source = '/home/filip_relander/Medical/chexpert/data/2Label/', dataset_type='2Ltrain_Chex&MIMIC_Shuffle_Frontal', transform=None, additional_ch_dir=None,
                 specific_image = None):
        self.image_dir = root_dir
        self.transform = transform
        self.index_dir = os.path.join(dataset_source, dataset_type + '.csv')
        self.specific_image = specific_image
        self.classes = pd.read_csv(self.index_dir, header=None, nrows=1).iloc[0, :].values[1:]
        self.label_index = pd.read_csv(self.index_dir, header=0)
        self.add_ch_dir = None
        # this is for training based on segmentation
        if additional_ch_dir is not None:
            self.add_ch_dir = additional_ch_dir

    def __getitem__(self, idx):
        img_path = self.label_index.iloc[idx, 0]
        if self.specific_image is None:
            img_dir = os.path.join(self.image_dir, img_path)
        else:
            img_dir = self.specific_image
        image = Image.open(img_dir).convert('RGB')
        label = self.label_index.iloc[idx, 1:].values.astype('int')
        if self.add_ch_dir is not None:
            add_ch_dir = os.path.join(self.add_ch_dir, img_path)
            add_ch = Image.open(add_ch_dir).convert('RGB')  # 3-dim mode

        # argumentation
        assert self.transform is not None, 'Please specify the transform!'
        image = self.transform(image)

        if self.add_ch_dir is not None:
            add_ch = self.transform(add_ch)
            return image, label, img_path, add_ch
        #print(label[4])
        return image, label, img_path

    def __len__(self):
        return int(len(self.label_index))


if __name__ == '__main__':
    # --- Data Debug Code ---
    trans = transforms.Compose([
        transforms.Resize(512),
        transforms.RandomCrop(size=512),
        transforms.ToTensor()])

    dl = CXRDataset(dataset_source='/home/filip/DeployedProjects/CSN/data/2Label/',root_dir='/raid/Medical/DX/', transform=trans)
    train_loader = torch.utils.data.DataLoader(dl, batch_size=32, shuffle=False, num_workers=32,
                                               pin_memory=True, drop_last=False)
    for i, batch in enumerate(train_loader):
        img, label, name = batch
        print(i, label, name)
    print('Done!')
