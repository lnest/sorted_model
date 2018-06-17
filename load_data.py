# -*- coding: utf-8 -*-

# ------------------------------------
# Create On 2018/6/5 20:49 
# File Name: load_data.py
# Edit Author: lnest
# ------------------------------------
import torch
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader


class SequenceDataset(Dataset):
    def __init__(self, csv_file, transform=None):
        self.csv_file = csv_file
        self.sequence_data = self.read_csv()
        self.transform = transform

    def read_csv(self):
        f_r = open(self.csv_file)
        lines = f_r.readlines()
        return lines

    def __len__(self):
        return len(self.sequence_data)

    def __getitem__(self, idx):
        data_in, data_out = self.sequence_data[idx].strip().split('\t')
        sample = {'input': [int(word) for word in data_in.strip().split()],
                  'output': [int(word) for word in data_out.strip().split()]}
        if self.transform:
            sample = self.transform(sample)
        return sample


class ToTensor(object):
    def __call__(self, sample):
        input_words, target_words = sample['input'], sample['output']
        return {'input': torch.tensor(input_words, dtype=torch.long),
                'output': torch.tensor(target_words, dtype=torch.long)}


sequence_dataset = SequenceDataset('./data/train', transform=transforms.Compose([ToTensor()]))
dataloader = DataLoader(sequence_dataset, batch_size=1024,
                        shuffle=True, num_workers=4)
