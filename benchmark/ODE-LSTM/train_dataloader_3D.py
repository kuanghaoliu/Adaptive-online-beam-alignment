# dataloader for training

import os
from os import listdir
from os.path import isfile, join
import scipy.io as sio
import numpy as np
import torch
import math
from collections import Counter


class train_Dataloader_3D():
    def __init__(self, path='', batch_size=32, device='cpu'):
        self.batch_size = batch_size
        self.device = device
        self.files = [join(path, f) for f in listdir(path) if isfile(join(path, f))]
        for i, f in enumerate(self.files):
            if not f.split('.')[-1] == 'mat':
                del (self.files[i])
        self.reset()

    def reset(self):
        self.done = False
        self.unvisited_files = [f for f in self.files]

        # batch_size * 2 * length * num of beam
        self.buffer = np.zeros((0, 2, 100, 64))

        # batch_size * length
        self.buffer_label = np.zeros((0, 100))

        # batch_size * length * num of beam
        self.buffer_beam_power = np.zeros((0, 100, 64))

        # batch_size * ratio_length * time_length
        self.buffer_ratio = np.zeros((0, 9, 10))

    def load(self, file):
        data = sio.loadmat(file)

        channels = data['MM_data'] # beam training received signal
        labels = data['beam_label'] - 1 # optimal beam index label
        beam_power = data['beam_power'] # beam amplitude
        # print(beam_power.shape)
        ratio = data['ratios']
        # print(ratio.shape)
        return channels, labels, beam_power, ratio

    def next_batch(self):
        done = False
        count = True

        # sequentially load data
        while self.buffer.shape[0] < self.batch_size:
            if len(self.unvisited_files) == 0:
                done = True
                count = False
                break
            channels, labels, beam_power, ratio = self.load(
                self.unvisited_files.pop(0))

            # load data into buffers
            self.buffer = np.concatenate((self.buffer, channels), axis=0)
            self.buffer_label = np.concatenate((self.buffer_label, labels), axis=0)
            self.buffer_beam_power = np.concatenate((self.buffer_beam_power, beam_power), axis=0)
            self.buffer_ratio = np.concatenate((self.buffer_ratio, ratio), axis=0)

        out_size = min(self.batch_size, self.buffer.shape[0])
        # get data from buffers
        batch_channels = self.buffer[0 : out_size, :, :, :]
        batch_labels = np.squeeze(self.buffer_label[0 : out_size, :])
        batch_beam_power = np.squeeze(self.buffer_beam_power[0:out_size, :, :])
        batch_ratio = np.squeeze(self.buffer_ratio[0:out_size, :, :])

        self.buffer = np.delete(self.buffer, np.s_[0 : out_size], 0)
        self.buffer_label = np.delete(self.buffer_label, np.s_[0 : out_size], 0)
        self.buffer_beam_power = np.delete(self.buffer_beam_power, np.s_[0:out_size], 0)
        self.buffer_ratio = np.delete(self.buffer_ratio, np.s_[0:out_size], 0)

        # format transformation for reducing overhead
        batch_channels = np.float32(batch_channels)
        batch_labels = batch_labels.astype(int)
        batch_beam_power = np.float32(batch_beam_power)
        batch_ratio = np.float32(batch_ratio)

        return torch.from_numpy(batch_channels).to(self.device), torch.from_numpy(batch_labels).to(
            self.device), torch.from_numpy(batch_beam_power).to(
            self.device), torch.from_numpy(batch_ratio).to(
            self.device),done, count