#!/usr/bin/python3

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import torch

from torch.utils.data import Dataset


class TrainDataset(Dataset):
    def __init__(self, quadruples, nentity, nrelation, negative_sample_size, negative_time_sample_size, mode,
                 type_index, type_reverse_index):
        self.len = len(quadruples)
        self.quadruples = quadruples
        self.quadruple_set = set(quadruples)
        self.nentity = nentity
        self.nrelation = nrelation
        self.negative_sample_size = negative_sample_size
        self.negative_time_sample_size = negative_time_sample_size
        self.mode = mode
        self.count = self.count_frequency(quadruples)
        self.true_head, self.true_tail = self.get_true_head_and_tail(self.quadruples)
        self.type_index = type_index
        self.type_reverse_index = type_reverse_index

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        positive_sample = self.quadruples[idx]

        head, relation, tail, day, hour = positive_sample

        subsampling_weight = self.count[(head, relation)] + self.count[(tail, -relation-1)]
        subsampling_weight = torch.sqrt(1 / torch.Tensor([subsampling_weight]))

        negative_sample_list = []
        negative_sample_size = 0

        while negative_sample_size < self.negative_sample_size:
            if self.type_index is None:
                negative_sample = np.random.randint(self.nentity, size=self.negative_sample_size*2)
            else:
                if self.mode == 'head-batch':
                    sample_space = self.type_index[self.type_reverse_index[head]]
                elif self.mode == 'tail-batch':
                    sample_space = self.type_index[self.type_reverse_index[tail]]
                negative_sample = np.random.choice(sample_space, size=self.negative_sample_size*2)
            if self.mode == 'head-batch':
                mask = np.in1d(
                    negative_sample,
                    self.true_head[(relation, tail)],
                    assume_unique=True,
                    invert=True
                )
            elif self.mode == 'tail-batch':
                mask = np.in1d(
                    negative_sample,
                    self.true_tail[(head, relation)],
                    assume_unique=True,
                    invert=True
                )
            else:
                raise ValueError('Training batch mode %s not supported' % self.mode)
            negative_sample = negative_sample[mask]
            negative_sample_list.append(negative_sample)
            negative_sample_size += negative_sample.size

        negative_sample = np.concatenate(negative_sample_list)[:self.negative_sample_size]

        negative_time_sample_list = []
        negative_time_sample_size = 0

        while negative_time_sample_size < self.negative_time_sample_size:
            negative_time_sample_day = np.random.randint(1, 32, size=self.negative_time_sample_size*2)
            negative_time_sample_hour = np.random.randint(24, size=self.negative_time_sample_size*2)
            negative_time_sample = np.stack([negative_time_sample_day, negative_time_sample_hour], axis=1)

            mask = (negative_time_sample[:, 0] != day) | (negative_time_sample[:, 1] != hour)

            negative_time_sample = negative_time_sample[mask]
            negative_time_sample_list.append(negative_time_sample)
            negative_time_sample_size += negative_time_sample.size

        if len(negative_time_sample_list) != 0:
            negative_time_sample = np.concatenate(negative_time_sample_list)[:self.negative_time_sample_size]
        else:
            negative_time_sample = np.array([])

        negative_sample = torch.from_numpy(negative_sample)
        negative_time_sample = torch.from_numpy(negative_time_sample)

        positive_sample = torch.LongTensor(positive_sample)

        return positive_sample, negative_sample, negative_time_sample, subsampling_weight, self.mode

    @staticmethod
    def collate_fn(data):
        positive_sample = torch.stack([_[0] for _ in data], dim=0)
        negative_sample = torch.stack([_[1] for _ in data], dim=0)
        negative_time_sample = torch.stack([_[2] for _ in data], dim=0)
        subsample_weight = torch.cat([_[3] for _ in data], dim=0)
        mode = data[0][4]
        return positive_sample, negative_sample, negative_time_sample, subsample_weight, mode

    @staticmethod
    def count_frequency(quadruples, start=4):
        '''
        Get frequency of a partial quadruple like (head, relation) or (relation, tail)
        The frequency will be used for subsampling like word2vec
        '''
        count = {}
        for head, relation, tail, _, _ in quadruples:
            if (head, relation) not in count:
                count[(head, relation)] = start
            else:
                count[(head, relation)] += 1

            if (tail, -relation-1) not in count:
                count[(tail, -relation-1)] = start
            else:
                count[(tail, -relation-1)] += 1
        return count

    @staticmethod
    def get_true_head_and_tail(quadruples):
        '''
        Build a dictionary of true quadruples that will
        be used to filter these true quadruples for negative sampling
        '''

        true_head = {}
        true_tail = {}

        for head, relation, tail, _, _ in quadruples:
            if (head, relation) not in true_tail:
                true_tail[(head, relation)] = []
            true_tail[(head, relation)].append(tail)
            if (relation, tail) not in true_head:
                true_head[(relation, tail)] = []
            true_head[(relation, tail)].append(head)

        for relation, tail in true_head:
            true_head[(relation, tail)] = np.array(list(set(true_head[(relation, tail)])))
        for head, relation in true_tail:
            true_tail[(head, relation)] = np.array(list(set(true_tail[(head, relation)])))

        return true_head, true_tail


class TestDataset(Dataset):
    def __init__(self, quadruples, all_true_quadruples, nentity, nrelation, mode):
        self.len = len(quadruples)
        self.quadruple_set = set(all_true_quadruples)
        self.quadruples = quadruples
        self.nentity = nentity
        self.nrelation = nrelation
        self.mode = mode

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        head, relation, tail, day, hour = self.quadruples[idx]

        if self.mode == 'head-batch':
            tmp = [(0, rand_head) if (rand_head, relation, tail, day, hour) not in self.quadruple_set
                   else (-1, head) for rand_head in range(self.nentity)]
            tmp[head] = (0, head)
        elif self.mode == 'tail-batch':
            tmp = [(0, rand_tail) if (head, relation, rand_tail, day, hour) not in self.quadruple_set
                   else (-1, tail) for rand_tail in range(self.nentity)]
            tmp[tail] = (0, tail)
        elif self.mode == 'time':
            tmp = []
            for rand_day in range(1, 32):
                for rand_hour in range(24):
                    if (head, relation, tail, rand_day, rand_hour) not in self.quadruple_set:
                        tmp.append((0, rand_day, rand_hour))
                    else:
                        tmp.append((-1, day, hour))
            tmp[(day - 1) * 24 + hour] = (0, day, hour)
        else:
            raise ValueError('negative batch mode %s not supported' % self.mode)

        tmp = torch.LongTensor(tmp)
        filter_bias = tmp[:, 0].float()
        if self.mode != 'time':
            negative_sample = tmp[:, 1]
        else:
            negative_sample = tmp[:, 1:3]

        positive_sample = torch.LongTensor((head, relation, tail, day, hour))

        return positive_sample, negative_sample, filter_bias, self.mode

    @staticmethod
    def collate_fn(data):
        positive_sample = torch.stack([_[0] for _ in data], dim=0)
        negative_sample = torch.stack([_[1] for _ in data], dim=0)
        filter_bias = torch.stack([_[2] for _ in data], dim=0)
        mode = data[0][3]
        return positive_sample, negative_sample, filter_bias, mode


class BidirectionalOneShotIterator(object):
    def __init__(self, dataloader_head, dataloader_tail):
        self.iterator_head = self.one_shot_iterator(dataloader_head)
        self.iterator_tail = self.one_shot_iterator(dataloader_tail)
        self.step = 0

    def __next__(self):
        self.step += 1
        if self.step % 2 == 0:
            data = next(self.iterator_head)
        else:
            data = next(self.iterator_tail)
        return data

    @staticmethod
    def one_shot_iterator(dataloader):
        '''
        Transform a PyTorch Dataloader into python iterator
        '''
        while True:
            for data in dataloader:
                yield data
