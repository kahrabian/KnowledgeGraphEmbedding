#!/usr/bin/python3

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from sklearn.metrics import average_precision_score

from torch.utils.data import DataLoader

from dataloader import TestDataset


class KGEModel(nn.Module):
    def __init__(self, model_name, nentity, nrelation, hidden_dim, time_hidden_dim, gamma, epsilon,
                 double_entity_embedding=False, double_relation_embedding=False, double_time_embedding=False,
                 type_index=None, type_reverse_index=None, issue_users_idx=None):
        super(KGEModel, self).__init__()
        self.model_name = model_name
        self.nentity = nentity
        self.nrelation = nrelation
        self.hidden_dim = hidden_dim
        self.epsilon = epsilon

        self.type_index = type_index
        self.type_reverse_index = type_reverse_index
        self.issue_users_idx = issue_users_idx

        self.gamma = nn.Parameter(
            torch.Tensor([gamma]),
            requires_grad=False
        )

        self.entity_dim = hidden_dim*2 if double_entity_embedding else hidden_dim
        self.relation_dim = hidden_dim*2 if double_relation_embedding else hidden_dim
        self.time_dim = time_hidden_dim*2 if double_time_embedding else time_hidden_dim

        self.relation_dim += (self.time_dim // 2) if double_entity_embedding and \
            not double_relation_embedding else self.time_dim

        self.embedding_range = nn.Parameter(
            torch.Tensor([(self.gamma.item() + self.epsilon) / self.relation_dim]),
            requires_grad=False
        )

        self.entity_embedding = nn.Parameter(torch.zeros(nentity, self.entity_dim))
        nn.init.uniform_(
            tensor=self.entity_embedding,
            a=-self.embedding_range.item(),
            b=self.embedding_range.item()
        )

        self.relation_embedding = nn.Parameter(torch.zeros(nrelation, self.relation_dim))
        nn.init.uniform_(
            tensor=self.relation_embedding,
            a=-self.embedding_range.item(),
            b=self.embedding_range.item()
        )

        self.d_frq_embedding = nn.Parameter(torch.zeros(nentity, self.time_dim))
        nn.init.uniform_(
            tensor=self.d_frq_embedding,
            a=-self.embedding_range.item(),
            b=self.embedding_range.item()
        )

        self.d_phi_embedding = nn.Parameter(torch.zeros(nentity, self.time_dim))
        nn.init.uniform_(
            tensor=self.d_phi_embedding,
            a=-self.embedding_range.item(),
            b=self.embedding_range.item()
        )

        self.d_amp_embedding = nn.Parameter(torch.zeros(nentity, self.time_dim))
        nn.init.uniform_(
            tensor=self.d_amp_embedding,
            a=-self.embedding_range.item(),
            b=self.embedding_range.item()
        )

        if model_name == 'pRotatE':
            self.modulus = nn.Parameter(torch.Tensor([[0.5 * self.embedding_range.item()]]))

        # Do not forget to modify this line when you add a new model in the "forward" function
        if model_name not in ['TransE', 'DistMult', 'ComplEx', 'RotatE', 'pRotatE']:
            raise ValueError('model %s not supported' % model_name)

        if model_name == 'RotatE' and \
                (not double_entity_embedding or double_relation_embedding or not double_time_embedding):
            raise ValueError('RotatE should use --double_entity_embedding')

        if model_name == 'ComplEx' and \
                (not double_entity_embedding or not double_relation_embedding or not double_time_embedding):
            raise ValueError('ComplEx should use --double_entity_embedding and --double_relation_embedding')

    def time_embedding(self, entity, day):
        d_amp = torch.index_select(self.d_amp_embedding, dim=0, index=entity)
        d_frq = torch.index_select(self.d_frq_embedding, dim=0, index=entity)
        d_phi = torch.index_select(self.d_phi_embedding, dim=0, index=entity)
        return d_amp * torch.sin(day * d_frq + d_phi)

    def forward(self, sample, mode='single'):
        '''
        Forward function that calculate the score of a batch of quadruples.
        In the 'single' mode, sample is a batch of quadruple.
        In the 'head-batch' or 'tail-batch' mode, sample consists two part.
        The first part is usually the positive sample.
        And the second part is the entities in the negative samples.
        Because negative samples and positive samples usually share two elements 
        in their quadruple ((head, relation) or (relation, tail)).
        '''

        if mode == 'single':
            batch_size, negative_sample_size = sample.size(0), 1

            day = sample[:, 3]

            head = torch.index_select(
                self.entity_embedding,
                dim=0,
                index=sample[:, 0]
            ).unsqueeze(1)

            head_time = self.time_embedding(sample[:, 0], day.view(-1, 1)).unsqueeze(1)

            relation = torch.index_select(
                self.relation_embedding,
                dim=0,
                index=sample[:, 1]
            ).unsqueeze(1)

            tail = torch.index_select(
                self.entity_embedding,
                dim=0,
                index=sample[:, 2]
            ).unsqueeze(1)

            tail_time = self.time_embedding(sample[:, 2], day.view(-1, 1)).unsqueeze(1)

            time_neg = None

        elif mode == 'head-batch':
            tail_part, head_part, time_part = sample
            batch_size, negative_sample_size = head_part.size(0), head_part.size(1)
            negative_time_sample_size = time_part.size(1)

            day = tail_part[:, 3]

            head = torch.index_select(
                self.entity_embedding,
                dim=0,
                index=head_part.view(-1)
            ).view(batch_size, negative_sample_size, -1)

            head_time = self.time_embedding(
                head_part.view(-1),
                day.repeat(negative_sample_size, 1).t().contiguous().view(-1, 1)
            ).view(batch_size, negative_sample_size, -1)

            relation = torch.index_select(
                self.relation_embedding,
                dim=0,
                index=tail_part[:, 1]
            ).unsqueeze(1)

            tail = torch.index_select(
                self.entity_embedding,
                dim=0,
                index=tail_part[:, 2]
            ).unsqueeze(1)

            tail_time = self.time_embedding(tail_part[:, 2], day.view(-1, 1)).unsqueeze(1)

            true_head = torch.index_select(
                self.entity_embedding,
                dim=0,
                index=tail_part[:, 0]
            ).unsqueeze(1)

            head_time_neg = self.time_embedding(
                tail_part[:, 0].repeat(negative_time_sample_size, 1).t().contiguous().view(-1),
                time_part.view(-1, 1),
            ).view(batch_size, negative_time_sample_size, -1)

            tail_time_neg = self.time_embedding(
                tail_part[:, 2].repeat(negative_time_sample_size, 1).t().contiguous().view(-1),
                time_part.view(-1, 1),
            ).view(batch_size, negative_time_sample_size, -1)

            time_neg = (true_head, tail, head_time_neg, tail_time_neg)

        elif mode == 'tail-batch':
            head_part, tail_part, time_part = sample
            batch_size, negative_sample_size = tail_part.size(0), tail_part.size(1)
            negative_time_sample_size = time_part.size(1)

            day = head_part[:, 3]

            head = torch.index_select(
                self.entity_embedding,
                dim=0,
                index=head_part[:, 0]
            ).unsqueeze(1)

            head_time = self.time_embedding(head_part[:, 0], day.view(-1, 1)).unsqueeze(1)

            relation = torch.index_select(
                self.relation_embedding,
                dim=0,
                index=head_part[:, 1]
            ).unsqueeze(1)

            tail = torch.index_select(
                self.entity_embedding,
                dim=0,
                index=tail_part.view(-1)
            ).view(batch_size, negative_sample_size, -1)

            tail_time = self.time_embedding(
                tail_part.view(-1),
                day.repeat(tail_part.shape[1], 1).t().contiguous().view(-1, 1)
            ).view(batch_size, negative_sample_size, -1)

            true_tail = torch.index_select(
                self.entity_embedding,
                dim=0,
                index=head_part[:, 2]
            ).unsqueeze(1)

            head_time_neg = self.time_embedding(
                head_part[:, 0].repeat(negative_time_sample_size, 1).t().contiguous().view(-1),
                time_part.view(-1, 1),
            ).view(batch_size, negative_time_sample_size, -1)

            tail_time_neg = self.time_embedding(
                head_part[:, 2].repeat(negative_time_sample_size, 1).t().contiguous().view(-1),
                time_part.view(-1, 1),
            ).view(batch_size, negative_time_sample_size, -1)

            time_neg = (head, true_tail, head_time_neg, tail_time_neg)

        elif mode == 'time':
            head_part, tail_part, time_part = sample
            batch_size, negative_sample_size = tail_part.size(0), tail_part.size(1)
            negative_time_sample_size = time_part.size(1)

            day = head_part[:, 3]

            head = torch.index_select(
                self.entity_embedding,
                dim=0,
                index=head_part[:, 0]
            ).unsqueeze(1)

            head_time = self.time_embedding(head_part[:, 0], day.view(-1, 1)).unsqueeze(1)

            relation = torch.index_select(
                self.relation_embedding,
                dim=0,
                index=head_part[:, 1]
            ).unsqueeze(1)

            tail = None

            tail_time = None

            true_tail = torch.index_select(
                self.entity_embedding,
                dim=0,
                index=head_part[:, 2]
            ).unsqueeze(1)

            head_time_neg = self.time_embedding(
                head_part[:, 0].repeat(negative_time_sample_size, 1).t().contiguous().view(-1),
                time_part.view(-1, 1),
            ).view(batch_size, negative_time_sample_size, -1)

            tail_time_neg = self.time_embedding(
                head_part[:, 2].repeat(negative_time_sample_size, 1).t().contiguous().view(-1),
                time_part.view(-1, 1),
            ).view(batch_size, negative_time_sample_size, -1)

            time_neg = (head, true_tail, head_time_neg, tail_time_neg)

        else:
            raise ValueError('mode %s not supported' % mode)

        model_func = {
            'TransE': self.TransE,
            'DistMult': self.DistMult,
            'ComplEx': self.ComplEx,
            'RotatE': self.RotatE,
            'pRotatE': self.pRotatE
        }

        if self.model_name in model_func:
            score = model_func[self.model_name](head, relation, tail, head_time, tail_time, time_neg, mode)
        else:
            raise ValueError('model %s not supported' % self.model_name)

        return score

    def TransE(self, head, relation, tail, head_time, tail_time, mode):
        head = torch.cat([head, head_time], dim=2)
        tail = torch.cat([tail, tail_time], dim=2)

        if mode == 'head-batch':
            score = head + (relation - tail)
        else:
            score = (head + relation) - tail

        score = self.gamma.item() - torch.norm(score, p=1, dim=2)
        return score

    def DistMult(self, head, relation, tail, head_time, tail_time, mode):
        head = torch.cat([head, head_time], dim=2)
        tail = torch.cat([tail, tail_time], dim=2)

        if mode == 'head-batch':
            score = head * (relation * tail)
        else:
            score = (head * relation) * tail

        score = score.sum(dim=2)
        return score

    def ComplEx(self, head, relation, tail, head_time, tail_time, mode):
        re_head, im_head = torch.chunk(head, 2, dim=2)
        re_relation, im_relation = torch.chunk(relation, 2, dim=2)
        re_tail, im_tail = torch.chunk(tail, 2, dim=2)

        re_head_time, im_head_time = torch.chunk(head_time, 2, dim=2)
        re_tail_time, im_tail_time = torch.chunk(tail_time, 2, dim=2)

        re_head = torch.cat([re_head, re_head_time], dim=2)
        im_head = torch.cat([im_head, im_head_time], dim=2)

        re_tail = torch.cat([re_tail, re_tail_time], dim=2)
        im_tail = torch.cat([im_tail, im_tail_time], dim=2)

        if mode == 'head-batch':
            re_score = re_relation * re_tail + im_relation * im_tail
            im_score = re_relation * im_tail - im_relation * re_tail
            score = re_head * re_score + im_head * im_score
        else:
            re_score = re_head * re_relation - im_head * im_relation
            im_score = re_head * im_relation + im_head * re_relation
            score = re_score * re_tail + im_score * im_tail

        score = score.sum(dim=2)
        return score

    def RotatE(self, head, relation, tail, head_time, tail_time, time_neg, mode):
        pi = 3.14159265358979323846

        if mode != 'time':
            re_head, im_head = torch.chunk(head, 2, dim=2)
            re_tail, im_tail = torch.chunk(tail, 2, dim=2)

        if mode != 'time':
            re_head_time, im_head_time = torch.chunk(head_time, 2, dim=2)
            re_tail_time, im_tail_time = torch.chunk(tail_time, 2, dim=2)

        if mode != 'single':
            re_head_time_neg, im_head_time_neg = torch.chunk(time_neg[2], 2, dim=2)
            re_tail_time_neg, im_tail_time_neg = torch.chunk(time_neg[3], 2, dim=2)

        if mode != 'single':
            re_true_head, im_true_head = torch.chunk(time_neg[0], 2, dim=2)
            re_head_neg = torch.cat([re_true_head.repeat(1, re_head_time_neg.size(1), 1), re_head_time_neg], dim=2)
            im_head_neg = torch.cat([im_true_head.repeat(1, im_head_time_neg.size(1), 1), im_head_time_neg], dim=2)

        if mode != 'time':
            re_head = torch.cat([re_head, re_head_time], dim=2)
            im_head = torch.cat([im_head, im_head_time], dim=2)

        if mode != 'single':
            re_true_tail, im_true_tail = torch.chunk(time_neg[1], 2, dim=2)
            re_tail_neg = torch.cat([re_true_tail.repeat(1, re_tail_time_neg.size(1), 1), re_tail_time_neg], dim=2)
            im_tail_neg = torch.cat([re_true_tail.repeat(1, im_tail_time_neg.size(1), 1), im_tail_time_neg], dim=2)

        if mode != 'time':
            re_tail = torch.cat([re_tail, re_tail_time], dim=2)
            im_tail = torch.cat([im_tail, im_tail_time], dim=2)

        # Make phases of relations uniformly distributed in [-pi, pi]

        phase_relation = relation/(self.embedding_range.item()/pi)

        re_relation = torch.cos(phase_relation)
        im_relation = torch.sin(phase_relation)

        if mode != 'single':
            re_relation_neg = re_relation.repeat(1, re_head_neg.size(1), 1)
            im_relation_neg = im_relation.repeat(1, im_tail_neg.size(1), 1)

        if mode == 'head-batch':
            re_score = re_relation * re_tail + im_relation * im_tail
            im_score = re_relation * im_tail - im_relation * re_tail
            re_score = re_score - re_head
            im_score = im_score - im_head

            re_score_neg = re_relation_neg * re_tail_neg + im_relation_neg * im_tail_neg
            im_score_neg = re_relation_neg * im_tail_neg - im_relation_neg * re_tail_neg
            re_score_neg = re_score_neg - re_head_neg
            im_score_neg = im_score_neg - im_head_neg

            re_score = torch.cat([re_score, re_score_neg], dim=1)
            im_score = torch.cat([im_score, im_score_neg], dim=1)
        elif mode == 'tail-batch':
            re_score = re_head * re_relation - im_head * im_relation
            im_score = re_head * im_relation + im_head * re_relation
            re_score = re_score - re_tail
            im_score = im_score - im_tail

            re_score_neg = re_head_neg * re_relation_neg - im_head_neg * im_relation_neg
            im_score_neg = im_head_neg * re_relation_neg + re_head_neg * im_relation_neg
            re_score_neg = re_score_neg - re_tail_neg
            im_score_neg = im_score_neg - im_tail_neg

            re_score = torch.cat([re_score, re_score_neg], dim=1)
            im_score = torch.cat([im_score, im_score_neg], dim=1)
        elif mode == 'time':
            re_score_neg = re_head_neg * re_relation_neg - im_head_neg * im_relation_neg
            im_score_neg = im_head_neg * re_relation_neg + re_head_neg * im_relation_neg
            re_score_neg = re_score_neg - re_tail_neg
            im_score_neg = im_score_neg - im_tail_neg

            re_score = re_score_neg
            im_score = im_score_neg
        else:
            re_score = re_head * re_relation - im_head * im_relation
            im_score = re_head * im_relation + im_head * re_relation
            re_score = re_score - re_tail
            im_score = im_score - im_tail

        score = torch.stack([re_score, im_score], dim=0)
        score = score.norm(dim=0)

        score = self.gamma.item() - score.sum(dim=2)
        return score

    def pRotatE(self, head, relation, tail, head_time, tail_time, mode):
        pi = 3.14159262358979323846

        # Make phases of entities and relations uniformly distributed in [-pi, pi]

        phase_head = head/(self.embedding_range.item()/pi)
        phase_relation = relation/(self.embedding_range.item()/pi)
        phase_tail = tail/(self.embedding_range.item()/pi)

        if mode == 'head-batch':
            score = phase_head + (phase_relation - phase_tail)
        else:
            score = (phase_head + phase_relation) - phase_tail

        score = torch.sin(score)
        score = torch.abs(score)

        score = self.gamma.item() - score.sum(dim=2) * self.modulus
        return score

    @staticmethod
    def train_step(model, optimizer, train_iterator, args):
        '''
        A single train step. Apply back-propation and return the loss
        '''

        model.train()

        optimizer.zero_grad()

        positive_sample, negative_sample, negative_time_sample, subsampling_weight, mode = next(train_iterator)

        if args.cuda:
            positive_sample = positive_sample.cuda()
            negative_sample = negative_sample.cuda()
            negative_time_sample = negative_time_sample.cuda()
            subsampling_weight = subsampling_weight.cuda()

        negative_score = model((positive_sample, negative_sample, negative_time_sample), mode=mode)

        if args.negative_adversarial_sampling:
            # In self-adversarial sampling, we do not apply back-propagation on the sampling weight
            negative_score = (F.softmax(negative_score * args.adversarial_temperature, dim=1).detach()
                              * F.logsigmoid(-negative_score)).sum(dim=1)
        else:
            negative_score = F.logsigmoid(-negative_score).mean(dim=1)

        positive_score = model(positive_sample)

        positive_score = F.logsigmoid(positive_score).squeeze(dim=1)

        if args.uni_weight:
            positive_sample_loss = - positive_score.mean()
            negative_sample_loss = - negative_score.mean()
        else:
            positive_sample_loss = - (subsampling_weight * positive_score).sum()/subsampling_weight.sum()
            negative_sample_loss = - (subsampling_weight * negative_score).sum()/subsampling_weight.sum()

        loss = (positive_sample_loss + negative_sample_loss)/2

        if args.regularization != 0.0:
            # Use L3 regularization for ComplEx and DistMult
            regularization = args.regularization * (
                model.entity_embedding.norm(p=3)**3 +
                model.relation_embedding.norm(p=3).norm(p=3)**3
            )
            loss = loss + regularization
            regularization_log = {'regularization': regularization.item()}
        else:
            regularization_log = {}

        loss.backward()

        optimizer.step()

        log = {
            **regularization_log,
            'positive_sample_loss': positive_sample_loss.item(),
            'negative_sample_loss': negative_sample_loss.item(),
            'loss': loss.item()
        }

        return log

    @staticmethod
    def test_step(model, test_quadruples, all_true_quadruples, args):
        '''
        Evaluate the model on test or valid datasets
        '''

        model.eval()

        if args.countries:
            # Countries S* datasets are evaluated on AUC-PR
            # Process test data for AUC-PR evaluation
            sample = list()
            y_true = list()
            for head, relation, tail in test_quadruples:
                for candidate_region in args.regions:
                    y_true.append(1 if candidate_region == tail else 0)
                    sample.append((head, relation, candidate_region))

            sample = torch.LongTensor(sample)
            if args.cuda:
                sample = sample.cuda()

            with torch.no_grad():
                y_score = model(sample).squeeze(1).cpu().numpy()

            y_true = np.array(y_true)

            # average_precision_score is the same as auc_pr
            auc_pr = average_precision_score(y_true, y_score)

            metrics = {'auc_pr': auc_pr}

        else:
            # Otherwise use standard (filtered) MRR, MR, HITS@1, HITS@3, and HITS@10 metrics
            # Prepare dataloader for evaluation
            if args.eval_mode != 'time':
                test_dataloader_head = DataLoader(
                    TestDataset(
                        test_quadruples,
                        all_true_quadruples,
                        args.nentity,
                        args.nrelation,
                        'head-batch'
                    ),
                    batch_size=args.test_batch_size,
                    num_workers=max(1, args.cpu_num//2),
                    collate_fn=TestDataset.collate_fn
                )

                test_dataloader_tail = DataLoader(
                    TestDataset(
                        test_quadruples,
                        all_true_quadruples,
                        args.nentity,
                        args.nrelation,
                        'tail-batch'
                    ),
                    batch_size=args.test_batch_size,
                    num_workers=max(1, args.cpu_num//2),
                    collate_fn=TestDataset.collate_fn
                )

                test_dataset_list = []
                if args.eval_mode != 'tail':
                    test_dataset_list.append(test_dataloader_head)
                if args.eval_mode != 'head':
                    test_dataset_list.append(test_dataloader_tail)
            else:
                test_dataloader_time = DataLoader(
                    TestDataset(
                        test_quadruples,
                        all_true_quadruples,
                        args.nentity,
                        args.nrelation,
                        'time'
                    ),
                    batch_size=args.test_batch_size,
                    num_workers=max(1, args.cpu_num//2),
                    collate_fn=TestDataset.collate_fn
                )
                test_dataset_list = [test_dataloader_time, ]

            logs = []

            step = 0
            total_steps = sum([len(dataset) for dataset in test_dataset_list])

            with torch.no_grad():
                for test_dataset in test_dataset_list:
                    for positive_sample, negative_sample, negative_time_sample, filter_bias, mode in test_dataset:
                        if args.cuda:
                            positive_sample = positive_sample.cuda()
                            negative_sample = negative_sample.cuda()
                            negative_time_sample = negative_time_sample.cuda()
                            filter_bias = filter_bias.cuda()

                        batch_size = positive_sample.size(0)

                        score = model((positive_sample, negative_sample, negative_time_sample), mode)
                        score += filter_bias

                        # Explicitly sort all the entities to ensure that there is no test exposure bias
                        argsort = torch.argsort(score, dim=1, descending=True)

                        if mode == 'head-batch':
                            positive_arg = positive_sample[:, 0]
                            positive_issue_idx = positive_sample[:, 2]
                        elif mode == 'tail-batch':
                            positive_arg = positive_sample[:, 2]
                            positive_issue_idx = positive_sample[:, 0]
                        elif mode == 'time':
                            positive_arg = positive_sample[:, 3] - 26
                        else:
                            raise ValueError('mode %s not supported' % mode)

                        for i in range(batch_size):
                            # Notice that argsort is not ranking
                            ranking = (argsort[i, :] == positive_arg[i]).nonzero()
                            assert ranking.size(0) == 1

                            # ranking + 1 is the true ranking used in evaluation metrics
                            ranking = 1 + ranking.item()

                            if mode != 'time' and model.type_index is not None:
                                index = model.type_index[model.type_reverse_index[positive_arg[i].item()]]
                                if model.issue_users_idx is None:
                                    ranking = np.isin(argsort[i, :].cpu().numpy(), index)[:ranking].sum()
                                else:
                                    issue_index = model.issue_users_idx.get(positive_issue_idx[i].item(), [])
                                    issue_ranking = np.isin(argsort[i, :].cpu().numpy(), issue_index)[:ranking].sum()
                                    if issue_ranking == 0:
                                        ranking = np.isin(argsort[i, :].cpu().numpy(), index)[:ranking].sum()
                                    else:
                                        ranking = issue_ranking
                            elif mode != 'time':
                                pass

                            logs.append({
                                'MRR': 1.0/ranking,
                                'MR': float(ranking),
                                'HITS@1': 1.0 if ranking <= 1 else 0.0,
                                'HITS@3': 1.0 if ranking <= 3 else 0.0,
                                'HITS@10': 1.0 if ranking <= 10 else 0.0,
                            })

                        if step % args.test_log_steps == 0:
                            logging.info('Evaluating the model... (%d/%d)' % (step, total_steps))

                        step += 1

            metrics = {}
            for metric in logs[0].keys():
                metrics[metric] = sum([log[metric] for log in logs])/len(logs)

        return metrics
