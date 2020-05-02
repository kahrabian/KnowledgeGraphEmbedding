#!/usr/bin/python3

import logging
import os
from datetime import datetime, timedelta

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from dataloader import TestDataset


class KGEModel(nn.Module):
    def __init__(self, model_name, nentity, nrelation, hidden_dim, time_hidden_dim, relative_hidden_dim,
                 gamma, epsilon, double_entity_embedding=False, double_relation_embedding=False,
                 double_time_embedding=False, double_relative_embedding=False,
                 type_index=None, type_reverse_index=None, users_idx=None):
        super(KGEModel, self).__init__()
        self.model_name = model_name
        self.nentity = nentity

        self.type_index = type_index
        self.type_reverse_index = type_reverse_index
        self.users_idx = users_idx

        self.epsilon = epsilon
        self.gamma = nn.Parameter(torch.Tensor([gamma]), requires_grad=False)

        self.stt_dim = hidden_dim * 2 if double_entity_embedding else hidden_dim
        self.abs_dim = time_hidden_dim * 2 if double_time_embedding else time_hidden_dim
        self.rel_dim = relative_hidden_dim * 2 if double_relative_embedding else relative_hidden_dim

        self.r_dim = hidden_dim * 2 if double_relation_embedding else hidden_dim
        if double_entity_embedding and not double_relation_embedding:
            self.r_dim += (self.abs_dim // 2) + (self.rel_dim // 2)
        else:
            self.r_dim += self.abs_dim + self.rel_dim

        self.emb_rng = nn.Parameter(
            torch.Tensor([(self.gamma.item() + self.epsilon) / self.r_dim]), requires_grad=False)

        self.e_emb = nn.Parameter(torch.zeros(self.nentity, self.stt_dim))
        nn.init.uniform_(tensor=self.e_emb, a=-self.emb_rng.item(), b=self.emb_rng.item())

        self.r_emb = nn.Parameter(torch.zeros(nrelation, self.r_dim))
        nn.init.uniform_(tensor=self.r_emb, a=-self.emb_rng.item(), b=self.emb_rng.item())

        self.abs_d_frq_emb = nn.Parameter(torch.zeros(self.nentity, self.abs_dim))
        self.abs_d_phi_emb = nn.Parameter(torch.zeros(self.nentity, self.abs_dim))
        self.abs_d_amp_emb = nn.Parameter(torch.zeros(self.nentity, self.abs_dim))
        nn.init.uniform_(tensor=self.abs_d_frq_emb, a=np.pi / 7, b=2 * np.pi / 3)
        nn.init.uniform_(tensor=self.abs_d_phi_emb, a=-np.pi, b=np.pi)
        nn.init.uniform_(tensor=self.abs_d_amp_emb, a=-self.emb_rng.item(), b=self.emb_rng.item())

        # self.abs_m_frq_emb = nn.Parameter(torch.zeros(self.nentity, self.abs_dim))
        # self.abs_m_phi_emb = nn.Parameter(torch.zeros(self.nentity, self.abs_dim))
        # self.abs_m_amp_emb = nn.Parameter(torch.zeros(self.nentity, self.abs_dim))
        # nn.init.uniform_(tensor=self.abs_m_frq_emb, a=np.pi / 7, b=2 * np.pi / 3)
        # nn.init.uniform_(tensor=self.abs_m_phi_emb, a=-np.pi, b=np.pi)
        # nn.init.uniform_(tensor=self.abs_m_amp_emb, a=-self.emb_rng.item(), b=self.emb_rng.item())

        self.rel_d_frq_emb = nn.Parameter(torch.zeros(self.nentity, self.rel_dim))
        self.rel_d_phi_emb = nn.Parameter(torch.zeros(self.nentity, self.rel_dim))
        self.rel_d_amp_emb = nn.Parameter(torch.zeros(self.nentity, self.rel_dim))
        nn.init.uniform_(tensor=self.rel_d_frq_emb, a=np.pi / 7, b=2 * np.pi / 3)
        nn.init.uniform_(tensor=self.rel_d_phi_emb, a=-np.pi, b=np.pi)
        nn.init.uniform_(tensor=self.rel_d_amp_emb, a=-self.emb_rng.item(), b=self.emb_rng.item())

        if model_name == 'pRotatE':
            self.mod = nn.Parameter(torch.Tensor([[0.5 * self.emb_rng.item()]]))

    def abs_emb(self, e, d, m):
        d_amp = torch.index_select(self.abs_d_amp_emb, dim=0, index=e)
        d_frq = torch.index_select(self.abs_d_frq_emb, dim=0, index=e)
        d_phi = torch.index_select(self.abs_d_phi_emb, dim=0, index=e)

        # m_amp = torch.index_select(self.abs_m_amp_emb, dim=0, index=e)
        # m_frq = torch.index_select(self.abs_m_frq_emb, dim=0, index=e)
        # m_phi = torch.index_select(self.abs_m_phi_emb, dim=0, index=e)

        return d_amp * torch.sin(d * d_frq + d_phi)  # + m_amp * torch.sin(m * m_frq + m_phi)

    def rel_emb(self, e, e_rel):
        d_amp = torch.index_select(self.rel_d_amp_emb, dim=0, index=e)
        d_frq = torch.index_select(self.rel_d_frq_emb, dim=0, index=e)
        d_phi = torch.index_select(self.rel_d_phi_emb, dim=0, index=e)

        return d_amp * torch.sin(e_rel * d_frq + d_phi)

    def t_emb(self, e, d_abs, m_abs, e_rel):
        re_abs, im_abs = torch.chunk(self.abs_emb(e, d_abs, m_abs), 2, dim=1)
        re_rel, im_rel = torch.chunk(self.rel_emb(e, e_rel), 2, dim=1)

        re_t = torch.cat([re_abs, re_rel], dim=1)
        im_t = torch.cat([im_abs, im_rel], dim=1)

        return torch.cat([re_t, im_t], dim=1)

    def forward(self, x, mode='single'):
        if mode == 'single':
            d_abs = x[:, 3].view(-1, 1)
            m_abs = x[:, 4].view(-1, 1)

            s = torch.index_select(self.e_emb, dim=0, index=x[:, 0]).unsqueeze(1)
            s_t = self.t_emb(x[:, 0], d_abs, m_abs, x[:, 5].view(-1, 1)).unsqueeze(1)

            r = torch.index_select(self.r_emb, dim=0, index=x[:, 1]).unsqueeze(1)

            o = torch.index_select(self.e_emb, dim=0, index=x[:, 2]).unsqueeze(1)
            o_t = self.t_emb(x[:, 2], d_abs, m_abs, x[:, 6].view(-1, 1)).unsqueeze(1)

            t_neg = None
        elif mode == 's':
            pos, neg, neg_abs, neg_abs_s_rel, neg_abs_o_rel, neg_rel = x

            d_abs = pos[:, 3].view(-1, 1)
            m_abs = pos[:, 4].view(-1, 1)

            s = torch.index_select(self.e_emb, dim=0, index=neg.view(-1)).view(neg.size(0), neg.size(1), self.stt_dim)
            s_t = self.t_emb(
                neg.view(-1),
                d_abs.repeat(neg.size(1), 1).contiguous(),
                m_abs.repeat(neg.size(1), 1).contiguous(),
                neg_rel.view(-1, 1),
            ).view(neg.size(0), neg.size(1), self.abs_dim + self.rel_dim)

            r = torch.index_select(self.r_emb, dim=0, index=pos[:, 1]).unsqueeze(1)

            o = torch.index_select(self.e_emb, dim=0, index=pos[:, 2]).unsqueeze(1)
            o_t = self.t_emb(pos[:, 2], d_abs, m_abs, pos[:, 6].view(-1, 1)).unsqueeze(1)

            true_s = torch.index_select(self.e_emb, dim=0, index=pos[:, 0]).unsqueeze(1)

            d_abs_neg, m_abs_neg = torch.chunk(neg_abs, 2, dim=1)

            s_t_neg = self.t_emb(
                pos[:, 0].repeat(neg_abs.size(2), 1).t().contiguous().view(-1),
                d_abs_neg.contiguous().view(-1, 1),
                m_abs_neg.contiguous().view(-1, 1),
                neg_abs_s_rel.view(-1, 1),
            ).view(neg_abs.size(0), neg_abs.size(2), self.abs_dim + self.rel_dim)

            o_t_neg = self.t_emb(
                pos[:, 2].repeat(neg_abs.size(2), 1).t().contiguous().view(-1),
                d_abs_neg.contiguous().view(-1, 1),
                m_abs_neg.contiguous().view(-1, 1),
                neg_abs_o_rel.view(-1, 1),
            ).view(neg_abs.size(0), neg_abs.size(2), self.abs_dim + self.rel_dim)

            t_neg = (true_s, o, s_t_neg, o_t_neg)
        elif mode == 'o':
            pos, neg, neg_abs, neg_abs_s_rel, neg_abs_o_rel, neg_rel = x

            d_abs = pos[:, 3].view(-1, 1)
            m_abs = pos[:, 4].view(-1, 1)

            s = torch.index_select(self.e_emb, dim=0, index=pos[:, 0]).unsqueeze(1)
            s_t = self.t_emb(pos[:, 0], d_abs, m_abs, pos[:, 5].view(-1, 1)).unsqueeze(1)

            r = torch.index_select(self.r_emb, dim=0, index=pos[:, 1]).unsqueeze(1)

            o = torch.index_select(self.e_emb, dim=0, index=neg.view(-1)).view(neg.size(0), neg.size(1), self.stt_dim)
            o_t = self.t_emb(
                neg.view(-1),
                d_abs.repeat(neg.size(1), 1).contiguous(),
                m_abs.repeat(neg.size(1), 1).contiguous(),
                neg_rel.view(-1, 1),
            ).view(neg.size(0), neg.size(1), self.abs_dim + self.rel_dim)

            true_o = torch.index_select(self.e_emb, dim=0, index=pos[:, 2]).unsqueeze(1)

            d_abs_neg, m_abs_neg = torch.chunk(neg_abs, 2, dim=1)

            s_t_neg = self.t_emb(
                pos[:, 0].repeat(neg_abs.size(2), 1).t().contiguous().view(-1),
                d_abs_neg.contiguous().view(-1, 1),
                m_abs_neg.contiguous().view(-1, 1),
                neg_abs_s_rel.view(-1, 1),
            ).view(neg_abs.size(0), neg_abs.size(2), self.abs_dim + self.rel_dim)

            o_t_neg = self.t_emb(
                pos[:, 2].repeat(neg_abs.size(2), 1).t().contiguous().view(-1),
                d_abs_neg.contiguous().view(-1, 1),
                m_abs_neg.contiguous().view(-1, 1),
                neg_abs_o_rel.view(-1, 1),
            ).view(neg_abs.size(0), neg_abs.size(2), self.abs_dim + self.rel_dim)

            t_neg = (s, true_o, s_t_neg, o_t_neg)
        elif mode == 't':
            pos, neg, neg_abs, neg_abs_s_rel, neg_abs_o_rel, neg_rel = x

            d_abs = pos[:, 3].view(-1, 1)
            m_abs = pos[:, 4].view(-1, 1)

            s = torch.index_select(self.e_emb, dim=0, index=pos[:, 0]).unsqueeze(1)
            s_t = None

            r = torch.index_select(self.r_emb, dim=0, index=pos[:, 1]).unsqueeze(1)

            o = None
            o_t = None

            true_o = torch.index_select(self.e_emb, dim=0, index=pos[:, 2]).unsqueeze(1)

            d_abs_neg, m_abs_neg = torch.chunk(neg_abs, 2, dim=1)

            s_t_neg = self.t_emb(
                pos[:, 0].repeat(neg_abs.size(2), 1).t().contiguous().view(-1),
                d_abs_neg.contiguous().view(-1, 1),
                m_abs_neg.contiguous().view(-1, 1),
                neg_abs_s_rel.view(-1, 1),
            ).view(neg_abs.size(0), neg_abs.size(2), self.abs_dim + self.rel_dim)

            o_t_neg = self.t_emb(
                pos[:, 2].repeat(neg_abs.size(2), 1).t().contiguous().view(-1),
                d_abs_neg.contiguous().view(-1, 1),
                m_abs_neg.contiguous().view(-1, 1),
                neg_abs_o_rel.view(-1, 1),
            ).view(neg_abs.size(0), neg_abs.size(2), self.abs_dim + self.rel_dim)

            t_neg = (s, true_o, s_t_neg, o_t_neg)

        return getattr(self, self.model_name)(s, r, o, s_t, o_t, t_neg, mode)

    def TransE(self, head, relation, tail, head_time, tail_time, mode):
        head = torch.cat([head, head_time], dim=2)
        tail = torch.cat([tail, tail_time], dim=2)

        if mode == 's':
            score = head + (relation - tail)
        else:
            score = (head + relation) - tail

        score = self.gamma.item() - torch.norm(score, p=1, dim=2)
        return score

    def DistMult(self, head, relation, tail, head_time, tail_time, mode):
        head = torch.cat([head, head_time], dim=2)
        tail = torch.cat([tail, tail_time], dim=2)

        if mode == 's':
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

        if mode == 's':
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

        if mode != 't':
            re_head, im_head = torch.chunk(head, 2, dim=2)
            re_tail, im_tail = torch.chunk(tail, 2, dim=2)

        if mode != 't':
            re_head_time, im_head_time = torch.chunk(head_time, 2, dim=2)
            re_tail_time, im_tail_time = torch.chunk(tail_time, 2, dim=2)

        if mode != 'single':
            re_head_time_neg, im_head_time_neg = torch.chunk(time_neg[2], 2, dim=2)
            re_tail_time_neg, im_tail_time_neg = torch.chunk(time_neg[3], 2, dim=2)

        if mode != 'single':
            re_true_head, im_true_head = torch.chunk(time_neg[0], 2, dim=2)
            re_head_neg = torch.cat([re_true_head.repeat(1, re_head_time_neg.size(1), 1), re_head_time_neg], dim=2)
            im_head_neg = torch.cat([im_true_head.repeat(1, im_head_time_neg.size(1), 1), im_head_time_neg], dim=2)

        if mode != 't':
            re_head = torch.cat([re_head, re_head_time], dim=2)
            im_head = torch.cat([im_head, im_head_time], dim=2)

        if mode != 'single':
            re_true_tail, im_true_tail = torch.chunk(time_neg[1], 2, dim=2)
            re_tail_neg = torch.cat([re_true_tail.repeat(1, re_tail_time_neg.size(1), 1), re_tail_time_neg], dim=2)
            im_tail_neg = torch.cat([re_true_tail.repeat(1, im_tail_time_neg.size(1), 1), im_tail_time_neg], dim=2)

        if mode != 't':
            re_tail = torch.cat([re_tail, re_tail_time], dim=2)
            im_tail = torch.cat([im_tail, im_tail_time], dim=2)

        # Make phases of relations uniformly distributed in [-pi, pi]

        phase_relation = relation/(self.emb_rng.item()/pi)

        re_relation = torch.cos(phase_relation)
        im_relation = torch.sin(phase_relation)

        if mode != 'single':
            re_relation_neg = re_relation.repeat(1, re_head_neg.size(1), 1)
            im_relation_neg = im_relation.repeat(1, im_tail_neg.size(1), 1)

        if mode == 's':
            re_score = re_relation * re_tail + im_relation * im_tail
            im_score = re_relation * im_tail - im_relation * re_tail
            re_score = re_score - re_head
            im_score = im_score - im_head

            re_score_neg = re_relation_neg * re_tail_neg + im_relation_neg * im_tail_neg
            im_score_neg = re_relation_neg * im_tail_neg - im_relation_neg * re_tail_neg
            re_score_neg = re_score_neg - re_head_neg
            im_score_neg = im_score_neg - im_head_neg

            if re_score.size(1) > 0 and re_score_neg.size(1) > 0:
                re_score = torch.cat([re_score, re_score_neg], dim=1)
            elif re_score_neg.size(1) > 0:
                re_score = re_score_neg

            if im_score.size(1) > 0 and im_score_neg.size(1) > 0:
                im_score = torch.cat([im_score, im_score_neg], dim=1)
            elif im_score_neg.size(1) > 0:
                im_score = im_score_neg
        elif mode == 'o':
            re_score = re_head * re_relation - im_head * im_relation
            im_score = re_head * im_relation + im_head * re_relation
            re_score = re_score - re_tail
            im_score = im_score - im_tail

            re_score_neg = re_head_neg * re_relation_neg - im_head_neg * im_relation_neg
            im_score_neg = im_head_neg * re_relation_neg + re_head_neg * im_relation_neg
            re_score_neg = re_score_neg - re_tail_neg
            im_score_neg = im_score_neg - im_tail_neg

            if re_score.size(1) > 0 and re_score_neg.size(1) > 0:
                re_score = torch.cat([re_score, re_score_neg], dim=1)
            elif re_score_neg.size(1) > 0:
                re_score = re_score_neg

            if im_score.size(1) > 0 and im_score_neg.size(1) > 0:
                im_score = torch.cat([im_score, im_score_neg], dim=1)
            elif im_score_neg.size(1) > 0:
                im_score = im_score_neg
        elif mode == 't':
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

        phase_head = head/(self.emb_rng.item()/pi)
        phase_relation = relation/(self.emb_rng.item()/pi)
        phase_tail = tail/(self.emb_rng.item()/pi)

        if mode == 's':
            score = phase_head + (phase_relation - phase_tail)
        else:
            score = (phase_head + phase_relation) - phase_tail

        score = torch.sin(score)
        score = torch.abs(score)

        score = self.gamma.item() - score.sum(dim=2) * self.mod
        return score

    @staticmethod
    def train_step(mdl, opt, tr_it, args):
        mdl.train()

        opt.zero_grad()

        pos, neg, neg_abs, neg_rel, neg_abs_s_rel, neg_abs_o_rel, smpl_w, md = next(tr_it)
        smpl_w = smpl_w.squeeze(dim=1)
        if args.cuda:
            pos = pos.cuda()
            neg = neg.cuda()
            neg_abs = neg_abs.cuda()
            neg_rel = neg_rel.cuda()
            neg_abs_s_rel = neg_abs_s_rel.cuda()
            neg_abs_o_rel = neg_abs_o_rel.cuda()
            smpl_w = smpl_w.cuda()

        pos_sc = F.logsigmoid(mdl(pos)).squeeze(dim=1)
        neg_sc = mdl((pos, neg, neg_abs, neg_rel, neg_abs_s_rel, neg_abs_o_rel), md)
        if args.negative_adversarial_sampling:
            neg_sc = (F.softmax(neg_sc * args.alpha, dim=1).detach() * F.logsigmoid(-neg_sc)).sum(dim=1)
        else:
            neg_sc = F.logsigmoid(-neg_sc).mean(dim=1)

        if args.uni_weight:
            pos_lss = -pos_sc.mean()
            neg_lss = -neg_sc.mean()
        else:
            pos_lss = -(smpl_w * pos_sc).sum() / smpl_w.sum()
            neg_lss = -(smpl_w * neg_sc).sum() / smpl_w.sum()

        lss = (pos_lss + neg_lss) / 2

        reg_log = {}
        if args.lmbda != 0.0:
            # Use L3 regularization for ComplEx and DistMult
            reg = args.lmbda * (mdl.e_emb.norm(p=3) ** 3 + mdl.r_emb.norm(p=3).norm(p=3) ** 3)
            lss = lss + reg
            reg_log = {'regularization': reg.item()}

        lss.backward()
        opt.step()

        return {**reg_log,
                'pos_loss': pos_lss.item(),
                'neg_loss': neg_lss.item(),
                'loss': lss.item()}

    @staticmethod
    def test_step(model, test_quadruples, all_true_quadruples, event_index, args):
        model.eval()

        if args.eval_mode != 'time':
            test_dataloader_head = DataLoader(
                TestDataset(args, test_quadruples, all_true_quadruples, event_index, 's'),
                batch_size=args.test_batch_size,
                num_workers=max(1, args.cpu_num // 2)
            )

            test_dataloader_tail = DataLoader(
                TestDataset(args, test_quadruples, all_true_quadruples, event_index, 'o'),
                batch_size=args.test_batch_size,
                num_workers=max(1, args.cpu_num // 2)
            )

            test_dataset_list = []
            if args.eval_mode != 'tail':
                test_dataset_list.append((test_dataloader_head, 's'))
            if args.eval_mode != 'head':
                test_dataset_list.append((test_dataloader_tail, 'o'))
        else:
            test_dataloader_time = DataLoader(
                TestDataset(args, test_quadruples, all_true_quadruples, event_index, 't'),
                batch_size=args.test_batch_size,
                num_workers=max(1, args.cpu_num // 2)
            )
            test_dataset_list = [(test_dataloader_time, 't'), ]

        logs = []

        step = 0
        total_steps = sum([len(dataset) for dataset, _ in test_dataset_list])

        with torch.no_grad():
            for test_dataset, md in test_dataset_list:
                for pos, neg, neg_abs, neg_rel, neg_abs_s_rel, neg_abs_o_rel, fil_b in test_dataset:
                    if args.cuda:
                        pos = pos.cuda()
                        neg = neg.cuda()
                        neg_abs = neg_abs.cuda()
                        neg_rel = neg_rel.cuda()
                        neg_abs_s_rel = neg_abs_s_rel.cuda()
                        neg_abs_o_rel = neg_abs_o_rel.cuda()
                        fil_b = fil_b.cuda()

                    score = model((pos, neg, neg_abs, neg_rel, neg_abs_s_rel, neg_abs_o_rel), md) + fil_b
                    argsort = torch.argsort(score, dim=1, descending=True)

                    if md == 's':
                        positive_arg = pos[:, 0]
                        positive_issue_idx = pos[:, 2]
                    elif md == 'o':
                        positive_arg = pos[:, 2]
                        positive_issue_idx = pos[:, 0]
                    elif md == 't':
                        positive_arg = []
                        for (d, m) in pos[:, 3:5]:
                            min_dt = datetime.fromtimestamp(test_dataset.dataset.min_ts, test_dataset.dataset.tz)
                            dt = datetime(day=d, month=m, year=min_dt.year, tzinfo=test_dataset.dataset.tz)
                            positive_arg.append((dt + timedelta(days=1) - timedelta(seconds=1) - min_dt).days)

                    for i in range(pos.size(0)):
                        r = (argsort[i, :] == positive_arg[i]).nonzero().item() + 1
                        if md != 't' and model.type_index is not None:
                            index = model.type_index[model.type_reverse_index[positive_arg[i].item()]]
                            if model.users_idx is None:
                                r = np.isin(argsort[i, :].cpu().numpy(), index)[:r].sum()
                            else:
                                issue_index = model.users_idx.get(positive_issue_idx[i].item(), [])
                                issue_ranking = np.isin(argsort[i, :].cpu().numpy(), issue_index)[:r].sum()
                                if issue_ranking == 0:
                                    r = np.isin(argsort[i, :].cpu().numpy(), index)[:r].sum()
                                else:
                                    r = issue_ranking

                        logs.append({'MRR': 1.0 / r,
                                     'MR': float(r),
                                     'HITS@1': 1.0 if r <= 1 else 0.0,
                                     'HITS@3': 1.0 if r <= 3 else 0.0,
                                     'HITS@10': 1.0 if r <= 10 else 0.0, })

                    if step % args.test_log_steps == 0:
                        logging.info('Evaluating the model... (%d/%d)' % (step, total_steps))

                    step += 1

        metrics = {}
        for metric in logs[0].keys():
            metrics[metric] = sum([log[metric] for log in logs])/len(logs)

        return metrics
