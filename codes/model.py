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
    def __init__(self, tp_ix, tp_rix, u_ix, args):
        super(KGEModel, self).__init__()
        self.mdl_nm = args.model

        self.tp_ix = tp_ix
        self.tp_rix = tp_rix
        self.u_ix = u_ix

        self.epsilon = args.epsilon
        self.gamma = nn.Parameter(torch.Tensor([args.gamma]), requires_grad=False)

        self.stt_dim = args.static_dim * 2 if self.mdl_nm in ['RotatE', 'ComplEx'] else args.static_dim
        self.abs_dim = args.absolute_dim * 2 if self.mdl_nm in ['RotatE', 'ComplEx'] else args.absolute_dim
        self.rel_dim = args.relative_dim * 2 if self.mdl_nm in ['RotatE', 'ComplEx'] else args.relative_dim

        self.r_dim = args.static_dim * 2 if self.mdl_nm == 'ComplEx' else args.static_dim
        if self.mdl_nm == 'RotatE':
            self.r_dim += (self.abs_dim // 2) + (self.rel_dim // 2)
        else:
            self.r_dim += self.abs_dim + self.rel_dim

        self.emb_rng = nn.Parameter(
            torch.Tensor([(self.gamma.item() + self.epsilon) / self.r_dim]), requires_grad=False)

        self.e_emb = nn.Parameter(torch.zeros(args.nentity, self.stt_dim))
        self.r_emb = nn.Parameter(torch.zeros(args.nrelation, self.r_dim))
        nn.init.uniform_(tensor=self.e_emb, a=-self.emb_rng.item(), b=self.emb_rng.item())
        nn.init.uniform_(tensor=self.r_emb, a=-self.emb_rng.item(), b=self.emb_rng.item())

        self.abs_d_frq_emb = nn.Parameter(torch.zeros(args.nentity, self.abs_dim))
        self.abs_d_phi_emb = nn.Parameter(torch.zeros(args.nentity, self.abs_dim))
        self.abs_d_amp_emb = nn.Parameter(torch.zeros(args.nentity, self.abs_dim))
        nn.init.uniform_(tensor=self.abs_d_frq_emb, a=np.pi / 7, b=2 * np.pi / 3)
        nn.init.uniform_(tensor=self.abs_d_phi_emb, a=-np.pi, b=np.pi)
        nn.init.uniform_(tensor=self.abs_d_amp_emb, a=-self.emb_rng.item(), b=self.emb_rng.item())

        # self.abs_m_frq_emb = nn.Parameter(torch.zeros(args.nentity, self.abs_dim))
        # self.abs_m_phi_emb = nn.Parameter(torch.zeros(args.nentity, self.abs_dim))
        # self.abs_m_amp_emb = nn.Parameter(torch.zeros(args.nentity, self.abs_dim))
        # nn.init.uniform_(tensor=self.abs_m_frq_emb, a=np.pi / 7, b=2 * np.pi / 3)
        # nn.init.uniform_(tensor=self.abs_m_phi_emb, a=-np.pi, b=np.pi)
        # nn.init.uniform_(tensor=self.abs_m_amp_emb, a=-self.emb_rng.item(), b=self.emb_rng.item())

        self.rel_d_frq_emb = nn.Parameter(torch.zeros(args.nentity, self.rel_dim))
        self.rel_d_phi_emb = nn.Parameter(torch.zeros(args.nentity, self.rel_dim))
        self.rel_d_amp_emb = nn.Parameter(torch.zeros(args.nentity, self.rel_dim))
        nn.init.uniform_(tensor=self.rel_d_frq_emb, a=np.pi / 7, b=2 * np.pi / 3)
        nn.init.uniform_(tensor=self.rel_d_phi_emb, a=-np.pi, b=np.pi)
        nn.init.uniform_(tensor=self.rel_d_amp_emb, a=-self.emb_rng.item(), b=self.emb_rng.item())

        if self.mdl_nm == 'pRotatE':
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

    def forward(self, x, md=None):
        if md is None:
            d_abs = x[:, 3].view(-1, 1)
            m_abs = x[:, 4].view(-1, 1)

            s = torch.index_select(self.e_emb, dim=0, index=x[:, 0]).unsqueeze(1)
            s_t = self.t_emb(x[:, 0], d_abs, m_abs, x[:, 5].view(-1, 1)).unsqueeze(1)

            r = torch.index_select(self.r_emb, dim=0, index=x[:, 1]).unsqueeze(1)

            o = torch.index_select(self.e_emb, dim=0, index=x[:, 2]).unsqueeze(1)
            o_t = self.t_emb(x[:, 2], d_abs, m_abs, x[:, 6].view(-1, 1)).unsqueeze(1)

            t_neg = None
        elif md == 's':
            pos, neg, neg_abs, neg_abs_s_rel, neg_abs_o_rel, neg_rel = x

            d_abs = pos[:, 3].view(-1, 1)
            m_abs = pos[:, 4].view(-1, 1)

            s = torch.index_select(self.e_emb, dim=0, index=neg.view(-1)).view(neg.size(0), neg.size(1), self.stt_dim)
            s_t = self.t_emb(
                neg.view(-1),
                d_abs.repeat(neg.size(1), 1).contiguous(),
                m_abs.repeat(neg.size(1), 1).contiguous(),
                neg_rel.view(-1, 1)
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
                neg_abs_s_rel.view(-1, 1)
            ).view(neg_abs.size(0), neg_abs.size(2), self.abs_dim + self.rel_dim)

            o_t_neg = self.t_emb(
                pos[:, 2].repeat(neg_abs.size(2), 1).t().contiguous().view(-1),
                d_abs_neg.contiguous().view(-1, 1),
                m_abs_neg.contiguous().view(-1, 1),
                neg_abs_o_rel.view(-1, 1)
            ).view(neg_abs.size(0), neg_abs.size(2), self.abs_dim + self.rel_dim)

            t_neg = (true_s, o, s_t_neg, o_t_neg)
        elif md == 'o':
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
                neg_rel.view(-1, 1)
            ).view(neg.size(0), neg.size(1), self.abs_dim + self.rel_dim)

            true_o = torch.index_select(self.e_emb, dim=0, index=pos[:, 2]).unsqueeze(1)

            d_abs_neg, m_abs_neg = torch.chunk(neg_abs, 2, dim=1)

            s_t_neg = self.t_emb(
                pos[:, 0].repeat(neg_abs.size(2), 1).t().contiguous().view(-1),
                d_abs_neg.contiguous().view(-1, 1),
                m_abs_neg.contiguous().view(-1, 1),
                neg_abs_s_rel.view(-1, 1)
            ).view(neg_abs.size(0), neg_abs.size(2), self.abs_dim + self.rel_dim)

            o_t_neg = self.t_emb(
                pos[:, 2].repeat(neg_abs.size(2), 1).t().contiguous().view(-1),
                d_abs_neg.contiguous().view(-1, 1),
                m_abs_neg.contiguous().view(-1, 1),
                neg_abs_o_rel.view(-1, 1)
            ).view(neg_abs.size(0), neg_abs.size(2), self.abs_dim + self.rel_dim)

            t_neg = (s, true_o, s_t_neg, o_t_neg)
        elif md == 't':
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
                neg_abs_s_rel.view(-1, 1)
            ).view(neg_abs.size(0), neg_abs.size(2), self.abs_dim + self.rel_dim)

            o_t_neg = self.t_emb(
                pos[:, 2].repeat(neg_abs.size(2), 1).t().contiguous().view(-1),
                d_abs_neg.contiguous().view(-1, 1),
                m_abs_neg.contiguous().view(-1, 1),
                neg_abs_o_rel.view(-1, 1)
            ).view(neg_abs.size(0), neg_abs.size(2), self.abs_dim + self.rel_dim)

            t_neg = (s, true_o, s_t_neg, o_t_neg)

        return getattr(self, self.mdl_nm)(s, r, o, s_t, o_t, t_neg, md)

    def TransE(self, s, r, o, s_t, o_t, t_neg, md):
        if md != 't':
            s = torch.cat([s, s_t], dim=2)
            o = torch.cat([o, o_t], dim=2)

        if md is not None:
            s_neg = torch.cat([t_neg[0].repeat(1, t_neg[2].size(1), 1), t_neg[2]], dim=2)
            o_neg = torch.cat([t_neg[1].repeat(1, t_neg[3].size(1), 1), t_neg[3]], dim=2)
            r_neg = r.repeat(1, s_neg.size(1), 1)

        if md == 's':
            sc_neg = s_neg + (r_neg - o_neg)
            sc = s + (r - o)

            if sc.size(1) > 0 and sc_neg.size(1) > 0:
                sc = torch.cat([sc, sc_neg], dim=1)
            elif sc_neg.size(1) > 0:
                sc = sc_neg
        if md == 'o':
            sc_neg = (s_neg + r_neg) - o_neg
            sc = (s + r) - o

            if sc.size(1) > 0 and sc_neg.size(1) > 0:
                sc = torch.cat([sc, sc_neg], dim=1)
            elif sc_neg.size(1) > 0:
                sc = sc_neg
        if md == 't':
            sc_neg = (s_neg + r_neg) - o_neg

            sc = sc_neg
        else:
            sc = (s + r) - o

        sc = self.gamma.item() - torch.norm(sc, p=1, dim=2)
        return sc

    def DistMult(self, s, r, o, s_t, o_t, t_neg, md):
        if md != 't':
            s = torch.cat([s, s_t], dim=2)
            o = torch.cat([o, o_t], dim=2)

        if md is not None:
            s_neg = torch.cat([t_neg[0].repeat(1, t_neg[2].size(1), 1), t_neg[2]], dim=2)
            o_neg = torch.cat([t_neg[1].repeat(1, t_neg[3].size(1), 1), t_neg[3]], dim=2)
            r_neg = r.repeat(1, s_neg.size(1), 1)

        if md == 's':
            sc_neg = s_neg * (r_neg * o_neg)
            sc = s * (r * o)

            if sc.size(1) > 0 and sc_neg.size(1) > 0:
                sc = torch.cat([sc, sc_neg], dim=1)
            elif sc_neg.size(1) > 0:
                sc = sc_neg
        if md == 'o':
            sc_neg = (s_neg * r_neg) * o_neg
            sc = (s * r) * o

            if sc.size(1) > 0 and sc_neg.size(1) > 0:
                sc = torch.cat([sc, sc_neg], dim=1)
            elif sc_neg.size(1) > 0:
                sc = sc_neg
        if md == 't':
            sc_neg = s_neg * (r_neg * o_neg)

            sc = sc_neg
        else:
            sc = (s * r) * o

        sc = sc.sum(dim=2)
        return sc

    def ComplEx(self, s, r, o, s_t, o_t, t_neg, md):
        if md != 't':
            re_s, im_s = torch.chunk(s, 2, dim=2)
            re_o, im_o = torch.chunk(o, 2, dim=2)

        if md != 't':
            re_s_t, im_s_t = torch.chunk(s_t, 2, dim=2)
            re_o_t, im_o_t = torch.chunk(o_t, 2, dim=2)

        if md is not None:
            re_s_t_neg, im_s_t_neg = torch.chunk(t_neg[2], 2, dim=2)
            re_o_t_neg, im_o_t_neg = torch.chunk(t_neg[3], 2, dim=2)

        if md is not None:
            re_true_s, im_true_s = torch.chunk(t_neg[0], 2, dim=2)
            re_s_neg = torch.cat([re_true_s.repeat(1, re_s_t_neg.size(1), 1), re_s_t_neg], dim=2)
            im_s_neg = torch.cat([im_true_s.repeat(1, im_s_t_neg.size(1), 1), im_s_t_neg], dim=2)

        if md != 't':
            re_s = torch.cat([re_s, re_s_t], dim=2)
            im_s = torch.cat([im_s, im_s_t], dim=2)

        if md is not None:
            re_true_o, im_true_o = torch.chunk(t_neg[1], 2, dim=2)
            re_o_neg = torch.cat([re_true_o.repeat(1, re_o_t_neg.size(1), 1), re_o_t_neg], dim=2)
            im_o_neg = torch.cat([re_true_o.repeat(1, im_o_t_neg.size(1), 1), im_o_t_neg], dim=2)

        if md != 't':
            re_o = torch.cat([re_o, re_o_t], dim=2)
            im_o = torch.cat([im_o, im_o_t], dim=2)

        re_r, im_r = torch.chunk(r, 2, dim=2)

        if md is not None:
            re_r_neg = re_r.repeat(1, re_s_neg.size(1), 1)
            im_r_neg = im_r.repeat(1, im_o_neg.size(1), 1)

        if md == 's':
            re_sc = re_r * re_o + im_r * im_o
            im_sc = re_r * im_o - im_r * re_o
            sc = re_s * re_sc + im_s * im_sc

            re_sc_neg = re_r_neg * re_o_neg + im_r_neg * im_o_neg
            im_sc_neg = re_r_neg * im_o_neg - im_r_neg * re_o_neg
            sc_neg = re_s_neg * re_sc_neg + im_s_neg * im_sc_neg

            if sc.size(1) > 0 and sc_neg.size(1) > 0:
                sc = torch.cat([sc, sc_neg], dim=1)
            elif sc_neg.size(1) > 0:
                sc = sc_neg
        if md == 'o':
            re_sc = re_s * re_r - im_s * im_r
            im_sc = re_s * im_r + im_s * re_r
            sc = re_sc * re_o + im_sc * im_o

            re_sc_neg = re_s_neg * re_r_neg - im_s_neg * im_r_neg
            im_sc_neg = re_s_neg * im_r_neg + im_s_neg * re_r_neg
            sc_neg = re_sc_neg * re_o_neg + im_sc_neg * im_o_neg

            if sc.size(1) > 0 and sc_neg.size(1) > 0:
                sc = torch.cat([sc, sc_neg], dim=1)
            elif sc_neg.size(1) > 0:
                sc = sc_neg
        if md == 't':
            re_sc_neg = re_s_neg * re_r_neg - im_s_neg * im_r_neg
            im_sc_neg = re_s_neg * im_r_neg + im_s_neg * re_r_neg
            sc_neg = re_sc_neg * re_o_neg + im_sc_neg * im_o_neg

            sc = sc_neg
        else:
            re_sc = re_s * re_r - im_s * im_r
            im_sc = re_s * im_r + im_s * re_r
            sc = re_sc * re_o + im_sc * im_o

        sc = sc.sum(dim=2)
        return sc

    def RotatE(self, s, r, o, s_t, o_t, t_neg, md):
        pi = 3.14159265358979323846

        if md != 't':
            re_s, im_s = torch.chunk(s, 2, dim=2)
            re_o, im_o = torch.chunk(o, 2, dim=2)

        if md != 't':
            re_s_t, im_s_t = torch.chunk(s_t, 2, dim=2)
            re_o_t, im_o_t = torch.chunk(o_t, 2, dim=2)

        if md is not None:
            re_s_t_neg, im_s_t_neg = torch.chunk(t_neg[2], 2, dim=2)
            re_o_t_neg, im_o_t_neg = torch.chunk(t_neg[3], 2, dim=2)

        if md is not None:
            re_true_s, im_true_s = torch.chunk(t_neg[0], 2, dim=2)
            re_s_neg = torch.cat([re_true_s.repeat(1, re_s_t_neg.size(1), 1), re_s_t_neg], dim=2)
            im_s_neg = torch.cat([im_true_s.repeat(1, im_s_t_neg.size(1), 1), im_s_t_neg], dim=2)

        if md != 't':
            re_s = torch.cat([re_s, re_s_t], dim=2)
            im_s = torch.cat([im_s, im_s_t], dim=2)

        if md is not None:
            re_true_o, im_true_o = torch.chunk(t_neg[1], 2, dim=2)
            re_o_neg = torch.cat([re_true_o.repeat(1, re_o_t_neg.size(1), 1), re_o_t_neg], dim=2)
            im_o_neg = torch.cat([re_true_o.repeat(1, im_o_t_neg.size(1), 1), im_o_t_neg], dim=2)

        if md != 't':
            re_o = torch.cat([re_o, re_o_t], dim=2)
            im_o = torch.cat([im_o, im_o_t], dim=2)

        p_r = r / (self.emb_rng.item() / pi)

        re_r = torch.cos(p_r)
        im_r = torch.sin(p_r)

        if md is not None:
            re_r_neg = re_r.repeat(1, re_s_neg.size(1), 1)
            im_r_neg = im_r.repeat(1, im_o_neg.size(1), 1)

        if md == 's':
            re_sc = re_r * re_o + im_r * im_o
            im_sc = re_r * im_o - im_r * re_o
            re_sc = re_sc - re_s
            im_sc = im_sc - im_s

            re_sc_neg = re_r_neg * re_o_neg + im_r_neg * im_o_neg
            im_sc_neg = re_r_neg * im_o_neg - im_r_neg * re_o_neg
            re_sc_neg = re_sc_neg - re_s_neg
            im_sc_neg = im_sc_neg - im_s_neg

            if re_sc.size(1) > 0 and re_sc_neg.size(1) > 0:
                re_sc = torch.cat([re_sc, re_sc_neg], dim=1)
            elif re_sc_neg.size(1) > 0:
                re_sc = re_sc_neg

            if im_sc.size(1) > 0 and im_sc_neg.size(1) > 0:
                im_sc = torch.cat([im_sc, im_sc_neg], dim=1)
            elif im_sc_neg.size(1) > 0:
                im_sc = im_sc_neg
        elif md == 'o':
            re_sc = re_s * re_r - im_s * im_r
            im_sc = re_s * im_r + im_s * re_r
            re_sc = re_sc - re_o
            im_sc = im_sc - im_o

            re_sc_neg = re_s_neg * re_r_neg - im_s_neg * im_r_neg
            im_sc_neg = im_s_neg * re_r_neg + re_s_neg * im_r_neg
            re_sc_neg = re_sc_neg - re_o_neg
            im_sc_neg = im_sc_neg - im_o_neg

            if re_sc.size(1) > 0 and re_sc_neg.size(1) > 0:
                re_sc = torch.cat([re_sc, re_sc_neg], dim=1)
            elif re_sc_neg.size(1) > 0:
                re_sc = re_sc_neg

            if im_sc.size(1) > 0 and im_sc_neg.size(1) > 0:
                im_sc = torch.cat([im_sc, im_sc_neg], dim=1)
            elif im_sc_neg.size(1) > 0:
                im_sc = im_sc_neg
        elif md == 't':
            re_sc_neg = re_s_neg * re_r_neg - im_s_neg * im_r_neg
            im_sc_neg = im_s_neg * re_r_neg + re_s_neg * im_r_neg
            re_sc_neg = re_sc_neg - re_o_neg
            im_sc_neg = im_sc_neg - im_o_neg

            re_sc = re_sc_neg
            im_sc = im_sc_neg
        else:
            re_sc = re_s * re_r - im_s * im_r
            im_sc = re_s * im_r + im_s * re_r
            re_sc = re_sc - re_o
            im_sc = im_sc - im_o

        sc = torch.stack([re_sc, im_sc], dim=0)
        sc = sc.norm(dim=0)

        sc = self.gamma.item() - sc.sum(dim=2)
        return sc

    def pRotatE(self, s, r, o, s_t, o_t, md):
        pi = 3.14159262358979323846

        if md != 't':
            s = torch.cat([s / (self.emb_rng.item() / pi), s_t], dim=2)
            o = torch.cat([o / (self.emb_rng.item() / pi), o_t], dim=2)

        if md is not None:
            s_neg = torch.cat([t_neg[0].repeat(1, t_neg[2].size(1), 1), t_neg[2]], dim=2)
            o_neg = torch.cat([t_neg[1].repeat(1, t_neg[3].size(1), 1), t_neg[3]], dim=2)

        r = r / (self.emb_rng.item() / pi)

        if md is not None:
            r_neg = r.repeat(1, s_neg.size(1), 1)

        if md == 's':
            sc_neg = s_neg + (r_neg - o_neg)
            sc = s + (r - o)

            if sc.size(1) > 0 and sc_neg.size(1) > 0:
                sc = torch.cat([sc, sc_neg], dim=1)
            elif sc_neg.size(1) > 0:
                sc = sc_neg
        if md == 'o':
            sc_neg = (s_neg + r_neg) - o_neg
            sc = (s + r) - o

            if sc.size(1) > 0 and sc_neg.size(1) > 0:
                sc = torch.cat([sc, sc_neg], dim=1)
            elif sc_neg.size(1) > 0:
                sc = sc_neg
        if md == 't':
            sc_neg = (s_neg + r_neg) - o_neg

            sc = sc_neg
        else:
            sc = (s + r) - o

        sc = torch.sin(sc)
        sc = torch.abs(sc)

        sc = self.gamma.item() - sc.sum(dim=2) * self.mod
        return sc

    @staticmethod
    def train_step(mdl, opt, tr_it, args):
        mdl.train()

        opt.zero_grad()

        pos, neg, neg_abs, neg_rel, neg_abs_s_rel, neg_abs_o_rel, smpl_w, md = next(tr_it)
        smpl_w = smpl_w.squeeze(dim=1)
        if torch.cuda.is_available():
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
            reg = args.lmbda * (mdl.module.e_emb.norm(p=3) ** 3 + mdl.module.r_emb.norm(p=3).norm(p=3) ** 3)
            lss = lss + reg
            reg_log = {'regularization': reg.item()}

        lss.backward()
        opt.step()

        return {**reg_log,
                'pos_loss': pos_lss.item(),
                'neg_loss': neg_lss.item(),
                'loss': lss.item()}

    @staticmethod
    def test_step(mdl, ts_q, al_q, ev_ix, args):
        mdl.eval()

        ts_dls = []
        if args.mode in ['head', 'both', 'full']:
            ts_dls.append((DataLoader(
                TestDataset(ts_q, al_q, ev_ix, 's', args),
                batch_size=args.test_batch_size,
                num_workers=max(1, os.cpu_count() // 2),
            ), 's'))
        if args.mode in ['tail', 'both', 'full']:
            ts_dls.append((DataLoader(
                TestDataset(ts_q, al_q, ev_ix, 'o', args),
                batch_size=args.test_batch_size,
                num_workers=max(1, os.cpu_count() // 2),
            ), 'o'))
        if args.mode in ['time', 'full']:
            ts_dls.append((DataLoader(
                TestDataset(ts_q, al_q, ev_ix, 't', args),
                batch_size=args.test_batch_size,
                num_workers=max(1, os.cpu_count() // 2),
            ), 't'))

        logs = []
        stp = 0
        tot_stp = sum([len(ts_dl) for ts_dl, _ in ts_dls])
        with torch.no_grad():
            for ts_dl, md in ts_dls:
                for pos, neg, neg_abs, neg_abs_s_rel, neg_abs_o_rel, neg_rel, fil_b in ts_dl:
                    if torch.cuda.is_available():
                        pos = pos.cuda()
                        neg = neg.cuda()
                        neg_abs = neg_abs.cuda()
                        neg_abs_s_rel = neg_abs_s_rel.cuda()
                        neg_abs_o_rel = neg_abs_o_rel.cuda()
                        neg_rel = neg_rel.cuda()
                        fil_b = fil_b.cuda()

                    sc = mdl((pos, neg, neg_abs, neg_abs_s_rel, neg_abs_o_rel, neg_rel), md) + fil_b
                    as_sc = torch.argsort(sc, dim=1, descending=True)

                    if md == 's':
                        true_pos, pos_u_ix = pos[:, 0], pos[:, 2]
                    elif md == 'o':
                        true_pos, pos_u_ix = pos[:, 2], pos[:, 0]
                    elif md == 't':
                        true_pos = []
                        for (d, m) in pos[:, 3:5]:
                            min_dt = datetime.fromtimestamp(ts_dl.dataset.min_ts, ts_dl.dataset.tz)
                            dt = datetime(day=d, month=m, year=min_dt.year, tzinfo=ts_dl.dataset.tz)
                            true_pos.append((dt + timedelta(days=1) - timedelta(seconds=1) - min_dt).days)

                    for i in range(pos.size(0)):
                        r = (as_sc[i, :] == true_pos[i]).nonzero().item() + 1
                        if md != 't' and args.negative_type_sampling:
                            ix = mdl.module.tp_ix[mdl.module.tp_rix[true_pos[i].item()]]
                            if args.heuristic_evaluation:
                                u_ix = mdl.module.u_ix.get(pos_u_ix[i].item(), [])
                                u_r = np.isin(as_sc[i, :].cpu().numpy(), u_ix)[:r].sum()
                                r = np.isin(as_sc[i, :].cpu().numpy(), ix)[:r].sum() if u_r == 0 else u_r
                            else:
                                r = np.isin(as_sc[i, :].cpu().numpy(), ix)[:r].sum()

                        logs.append({'MRR': 1.0 / r,
                                     'MR': float(r),
                                     'H1': 1.0 if r <= 1 else 0.0,
                                     'H3': 1.0 if r <= 3 else 0.0,
                                     'H10': 1.0 if r <= 10 else 0.0, })

                    if stp % args.test_log_steps == 0:
                        logging.info(f'Evaluating the model ... ({stp}/{tot_stp})')

                    stp += 1

        mtrs = {}
        for mtr in logs[0].keys():
            mtrs[mtr] = sum([log[mtr] for log in logs]) / len(logs)

        return mtrs
