import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class PositionalEmbedding(nn.Module):
    def __init__(self, r_dim):
        super(PositionalEmbedding, self).__init__()

        frq = 1 / (10000 ** (torch.arange(0.0, r_dim, 2.0) / r_dim))
        self.register_buffer('frq', frq)

    def forward(self, r):
        r_sin = torch.ger(r.float(), self.frq)
        return torch.cat([r_sin.cos(), r_sin.sin()], dim=1)


class KGEModel(nn.Module):
    def __init__(self, args):
        super(KGEModel, self).__init__()
        self.mdl_nm = args.model
        self.nr = args.nrelation
        self.drp = args.dropout
        self.gamma = args.gamma

        self.stt_dim = args.static_dim * 2
        self.abs_dim = args.absolute_dim * 2
        self.rel_dim = args.relative_dim * 2
        self.r_dim = args.static_dim + (self.abs_dim // 2)

        self.emb_rng_r = np.sqrt(6 / (args.nrelation + self.r_dim))

        self.e_emb = nn.Parameter(torch.zeros(args.nentity, self.stt_dim))
        self.r_emb = nn.Parameter(torch.zeros(args.nrelation, self.r_dim))
        nn.init.xavier_uniform_(self.e_emb)
        nn.init.xavier_uniform_(self.r_emb)

        self.abs_d_frq_emb = nn.Parameter(torch.zeros(args.nentity, self.abs_dim))
        self.abs_d_phi_emb = nn.Parameter(torch.zeros(args.nentity, self.abs_dim))
        self.abs_d_amp_emb = nn.Parameter(torch.zeros(args.nentity, self.abs_dim // 2))
        nn.init.xavier_uniform_(self.abs_d_frq_emb)
        nn.init.xavier_uniform_(self.abs_d_phi_emb)
        nn.init.xavier_uniform_(self.abs_d_amp_emb)

        self.abs_m_frq_emb = nn.Parameter(torch.zeros(args.nentity, self.abs_dim))
        self.abs_m_phi_emb = nn.Parameter(torch.zeros(args.nentity, self.abs_dim))
        self.abs_m_amp_emb = nn.Parameter(torch.zeros(args.nentity, self.abs_dim // 2))
        nn.init.xavier_uniform_(self.abs_m_frq_emb)
        nn.init.xavier_uniform_(self.abs_m_phi_emb)
        nn.init.xavier_uniform_(self.abs_m_amp_emb)

        self.p_emb = PositionalEmbedding(self.rel_dim)

        self.w_e = nn.Parameter(torch.zeros(self.stt_dim, self.rel_dim // 2))
        self.w_rp = nn.Parameter(torch.zeros(args.nrelation, args.nrelation, 1))
        nn.init.xavier_uniform_(self.w_e)
        nn.init.xavier_uniform_(self.w_rp)

    def e_p_emb(self, e_emb):
        re_w_e, im_w_e = torch.chunk(self.w_e, 2, dim=0)
        re_e_emb, im_e_emb = torch.chunk(e_emb, 2, dim=2)
        re_e_p_emb = (re_e_emb @ re_w_e) - (im_e_emb @ im_w_e)
        im_e_p_emb = (im_e_emb @ re_w_e) - (re_e_emb @ im_w_e)
        return torch.cat([re_e_p_emb, im_e_p_emb], dim=2)

    def e_r_emb(self, r, e_t):
        e_r = self.p_emb(e_t.view(-1)).view(e_t.size(0), e_t.size(1), self.rel_dim).permute(0, 2, 1)
        return e_r @ torch.index_select(self.w_rp, dim=0, index=r.long())

    def t_emb(self, e, d, m, y):
        d_amp = torch.index_select(self.abs_d_amp_emb, dim=0, index=e)
        d_frq = torch.index_select(self.abs_d_frq_emb, dim=0, index=e)
        d_phi = torch.index_select(self.abs_d_phi_emb, dim=0, index=e)

        m_amp = torch.index_select(self.abs_m_amp_emb, dim=0, index=e)
        m_frq = torch.index_select(self.abs_m_frq_emb, dim=0, index=e)
        m_phi = torch.index_select(self.abs_m_phi_emb, dim=0, index=e)

        re_d_emb, im_d_emb = torch.chunk(d * d_frq + d_phi, 2, dim=1)
        d_emb = torch.cat([d_amp * torch.cos(re_d_emb), d_amp * torch.sin(im_d_emb)], dim=1)

        re_m_emb, im_m_emb = torch.chunk(m * m_frq + m_phi, 2, dim=1)
        m_emb = torch.cat([m_amp * torch.cos(re_m_emb), m_amp * torch.sin(im_m_emb)], dim=1)

        return d_emb + m_emb

    def forward(self, x, md=None):
        if md is None:
            d_abs = x[:, 3].view(-1, 1)
            m_abs = x[:, 4].view(-1, 1)
            y_abs = x[:, 5].view(-1, 1)

            s = torch.index_select(self.e_emb, dim=0, index=x[:, 0]).unsqueeze(1)
            s_t = self.t_emb(x[:, 0], d_abs, m_abs, y_abs).unsqueeze(1)
            s_p = self.e_p_emb(s)
            s_r = self.e_r_emb(x[:, 1], x[:, -self.nr * 2:-self.nr].view(-1, self.nr).contiguous())

            r = torch.index_select(self.r_emb, dim=0, index=x[:, 1]).unsqueeze(1)

            o = torch.index_select(self.e_emb, dim=0, index=x[:, 2]).unsqueeze(1)
            o_t = self.t_emb(x[:, 2], d_abs, m_abs, y_abs).unsqueeze(1)
            o_p = self.e_p_emb(o)
            o_r = self.e_r_emb(x[:, 1], x[:, -self.nr:].view(-1, self.nr).contiguous())

            t_neg = None
        elif md == 's':
            pos, neg, neg_abs, neg_abs_s_rel, neg_abs_o_rel, neg_rel = x

            d_abs = pos[:, 3].view(-1, 1)
            m_abs = pos[:, 4].view(-1, 1)
            y_abs = pos[:, 5].view(-1, 1)

            s = torch.index_select(self.e_emb, dim=0, index=neg.view(-1)).view(neg.size(0), neg.size(1), self.stt_dim)
            s_t = self.t_emb(
                neg.view(-1),
                d_abs.repeat(1, neg.size(1)).view(-1, 1),
                m_abs.repeat(1, neg.size(1)).view(-1, 1),
                y_abs.repeat(1, neg.size(1)).view(-1, 1),
            ).view(neg.size(0), neg.size(1), self.abs_dim)
            s_p = self.e_p_emb(
                s.view(neg.size(0) * neg.size(1), self.stt_dim).unsqueeze(1)
            ).view(neg.size(0), neg.size(1), self.rel_dim)
            s_r = self.e_r_emb(
                pos[:, 1].repeat(neg_rel.size(1), 1).t().contiguous().view(-1),
                neg_rel.view(-1, self.nr)
            ).view(neg_rel.size(0), neg_rel.size(1), self.rel_dim)

            r = torch.index_select(self.r_emb, dim=0, index=pos[:, 1]).unsqueeze(1)

            o = torch.index_select(self.e_emb, dim=0, index=pos[:, 2]).unsqueeze(1)
            o_t = self.t_emb(pos[:, 2], d_abs, m_abs, y_abs).unsqueeze(1)
            o_p = self.e_p_emb(o)
            o_r = self.e_r_emb(pos[:, 1], pos[:, -self.nr:].contiguous().view(-1, self.nr))

            true_s = torch.index_select(self.e_emb, dim=0, index=pos[:, 0]).unsqueeze(1)

            d_abs_neg, m_abs_neg, y_abs_neg = torch.chunk(neg_abs, 3, dim=1)

            s_t_neg = self.t_emb(
                pos[:, 0].repeat(neg_abs.size(2), 1).t().contiguous().view(-1),
                d_abs_neg.contiguous().view(-1, 1),
                m_abs_neg.contiguous().view(-1, 1),
                y_abs_neg.contiguous().view(-1, 1),
            ).view(neg_abs.size(0), neg_abs.size(2), self.abs_dim)
            s_p_neg = self.e_p_emb(true_s)
            s_r_neg = self.e_r_emb(
                pos[:, 1].repeat(neg_abs_s_rel.size(1), 1).t().contiguous().view(-1),
                neg_abs_s_rel.view(-1, self.nr)
            ).view(neg_abs_s_rel.size(0), neg_abs_s_rel.size(1), self.rel_dim)

            o_t_neg = self.t_emb(
                pos[:, 2].repeat(neg_abs.size(2), 1).t().contiguous().view(-1),
                d_abs_neg.contiguous().view(-1, 1),
                m_abs_neg.contiguous().view(-1, 1),
                y_abs_neg.contiguous().view(-1, 1),
            ).view(neg_abs.size(0), neg_abs.size(2), self.abs_dim)
            o_r_neg = self.e_r_emb(
                pos[:, 1].repeat(neg_abs_o_rel.size(1), 1).t().contiguous().view(-1),
                neg_abs_o_rel.view(-1, self.nr)
            ).view(neg_abs_o_rel.size(0), neg_abs_o_rel.size(1), self.rel_dim)

            t_neg = (true_s, o, s_t_neg, o_t_neg, s_p_neg, o_p, s_r_neg, o_r_neg)
        elif md == 'o':
            pos, neg, neg_abs, neg_abs_s_rel, neg_abs_o_rel, neg_rel = x

            d_abs = pos[:, 3].view(-1, 1)
            m_abs = pos[:, 4].view(-1, 1)
            y_abs = pos[:, 5].view(-1, 1)

            s = torch.index_select(self.e_emb, dim=0, index=pos[:, 0]).unsqueeze(1)
            s_t = self.t_emb(pos[:, 0], d_abs, m_abs, y_abs).unsqueeze(1)
            s_p = self.e_p_emb(s)
            s_r = self.e_r_emb(pos[:, 1], pos[:, -self.nr * 2:-self.nr].contiguous().view(-1, self.nr))

            r = torch.index_select(self.r_emb, dim=0, index=pos[:, 1]).unsqueeze(1)

            o = torch.index_select(self.e_emb, dim=0, index=neg.view(-1)).view(neg.size(0), neg.size(1), self.stt_dim)
            o_t = self.t_emb(
                neg.view(-1),
                d_abs.repeat(1, neg.size(1)).view(-1, 1),
                m_abs.repeat(1, neg.size(1)).view(-1, 1),
                y_abs.repeat(1, neg.size(1)).view(-1, 1),
            ).view(neg.size(0), neg.size(1), self.abs_dim)
            o_p = self.e_p_emb(
                o.view(neg.size(0) * neg.size(1), self.stt_dim).unsqueeze(1)
            ).view(neg.size(0), neg.size(1), self.rel_dim)
            o_r = self.e_r_emb(
                pos[:, 1].repeat(neg_rel.size(1), 1).t().contiguous().view(-1),
                neg_rel.view(-1, self.nr)
            ).view(neg_rel.size(0), neg_rel.size(1), self.rel_dim)

            true_o = torch.index_select(self.e_emb, dim=0, index=pos[:, 2]).unsqueeze(1)

            d_abs_neg, m_abs_neg, y_abs_neg = torch.chunk(neg_abs, 3, dim=1)

            s_t_neg = self.t_emb(
                pos[:, 0].repeat(neg_abs.size(2), 1).t().contiguous().view(-1),
                d_abs_neg.contiguous().view(-1, 1),
                m_abs_neg.contiguous().view(-1, 1),
                y_abs_neg.contiguous().view(-1, 1),
            ).view(neg_abs.size(0), neg_abs.size(2), self.abs_dim)
            s_r_neg = self.e_r_emb(
                pos[:, 1].repeat(neg_abs_s_rel.size(1), 1).t().contiguous().view(-1),
                neg_abs_s_rel.view(-1, self.nr)
            ).view(neg_abs_s_rel.size(0), neg_abs_s_rel.size(1), self.rel_dim)

            o_t_neg = self.t_emb(
                pos[:, 2].repeat(neg_abs.size(2), 1).t().contiguous().view(-1),
                d_abs_neg.contiguous().view(-1, 1),
                m_abs_neg.contiguous().view(-1, 1),
                y_abs_neg.contiguous().view(-1, 1),
            ).view(neg_abs.size(0), neg_abs.size(2), self.abs_dim)
            o_p_neg = self.e_p_emb(true_o)
            o_r_neg = self.e_r_emb(
                pos[:, 1].repeat(neg_abs_o_rel.size(1), 1).t().contiguous().view(-1),
                neg_abs_o_rel.view(-1, self.nr)
            ).view(neg_abs_o_rel.size(0), neg_abs_o_rel.size(1), self.rel_dim)

            t_neg = (s, true_o, s_t_neg, o_t_neg, s_p, o_p_neg, s_r_neg, o_r_neg)
        elif md == 't':
            pos, neg, neg_abs, neg_abs_s_rel, neg_abs_o_rel, neg_rel = x

            d_abs = pos[:, 3].view(-1, 1)
            m_abs = pos[:, 4].view(-1, 1)
            y_abs = pos[:, 5].view(-1, 1)

            s = torch.index_select(self.e_emb, dim=0, index=pos[:, 0]).unsqueeze(1)
            s_t = None
            s_p = self.e_p_emb(s)
            s_r = None

            r = torch.index_select(self.r_emb, dim=0, index=pos[:, 1]).unsqueeze(1)

            o = None
            o_t = None
            o_p = None
            o_r = None

            true_o = torch.index_select(self.e_emb, dim=0, index=pos[:, 2]).unsqueeze(1)

            d_abs_neg, m_abs_neg, y_abs_neg = torch.chunk(neg_abs, 3, dim=1)

            s_t_neg = self.t_emb(
                pos[:, 0].repeat(neg_abs.size(2), 1).t().contiguous().view(-1),
                d_abs_neg.contiguous().view(-1, 1),
                m_abs_neg.contiguous().view(-1, 1),
                y_abs_neg.contiguous().view(-1, 1),
            ).view(neg_abs.size(0), neg_abs.size(2), self.abs_dim)
            s_r_neg = self.e_r_emb(
                pos[:, 1].repeat(neg_abs_s_rel.size(1), 1).t().contiguous().view(-1),
                neg_abs_s_rel.view(-1, self.nr)
            ).view(neg_abs_s_rel.size(0), neg_abs_s_rel.size(1), self.rel_dim)

            o_t_neg = self.t_emb(
                pos[:, 2].repeat(neg_abs.size(2), 1).t().contiguous().view(-1),
                d_abs_neg.contiguous().view(-1, 1),
                m_abs_neg.contiguous().view(-1, 1),
                y_abs_neg.contiguous().view(-1, 1),
            ).view(neg_abs.size(0), neg_abs.size(2), self.abs_dim)
            o_p_neg = self.e_p_emb(true_o)
            o_r_neg = self.e_r_emb(
                pos[:, 1].repeat(neg_abs_o_rel.size(1), 1).t().contiguous().view(-1),
                neg_abs_o_rel.view(-1, self.nr)
            ).view(neg_abs_o_rel.size(0), neg_abs_o_rel.size(1), self.rel_dim)

            t_neg = (s, true_o, s_t_neg, o_t_neg, s_p, o_p_neg, s_r_neg, o_r_neg)

        return getattr(self, self.mdl_nm)(s, r, o, s_t, o_t, s_p, o_p, s_r, o_r, t_neg, md)

    def RotatE(self, s, r, o, s_t, o_t, s_p, o_p, s_r, o_r, t_neg, md):
        pi = 3.14159265358979323846

        if md != 't':
            re_s, im_s = torch.chunk(s, 2, dim=2)
            re_s_t, im_s_t = torch.chunk(s_t, 2, dim=2)
            re_s, im_s = torch.cat([re_s, re_s_t], dim=2), torch.cat([im_s, im_s_t], dim=2)

            re_o, im_o = torch.chunk(o, 2, dim=2)
            re_o_t, im_o_t = torch.chunk(o_t, 2, dim=2)
            re_o, im_o = torch.cat([re_o, re_o_t], dim=2), torch.cat([im_o, im_o_t], dim=2)

        if md is not None:
            re_true_s, im_true_s = torch.chunk(t_neg[0], 2, dim=2)
            re_s_t_neg, im_s_t_neg = torch.chunk(t_neg[2], 2, dim=2)
            re_s_neg = torch.cat([re_true_s.repeat(1, re_s_t_neg.size(1), 1), re_s_t_neg], dim=2)
            im_s_neg = torch.cat([im_true_s.repeat(1, im_s_t_neg.size(1), 1), im_s_t_neg], dim=2)

            re_true_o, im_true_o = torch.chunk(t_neg[1], 2, dim=2)
            re_o_t_neg, im_o_t_neg = torch.chunk(t_neg[3], 2, dim=2)
            re_o_neg = torch.cat([re_true_o.repeat(1, re_o_t_neg.size(1), 1), re_o_t_neg], dim=2)
            im_o_neg = torch.cat([im_true_o.repeat(1, im_o_t_neg.size(1), 1), im_o_t_neg], dim=2)

        p_r = pi * (r / self.emb_rng_r)
        re_r, im_r = torch.cos(p_r), torch.sin(p_r)
        if md is not None:
            re_r_neg, im_r_neg = re_r.repeat(1, re_s_neg.size(1), 1), im_r.repeat(1, im_o_neg.size(1), 1)

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

            b = torch.cat([s_p - o_r.permute(0, 2, 1).repeat(1, s_p.size(1), 1),
                           t_neg[4].repeat(1, t_neg[7].size(1) or 1, 1) - t_neg[7]], dim=1)
            c = torch.cat([o_p.repeat(1, s_r.size(1), 1) - s_r,
                           t_neg[5].repeat(1, t_neg[6].size(1) or 1, 1) - t_neg[6]], dim=1)
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

            b = torch.cat([s_p.repeat(1, o_r.size(1), 1) - o_r,
                           t_neg[4].repeat(1, t_neg[7].size(1) or 1, 1) - t_neg[7]], dim=1)
            c = torch.cat([o_p - s_r.permute(0, 2, 1).repeat(1, o_p.size(1), 1),
                           t_neg[5].repeat(1, t_neg[6].size(1) or 1, 1) - t_neg[6]], dim=1)
        elif md == 't':
            re_sc_neg = re_s_neg * re_r_neg - im_s_neg * im_r_neg
            im_sc_neg = im_s_neg * re_r_neg + re_s_neg * im_r_neg
            re_sc_neg = re_sc_neg - re_o_neg
            im_sc_neg = im_sc_neg - im_o_neg

            re_sc = re_sc_neg
            im_sc = im_sc_neg

            b = t_neg[4].repeat(1, t_neg[7].size(1), 1) - t_neg[7]
            c = t_neg[5].repeat(1, t_neg[6].size(1), 1) - t_neg[6]
        else:
            re_sc = re_s * re_r - im_s * im_r
            im_sc = re_s * im_r + im_s * re_r
            re_sc = re_sc - re_o
            im_sc = im_sc - im_o

            b = s_p - o_r.permute(0, 2, 1)
            c = o_p - s_r.permute(0, 2, 1)

        re_b, im_b = torch.chunk(b, 2, dim=2)
        re_c, im_c = torch.chunk(c, 2, dim=2)

        sc = torch.stack([torch.cat([re_sc, re_b, re_c], dim=2), torch.cat([im_sc, im_b, im_c], dim=2)], dim=0)
        sc = F.dropout(sc, p=self.drp, training=self.training)
        sc = sc.norm(dim=0)
        sc = self.gamma - sc.sum(dim=2)

        return sc
