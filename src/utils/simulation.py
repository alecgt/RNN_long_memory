'''
Tools for simulation of long memory data.
'''
import torch
import numpy as np


# VARMA sim
def sim_VARMA(T, k=1, VAR=None, VMA=None, cov=None, innov=None):
    if cov is None:
        cov = torch.eye(k)
        chol = torch.eye(k)
    else:
        chol = torch.from_numpy(np.linalg.cholesky(cov))

    if innov is None:
        innov = torch.matmul(chol, torch.randn((k, T)))

    if VAR is not None:
        p = VAR.shape[2]
    else:
        p = 0

    if VMA is not None:
        q = VMA.shape[2]
    else:
        q = 0

    u0 = innov
    if q > 0:  # moving average
        for i in range(q, T):
            for qq in range(1, q + 1):
                u0[:, i] += torch.matmul(VMA[:, :, qq - 1], u0[:, i - qq])

    if p > 0:  # vector autoregression
        for i in range(p, T):
            for pp in range(1, p + 1):
                u0[:, i] -= torch.matmul(VAR[:, :, pp - 1], u0[:, i - pp])

    return u0


# fractional differencing
def fracdiff(seq, d):  # takes k x T sequence, len-k d and returns row-wise fractionally differenced seq of same dim
    k, T = seq.shape
    seq_ext = torch.cat((seq, torch.zeros(k, T - 1)), dim=1)
    seq_ext = torch.cat((seq_ext.unsqueeze(2), torch.zeros(seq_ext.unsqueeze(2).shape)), dim=2)  # add complex dim
    seq_ext_fft = torch.fft(seq_ext, signal_ndim=1)

    filt = torch.zeros((k, 2 * T - 1))
    for i in range(k):
        filt[i, :T] = fd_filter(d[i], T)
    filt = torch.cat((filt.unsqueeze(2), torch.zeros(filt.unsqueeze(2).shape)), dim=2)
    filt_fft = torch.fft(filt, signal_ndim=1)

    prod = torch.zeros(filt_fft.shape)  # still taking complex products by hand
    prod[:, :, 0] = filt_fft[:, :, 0] * seq_ext_fft[:, :, 0] - filt_fft[:, :, 1] * seq_ext_fft[:, :, 1]
    prod[:, :, 1] = filt_fft[:, :, 0] * seq_ext_fft[:, :, 1] + filt_fft[:, :, 1] * seq_ext_fft[:, :, 0]

    return torch.ifft(prod, signal_ndim=1)[:, :T, 0]


def fd_filter(d, T):
    filt = torch.cumprod((torch.arange(1, T, dtype=torch.float32) + d - 1) / torch.arange(1, T, dtype=torch.float32), 0)
    return torch.cat((torch.ones(1), filt))


# VARFIMA
def sim_VARFIMA(T, k, d, VAR, VMA, cov=None):
    skip = 2000

    if cov is None:
        cov = torch.eye(k)
        innov = torch.randn((k, T + 2 * skip))
    else:
        chol = torch.from_numpy(np.linalg.cholesky(cov))
        innov = torch.matmul(chol, torch.randn((k, T + 2 * skip)))

    if VAR is not None:
        p = VAR.shape[2]
    else:
        p = 0

    if VMA is not None:
        q = VMA.shape[2]
    else:
        q = 0

    seq = sim_VARMA(T + 2 * skip, k, VAR, VMA, cov, innov)
    seq = fracdiff(seq, d)

    CVMA = cov
    if q > 0:
        sum_MA = torch.sum(VMA, dim=2)
        CVMA = torch.matmul((torch.eye(k) + sum_MA), torch.matmul(CVMA, (torch.eye(k) + sum_MA).t()))

    CVAR = torch.eye(k)
    if p > 0:
        sum_AR = torch.sum(VAR, dim=2)
        CVAR = torch.inverse(torch.eye(k) + sum_AR)

    long_run_cov = torch.matmul(CVAR, torch.matmul(CVMA, CVAR.t()))

    return seq[:, -T:], long_run_cov

def sim_FD(T, k, d): # wrapper for fractionally differenced Gaussian noise
    return sim_VARFIMA(T, k, d, None, None)