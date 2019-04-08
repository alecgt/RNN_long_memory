'''
Semiparametric estimation tools for the long memory parameter.
'''
import numpy as np
import scipy.optimize as opt
import scipy.stats as scst

# UNIVARIATE ESTIMATOR (for initialization)
def K_fn(d,lam,pdg):
    m = len(lam)
    k = -np.log(m)+np.log(np.sum(pdg*(lam**(2*d))))-(d*(2/m)*np.sum(np.log(lam)))
    return k


def K_grad(d,lam,pdg):
    m = len(lam)
    num = 2*np.sum(pdg*np.log(lam)*(lam**(2*d)))
    denom = np.sum(pdg*lam**(2*d))
    grad = np.array((num/denom-(2/m)*np.sum(np.log(lam))))
    return grad


def mypdg(seq):
    T = len(seq)
    lam = 2*np.pi*np.arange(T)/T
    return lam,(1/(2*np.pi*T))*np.abs(np.fft.fft(seq))**2


def d_lw(seq,m):
    lam,pdg = mypdg(seq-np.mean(seq))
    opti = opt.minimize(K_fn,0,args = (lam[1:m],pdg[1:m]),jac=K_grad,method='L-BFGS-B',bounds=[(-0.5,0.5)])
    return opti

# MULTIVARIATE ESTIMATION
def multivar_pdg_m(seq, m):  # only compute periodogram for Fourier frequencies 2*pi*(1,...,m)/T
    k, T = seq.shape
    out = np.zeros((k, k, m), dtype=np.complex)
    seq_fft = np.fft.fft(seq)
    for i in range(1, m + 1):
        out[:, :, i - 1] = np.outer(seq_fft[:, i], np.conj(seq_fft[:, i]))
        x = 2 * np.pi * np.arange(T / 2 + 1) / T  # Fourier frequencies
    return x[1:(m + 1)], (1 / (2 * np.pi * T)) * out


def lambda_d(lam, d):
    return np.diag(lam ** -d * np.exp(1j * (np.pi - lam) * d / 2))


def G_hat(d, lam, pdg):
    ghat = np.zeros(pdg.shape)
    m = pdg.shape[2]
    for i in range(m):
        linv = np.diag(1 / (np.diag(lambda_d(lam[i], d))))  # np.linalg.inv(lambda_d(lam[i],d))
        ghat[:, :, i] = np.real(np.dot(np.dot(linv, pdg[:, :, i]), np.conj(linv)))
    return (1 / m) * np.sum(ghat, axis=2)


def R_d(d, lam, pdg, verbose=False):
    m = pdg.shape[2]
    ghat = G_hat(d, lam, pdg)
    s, det = np.linalg.slogdet(ghat)
    R = s * det - (2 / m) * np.sum(d) * np.sum(np.log(lam))

    R_grad = GSE_grad(d, lam, pdg, ghat)
    return R, R_grad


def multi_GSE(seq, m, options={}, method='L-BFGS-B'):
    lam, pdg = multivar_pdg_m(seq, m)
    k = pdg.shape[0]
    init_d = np.zeros(k)
    for i in range(k):
        init_d[i] = d_lw(seq[i, :], m)['x'][0]  # initalize at vector of univariate estimates

    opti = opt.minimize(R_d, init_d, args=(lam, pdg), method=method, bounds=[(-0.5, 0.5)] * k, jac=True,
                        options=options)
    return opti


def compute_total_memory(seq):
    k, T = seq.shape
    m = int(np.sqrt(T))

    cvg = False
    while not cvg:
        GSE = multi_GSE(seq, m)
        cvg = GSE['success']

    d = GSE['x']
    tot_mem = np.mean(d)
    asy_var = np.sum(dGSE_asymptotic_var(seq, d, m)) / k**2
    z = tot_mem / np.sqrt(asy_var)
    p_val = 1-scst.norm.cdf(z)

    return tot_mem, asy_var, p_val



# GRADIENT
def GSE_grad(d, lam, pdg, ghat):  # gradient of R_d w.r.t. d
    grad = np.zeros(len(d))
    ghat_inv = np.linalg.pinv(ghat)
    m = pdg.shape[2]

    for i in range(len(d)):
        gdv = Ghat_deriv(d, lam, pdg, i)
        grad[i] = np.trace(np.dot(ghat_inv, gdv)) - (2 / m) * np.sum(np.log(lam))
    return grad


def Ghat_deriv(d, lam, pdg, l):
    k = pdg.shape[1]
    m = pdg.shape[2]
    dl = d[l]

    c_pls = np.log(lam) + 1j * (np.pi - lam) / 2
    c_min = np.log(lam) - 1j * (np.pi - lam) / 2

    d_mat = np.tile(d, (m, 1)).T  # all k x m
    c_pls_mat = np.tile(c_pls, (k, 1))
    c_min_mat = np.tile(c_min, (k, 1))

    row = np.real(np.sum(pdg[l, :, :] * c_min_mat * np.exp(dl * c_min_mat) * np.exp(d_mat * c_pls_mat), axis=1))
    col = np.real(np.sum(pdg[:, l, :] * c_pls_mat * np.exp(dl * c_pls_mat) * np.exp(d_mat * c_min_mat), axis=1))
    dg = np.real(np.sum(pdg[l, l, :] * 2 * np.log(lam) * np.exp(2 * dl * np.log(lam))))

    out = np.zeros((k, k))
    out[l, :] = row
    out[:, l] = col
    out[l, l] = dg
    return (1 / m) * out


# asymptotic variance of Shimotsu estimator
def dGSE_asymptotic_var(seq, d, m):
    lam, pdg = multivar_pdg_m(seq, m)
    k = len(d)
    ghat = G_hat(d, lam, pdg)
    ginv = np.linalg.inv(ghat)
    omega = 2 * (np.eye(k) + (ghat * ginv) + ((np.pi ** 2) / 4) * (ghat * ginv - np.eye(k)))
    return np.linalg.inv(omega)