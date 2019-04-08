import matplotlib.pyplot as plt
import numpy as np

# helper function to generate example plot in paper / notebook
def plot_example(ar_data, fiar_data, lam):
    ar_acvs, ar_sdf = ar_data
    fiar_acvs, fiar_sdf = fiar_data

    max_lag = len(ar_acvs)

    fig, (ax1, ax3) = plt.subplots(nrows=1, ncols=2, figsize=(10, 4))
    ax1.plot(np.arange(max_lag), ar_acvs/ar_acvs[0], linewidth=3.0, label='AR(1)')
    ax1.plot(np.arange(max_lag), fiar_acvs/fiar_acvs[0], linewidth=3.0, label='FI(d)-AR(1)')
    ax1.set_xlabel('t')
    ax1.set_ylabel('γ(t)')
    ax1.legend(loc=9)

    ax2 = ax1.twinx()
    ax2.plot(np.arange(max_lag), np.cumsum(np.abs(ar_acvs/ar_acvs[0])), linewidth=3, linestyle='--')
    ax2.plot(np.arange(max_lag), np.cumsum(np.abs(fiar_acvs/fiar_acvs[0])), linewidth=3, linestyle='--')
    ax2.set_ylabel('Σ γ(t)')
    ax2.set_title('Autocovariance sequence (solid)\n and partial sums (dashed)')

    ax3.plot(-2*np.log(lam), np.log(ar_sdf), linewidth=3.0, label='AR(1)')
    ax3.plot(-2*np.log(lam), np.log(fiar_sdf), linewidth=3.0, label='FI(d)-AR(1)')
    ax3.set_xlabel('-2 log λ')
    ax3.set_ylabel('log f(λ)')
    ax3.legend()
    ax3.set_title('Log spectral density function')
    fig.tight_layout()
    plt.subplots_adjust(left=-0.05)
    plt.show()
