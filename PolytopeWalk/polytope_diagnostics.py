import numpy as np
from numpy.fft import fft, ifft
import arviz as az

# This file holds some diagnostic functions. Some use arviz while for 
# univariate_psrf and ess I copied the Volesti methods to make sure the values are the same.

def univariate_psrf(samples):

    arr = np.asarray(samples)
    N, d = arr.shape
    N1 = N // 2
    N2 = N - N1
    rhat = np.zeros(d)

    for i in range(d):
        coord = arr[:, i]
        mean1 = np.mean(coord[:N1])
        mean2 = np.mean(coord[N1:])
        W1 = np.var(coord[:N1], ddof=1)
        W2 = np.var(coord[N1:], ddof=1)
        W = 0.5 * (W1 + W2)
        mean00 = np.mean(coord)
        B = (mean1 - mean00)**2 + (mean2 - mean00)**2
        sigma = ((N1 - 1) / N1) * W + B
        R = np.sqrt(sigma / W)
        rhat[i] = R

    rhat_max = np.nanmax(rhat)
    return rhat_max

def ess(samples):

    arr = np.asarray(samples)
    N, d = arr.shape
    N_even = N - (N % 2)
    ess = np.zeros(d)

    for i in range(d):
        x = arr[:, i]
        x = x - np.mean(x)
        var = np.var(x)
        if var < 1e-16:
            ess[i] = np.nan
            continue

        x = x / np.sqrt(var)

        x_padded = np.concatenate([x, np.zeros(N)])
        f = fft(x_padded)
        psd = np.real(f * np.conjugate(f))
        acf = np.real(ifft(psd))[:N] / N

        acf = acf[:N_even]
        acf_summed = acf[0::2] + acf[1::2]

        for j in range(1, len(acf_summed)):
            if acf_summed[j] > acf_summed[j - 1]:
                acf_summed[j] = acf_summed[j - 1]

        gap = 0.0
        for val in acf_summed:
            if val > 0:
                gap += val

        gap = 2 * gap - 1.0
        if gap < 1.0:
            gap = 1.0

        ess[i] = N / gap

    ess_min = np.nanmin(ess)
    return ess_min

def ess_arviz(samples):

    ds = az.convert_to_dataset(samples[None, :, :])  
    ess_vals = az.ess(ds).to_array().values          
    ess_min = float(np.nanmin(ess_vals))
    return ess_min


def psrf_arviz(samples):

    arr = np.asarray(samples)
    N, d = arr.shape
    half = N // 2

    # SAFETY CHECK: Handle odd number of samples 
    if N % 2 != 0:
        # Drop the last sample to make N even
        arr = arr[:-1, :]
        N -= 1

    # Split the single chain into two halves
    chains = np.stack([arr[:half, :], arr[half:, :]], axis=0)  # shape (2, half, d)

    # Convert to ArviZ dataset and compute R-hat
    ds = az.convert_to_dataset(chains)
    rhat_vals = az.rhat(ds).to_array().values # type: ignore

    # Handle NaNs safely
    if np.all(np.isnan(rhat_vals)):
        return np.nan

    return float(np.nanmax(rhat_vals))