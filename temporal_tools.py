import math
import cmath
import numpy as np
import matplotlib.pyplot as plt

# from numpy.core.defchararray import array

from scipy.fft import fft, ifft

def complex_wavelet(freq, res, n_cycles=7, bounds=[-2,2], scale=True,):
    '''
    returns the points of a complex wavelet between the bounds at a sampling resolution of 1/res
    
    --Parameters--
    freq : frequency of the wavelet in hz
    n_cycles : number of cycles in the wavelet (+freq precision, -temporal precision)
    res : resolution - number of samples between 0 and 1
    bounds : maximum and minimum x values
    scale : bool controlling whether the amplitude of higher frequency wavelets is scaled up
            (to counteract power scaling)
    '''
    sd = n_cycles / (2 * cmath.pi * freq)
    
    if scale:
        A = 1 / cmath.sqrt(sd * cmath.sqrt(cmath.pi))
    else:
        A = 1

    def sample(x):
        gaus_exp = (-1 * x**2) / (2 * sd**2)
        wave_exp = 1j * 2 * cmath.pi * freq * x
        return A * cmath.exp(gaus_exp+wave_exp)

    s_rate = 1 / res

    return [sample(x) for x in np.arange(bounds[0], bounds[1],  s_rate)]

def wavelet_conv(data, res, f_min, f_max, n_wavelets, n_cycles=7,
                    log_spacing=False, bounds=[-2,2], scale=True):

    step = (f_max - f_min) / n_wavelets

    wavelet_family = [complex_wavelet(f, res, n_cycles, bounds, scale)
                        for f in np.arange(f_min, f_max, step)]

    conv_len = len(data) + len(wavelet_family[0])
    pad = math.ceil(len(wavelet_family[0])/2)

    data_ft = fft(data, conv_len)
    results = []
    for wavelet in wavelet_family:
        wavelet_ft = fft(wavelet, conv_len)
        results.append(ifft(data_ft*wavelet_ft)[pad:-1*pad])

    return results
    
def extract_phase_power(data):
    phase = np.arctan(np.imag(data)/np.real(data))
    power = (data * np.conj(data)).astype(float)
    return phase,power

def phase_cluster(data):
    '''
    returns the phase clustering of the data at each time point

    --Parameters--
    data : N x T array - clustered across N for each time point T
    '''

    results = []
    for timepoint in data.T:
        results.append(abs(np.sum(np.exp(1j * timepoint)) / len(timepoint)))
    
    return results