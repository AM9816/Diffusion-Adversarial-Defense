
import torch
from torch import fft, nn, Tensor as T

from torch_common_clean import BPDA_TYPE, make_arbitrary_BPDA_identity








# proper freqpure frequency band blending 
# https://github.com/GaozhengPei/FreqPure
# https://arxiv.org/abs/2505.01267v1
_max_rad = (2*(.5**2))**.5
def _FFT_reconstruct_from_two_circle_backend(_l: T, _h: T, r: float, delta: float = .05) -> T:
# def _FFT_reconstruct_from_two_circle_BASIC(_l: T, _h: T, r: float, delta: float = .05) -> T:
    n, c, h, w = _l.shape
    d = _l.device

    fft_low = fft.fft2(_l)
    fft_high = fft.fft2(_h)


    grid = fft.fftfreq(h, device=d), fft.fftfreq(w, device=d)
    x, y = torch.meshgrid(*grid, indexing='ij')
    
    rad = torch.sqrt(x.pow(2) + y.pow(2))
    mask = (rad <= _max_rad * r).float()

    amp_low, phase_low = fft_low.abs(), fft_low.angle()
    amp_high, phase_high = fft_high.abs(), fft_high.angle()

    amp_out = amp_low * mask + amp_high * (1. - mask)

    if delta > 0.0:
        clipped_phase = torch.clamp(phase_high, phase_low - delta, phase_low + delta)
        phase_out = clipped_phase * mask + phase_high * (1. - mask)
    else:
        phase_out = phase_low * mask + phase_high * (1. - mask)

    out = torch.polar(amp_out, phase_out)

    return fft.ifft2(out).real.clamp(0, 1)



def _FFT_reconstruct_from_two_circle_BASIC(_l:T, _h:T, r):
    n, c, h, w = _l.shape
    d=_l.device

    fft_low = fft.fft2(_l)
    fft_high= fft.fft2(_h)

    grid = fft.fftfreq(h, device=d),\
           fft.fftfreq(w, device=d)

    x, y = torch.meshgrid(*grid, indexing='ij')
    rad = torch.sqrt(x.pow(2)+y.pow(2))
    mask = (rad <= _max_rad*r).float()

    out = fft_low * mask + fft_high * (1. - mask)

    return fft.ifft2(out).real.clamp(0, 1)




_FFT_reconstruct_from_two_circle =  _FFT_reconstruct_from_two_circle_backend\
    # make_arbitrary_BPDA_identity(
    #     _FFT_reconstruct_from_two_circle_backend)

