# Copyright Thomas Lecocq, originally European Union Public Licence V. 1.1
# from msnoise.api

import numpy as np
from scipy.fftpack import fft, ifft, fftfreq, next_fast_len
import logging

log = logging.getLogger('yam.correlate')


def check_and_phase_shift(trace):
    if trace.stats.npts < 10 * trace.stats.sampling_rate:
        trace.data = np.zeros(trace.stats.npts)
        return
    dt = np.mod(trace.stats.starttime.datetime.microsecond * 1.0e-6,
                trace.stats.delta)
    if (trace.stats.delta - dt) <= np.finfo(float).eps:
        dt = 0.
    if dt != 0.:
        if dt <= (trace.stats.delta / 2.):
            dt = -dt
        else:
            dt = (trace.stats.delta - dt)
        log.debug("correcting time of trace %s with starttime %s by %.6fs",
                  trace.id, trace.stats.starttime, dt)
        trace.detrend(type="demean")
        trace.detrend(type="simple")
        trace.taper(max_percentage=None, max_length=1.0)

        n = next_fast_len(int(trace.stats.npts))
        FFTdata = fft(trace.data, n=n)
        freq = fftfreq(n, d=trace.stats.delta)
        FFTdata = FFTdata * np.exp(1j * 2. * np.pi * freq * dt)
        FFTdata = FFTdata.astype(np.complex64)
        ifft(FFTdata, n=n, overwrite_x=True)
        trace.data = np.real(FFTdata[:len(trace.data)]).astype(np.float)
        trace.stats.starttime += dt
#        clean_scipy_cache()
