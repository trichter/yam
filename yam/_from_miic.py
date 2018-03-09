### file miic.core.stretch_mod adapted by TE ###
"""
@author:
Eraldo Pomponi
@copyright:
The MIIC Development Team (eraldo.pomponi@uni-leipzig.de)
@license:
GNU Lesser General Public License, Version 3
(http://www.gnu.org/copyleft/lesser.html)
Created on Aug 19, 2011
"""
import logging

# Main imports
#import scipy.ndimage
import numpy as np
from scipy.interpolate import UnivariateSpline
from obspy.signal.invsim import cosine_taper

log = logging.getLogger('yam.stretch')

## ETS imports
#try:
#    BC_UI = True
#    from traits.api import HasTraits, Int, Float, Bool, List, Enum
#    from traitsui.api import View, Item, Tabbed
#except ImportError:
#    BC_UI = False
#    pass
#
#
##from miic.core.miic_utils import nextpow2, from_str_to_datetime, dv_check, \
##    flatten_recarray, convert_time
#
#
#def _stretch_mat_creation(ref_tr, str_range=0.1, nstr=100):
#    """ Matrix of stretched instance of a reference trace.
#    The reference trace is stretched using a cubic spline interpolation
#    algorithm form ``-str_range`` to ``str_range`` (in %) for totally
#    ``nstr`` steps.
#    The output of this function is a matrix containing the stretched version
#    of the reference trace (one each row) (``new_m``) and the corresponding
#    stretching amount (``deltas``).
#    :type ref_tr: :class:`~numpy.ndarray`
#    :param ref_tr: 1d ndarray. The reference trace that will be stretched
#    :type str_range: float
#    :param str_range: Amount, in percent, of the desired stretching (one side)
#    :type nstr: int
#    :param nstr: Number of stretching steps (one side)
#    :rtype: :class:`~numpy.ndarray`, float
#    :return:
#        **new_m**:
#            2d ndarray of stretched version of the reference trace.
#            Its size is ``(nstr,len(ref_tr)/2)`` if ``signle_side==True``
#            otherwise it is ``(nstr,len(ref_tr))``
#        **deltas**: List of float, stretch amount for each row
#            of ``new_m``
#    """
#
#    n = len(ref_tr)
#    samples_idx = np.arange(n) - n // 2
#    deltas = 1 + np.linspace(-str_range, str_range, nstr)
#    str_timemat = np.zeros((nstr, n))
#    for ii in np.arange(nstr):
#        str_timemat[ii, :] = samples_idx / deltas[nstr - 1 - ii]
#    new_m = np.zeros_like(str_timemat)
#    coord = np.zeros((2, n))
#    for (i, row) in enumerate(str_timemat):
#        coord[0, :] = row + n // 2
#        new_m[i, :] = scipy.ndimage.map_coordinates(\
#                                    ref_tr.reshape((len(ref_tr), 1)), coord)
#    return new_m, deltas
#
#
#if BC_UI:
#    class _stretch_mat_creation_view(HasTraits):
#        str_range = Float(0.1)
#        nstr = Int(100)
#        single_side = Bool(False)
#
#        trait_view = View(Item('str_range'),
#                          Item('nstr'),
#                          Item('single_side'))


def velocity_change_estimete(mat, tw, strrefmat, strvec, sides='both',
                             return_sim_mat=False, remove_nans=True):
    """ Velocity change estimate through stretching and comparison.
    Velocity changes are estimated comparing each correlation function stored
    in the ``mat`` matrix (one for each row) with ``strrefmat.shape[0]``
    stretched versions  of a reference trace stored in ``strrefmat``.
    The stretch amount for each row of ``strrefmat`` must be passed in the
    ``strvec`` vector.
    The best match (stretch amount and corresponding correlation value) is
    calculated on different time windows (each row of ``tw`` is a different
    one) instead of on the whole trace.
    :type mat: :class:`~numpy.ndarray`
    :param mat: 2d ndarray containing the correlation functions.
        One for each row.
    :type tw: :class:`~numpy.ndarray` of int
    :param tw: 2d ndarray of time windows to be use in the velocity change
         estimate.
    :type strrefmat: :class:`~numpy.ndarray`
    :param strrefmat: 2D array containing stretched version of the reference
         matrix
    :type strvec: :class:`~numpy.ndarray` or list
    :param strvec: Stretch amount for each row of ``strrefmat``
    :type sides: string
    :param sides: Side of the reference matrix to be used for the velocity
        change estimate ('both' | 'left' | 'right' | 'single')
    :type remove_nans: bool
    :param remove_nans: If `True` applay :func:`~numpy.nan_to_num` to the
        given correlation matrix before any other operation.
    :rtype: Dictionary
    :return: **dv**: Dictionary with the following keys
        *corr*: 2d ndarray containing the correlation value for the best
            match for each row of ``mat`` and for each time window.
            Its dimension is: :func:(len(tw),mat.shape[1])
        *value*: 2d ndarray containing the stretch amount corresponding to
            the best match for each row of ``mat`` and for each time window.
            Its dimension is: :func:(len(tw),mat.shape[1])
        *sim_mat*: 3d ndarray containing the similarity matricies that
            indicate the correlation coefficient with the reference for the
            different time windows, different times and different amount of
            stretching.
            Its dimension is: :py:func:`(len(tw),mat.shape[1],len(strvec))`
        *second_axis*: It contains the stretch vector used for the velocity
            change estimate.
        *vale_type*: It is equal to 'stretch' and specify the content of
            the returned 'value'.
        *method*: It is equal to 'single_ref' and specify in which "way" the
            values have been obtained.
    """

    # Mat must be a 2d vector in every case so
    mat = np.atleast_2d(mat)

    assert(strrefmat.shape[1] == mat.shape[1])

    if remove_nans:
        mat = np.nan_to_num(mat)

    center_p = np.floor((mat.shape[1] - 1.) / 2.)

    nstr = strrefmat.shape[0]

    corr = np.zeros((len(tw), mat.shape[0]))
    dt = np.zeros((len(tw), mat.shape[0]))
    sim_mat = np.zeros([mat.shape[0], len(strvec), len(tw)])

    for (ii, ctw) in enumerate(tw):

        if sides == 'both':
            ctw = np.hstack((center_p - ctw[::-1],
                             center_p + ctw)).astype(np.int32)
        elif sides == 'left':
            ctw = (center_p - ctw[::-1]).astype(np.int32)
        elif sides == 'right':
            ctw = (center_p + ctw).astype(np.int32)
        elif sides == 'single':
            ctw = ctw.astype(np.int32)
        else:
            msg = 'sides = %s not a valid option. Using sides = single'
            log.warning(msg, sides)

        mask = np.zeros((mat.shape[1],))
        mask[ctw] = 1

        ref_mask_mat = np.tile(mask, (nstr, 1))
        mat_mask_mat = np.tile(mask, (mat.shape[0], 1))

        first = mat * mat_mask_mat
        second = strrefmat * ref_mask_mat

        dprod = np.dot(first, second.T)

        # Normalization
        f_sq = np.sum(first ** 2, axis=1)
        s_sq = np.sum(second ** 2, axis=1)

        f_sq = f_sq.reshape(1, len(f_sq))
        s_sq = s_sq.reshape(1, len(s_sq))

        den = np.sqrt(np.dot(f_sq.T, s_sq))

        tmp = dprod / den
        tmp[np.isnan(tmp)] = 0  # warning is already issued
        sim_mat[:, :, ii] = tmp

#        tmp_corr_vect = tmp.max(axis=1)
#        corr[ii, :] = tmp_corr_vect
#        dt[ii, :] = strvec[tmp.argmax(axis=1)]

        # Set dt to NaN where the correlation is NaN instead of having it equal
        # to one of the two stretch_range limits
#        dt[ii, np.isnan(tmp_corr_vect)] = np.nan

    dv = {#'corr': corr,
          #'value': dt,
          'second_axis': strvec,
          'value_type': np.array(['stretch']),
          'method': np.array(['single_ref'])}

    if return_sim_mat:
        dv.update({'sim_mat': sim_mat})

    return dv


#if BC_UI:
#    class _velocity_change_estimete_view(HasTraits):
#        sides = Enum('both', 'left', 'right', 'single')
#        trait_view = View(Item('sides'))
#
#
#def time_shift_estimate(corr_data, ref_trc=None, tw=None, shift_range=10,
#                        shift_steps=100, single_sided=False,
#                        return_sim_mat=True, remove_nans=True):
#    """ Time shift estimate through shifting and comparison.
#    This function is intended to estimate shift of traces as they can occur
#    in noise cross-correlation in case of drifting clocks.
#    Time shifts are estimated comparing each correlation function stored
#    in the ``corr_data`` matrix (one for each row) with ``shift_steps``
#    shifted versions  of reference trace stored in ``ref_trc``.
#    The maximum amount of shifting may be passed in ``shift_range``.
#    The best match (shifting amount and corresponding correlation value) is
#    calculated on different time windows. If ``tw = None`` the shifting is
#    estimated on the whole trace.
#    :type corr_data: :class:`~numpy.ndarray`
#    :param corr_data: 2d ndarray containing the correlation functions.
#        One for each row.
#    :type ref_trc: :class:`~numpy.ndarray`
#    :param ref_trc: 1D array containing the reference trace to be shifted and
#        compared to the individual traces in ``mat``
#    :type tw: list of :class:`~numpy.ndarray` of int
#    :param tw: list of 1D ndarrays holding the indices of sampels in the time
#        windows to be use in the time shift estimate. The sampels are counted
#        from the zero lag time with the index of the first sample being 0. If
#        ``tw = None`` the full time range is used.
#    :type shift_range: scalar
#    :param shift_range: Maximum amount of time shift in samples (in one
#        direction).
#        Shifting is tested in both directions from ``-shift_range`` to
#        ``shift_range``
#    :type shift_steps: scalar`
#    :param shift_steps: Number of shifted version to be tested. The increment
#            will be ``(2 * shift_range) / shift_steps``
#    :type sinlge_sided: boolean
#    :param single_sided: If ``True`` the zero lag time of the traces is in the
#        first sample. If ``False`` zero lag is assumed to be in the center of
#        the traces and the shifting is evaluated on the causal and acausal
#        parts of the traces separately and averaged. This is done to avoid bias
#        from velocity changes (stretching) in the case of strongly asymmetric
#        traces.
#    :type remove_nans: bool
#    :param remove_nans: If `True` applay :func:`~numpy.nan_to_num` to the
#        given correlation matrix before any other operation.
#    :rtype: Dictionary
#    :return: **dt**: Dictionary with the following keys
#        *corr*: :class:`~numpy.ndarray` 2d ndarray containing the correlation
#            value for the best match for each row of ``mat`` and for each
#            time window.
#            Its dimension is: :func:(len(tw),mat.shape[1])
#        *value*: :class:`~numpy.ndarray` 2d ndarray containing the amount of
#            shifting corresponding to the best match for each row of ``mat``
#            and for each time window. Shift is measured in units of the
#            sampling interval.
#            Its dimension is: :py:func:`(len(tw),mat.shape[1])`
#        *sim_mat*: 2d ndarray containing the similarity matrix that
#            indicate the time shift respect to the reference for the
#            selected time windows, for different times and different amount of
#            shift.
#            Its dimension is: :py:func:`(mat.shape[1],shift_steps)`
#        *second_axis*: It contains the shift vector used for the velocity
#            change estimate.
#        *vale_type*: It is equal to 'shift' and specify the content of
#            the returned 'value'.
#        *method*: It is equal to 'time_shift' and specify in which "way" the
#            values have been obtained.
#    """
#
#    # Mat must be a 2d vector in every case so
#    mat = np.atleast_2d(corr_data)
#
#    # FIX: Trace back this problem to the original source and remove
#    # this statement
#    if remove_nans:
#        mat = np.nan_to_num(mat)
#
#    # generate the reference trace if not given (use the whole time span)
#    if ref_trc is None:
#        ref_trc = np.nansum(mat, axis=0) / mat.shape[0]
#
#    # generate time window if not given (use the full length of
#    # the correlation trace)
#    if tw is None:
#        if single_sided:
#            tw = time_windows_creation([0], [mat.shape[1]])
#        # if center is in the middle use only half the length for the
#        # time window
#        else:
#            tw = time_windows_creation([0],
#                                       [int(np.floor(mat.shape[1] / 2.))])
#
#    # taper and extend the reference trace to avoid interpolation
#    # artefacts at the ends of the trace
#    taper = cosine_taper(len(ref_trc), 0.05)
#    ref_trc *= taper
#
#    # different values of shifting to be tested
#    shifts = np.linspace(-shift_range, shift_range, shift_steps)
#
#    # time axis
#    time_idx = np.arange(len(ref_trc))
#
#    # create the array to hold the shifted traces
#    ref_shift = np.zeros((len(shifts), len(ref_trc)))
#
#    # create a spline object for the reference trace
#    ref_tr_spline = UnivariateSpline(time_idx, ref_trc, s=0)
#
#    # evaluate the spline object at different points and put in the prepared
#    # array
#    for (k, this_shift) in enumerate(shifts):
#        ref_shift[k, :] = ref_tr_spline(time_idx - this_shift)
#
#    # search best fit of the crosscorrs to one of the shifted ref_traces
#    if single_sided:
#        vdict = velocity_change_estimete(mat, tw, ref_shift,
#                                         shifts, sides='single',
#                                         return_sim_mat=True,
#                                         remove_nans=remove_nans)
#        corr = vdict['corr']
#        shift = vdict['value']
#        sim_mat = vdict['sim_mat']
#
#    else:
#        """
#        # estimate shifts for causal and acausal part individually and avarage
#        # to avoid apparent shift from velocity change and asymmetric
#        # amplitudes
#        lvdict = velocity_change_estimete(mat, tw, ref_shift,
#                                          shifts,
#                                          sides='left',
#                                          return_sim_mat=True,
#                                          remove_nans=remove_nans)
#        lcorr = lvdict['corr']
#        lshift = lvdict['value']
#        lsim_mat = lvdict['sim_mat']
#        rvdict = velocity_change_estimete(mat, tw, ref_shift,
#                                          shifts,
#                                          sides='right',
#                                          return_sim_mat=True,
#                                          remove_nans=remove_nans)
#        rcorr = rvdict['corr']
#        rshift = rvdict['value']
#        rsim_mat = rvdict['sim_mat']
#        shift = np.zeros_like(lshift)
#        corr = np.zeros_like(lshift)
#        sim_mat = np.zeros_like(lsim_mat)
#
#        corr = (lcorr + rcorr) / 2.
#        shift = (lshift + rshift) / 2.
#        sim_mat = (lsim_mat + rsim_mat) / 2.
#        """
#        dtdict = velocity_change_estimete(mat, tw, ref_shift,
#                                          shifts,
#                                          sides='both',
#                                          return_sim_mat=True,
#                                          remove_nans=remove_nans)
#        corr = dtdict['corr']
#        shift = dtdict['value']
#        sim_mat = dtdict['sim_mat']
#
#
#    # create the result dictionary
#    dt = {'corr': np.squeeze(corr),
#          'value': np.squeeze(shift),
#          'second_axis': shifts,
#          'value_type': np.array(['shift']),
#          'method': np.array(['time_shift'])}
#
#    if return_sim_mat:
#        dt.update({'sim_mat': np.squeeze(sim_mat)})
#
#    return dt
#
#
#if BC_UI:
#    class _time_shift_estimate_view(HasTraits):
#
#        shift_range = Float(10.0)
#        shift_steps = Int(100)
#        single_sided = Bool(False)
#
#        trait_view = View(Item('shift_range'),
#                          Item('shift_steps'),
#                          Item('single_sided'))
#
#
#def time_shift_apply(corr_data, shift):
#    """ Apply time shift to traces.
#    Apply time shifts to traces e.g. to align them to a common time base.
#    Such shifts can occur in correlation traces in case of a drifting clock.
#    This function ``applies`` the shifts. To correct for shift estimated with
#    :class:`~miic.core.stretch_mod.time_shift_estimate` you need to apply
#    negative shifts.
#    Shifting is done in frequency domain with 5% tapering.
#    :type corr_data: :py:class:`~numpy.ndarray`
#    :param corr_data: 2d ndarray containing the correlation functions that are
#        to be shifted.
#        One for each row.
#    :type shift: :py:class:`~numpy.ndarray`
#    :param shift: ndarray with shift.shape[0] = corr_data.shape[0] containing
#        the shifts in units of the sampling interval by which the trace are to
#        be shifted
#    :rtype: :py:class:`~numpy.ndarray`
#    :return: **shifted_mat**: shifted version of the input matrix
#    """
#
#    # Mat must be a 2d vector in every case so
#    mat = np.atleast_2d(corr_data)
#
#    # check input
#    # shift is just a 1d array
#    if len(shift.shape) == 1:
#        t_shift = np.zeros([shift.shape[0], 1])
#        t_shift[:, 0] = shift
#        shift = t_shift
#    # shift has the wrong length
#    elif shift.shape[0] != mat.shape[0]:
#        msg = 'InputError: shift.shape[0] must be equal corr_data.shape[0]'
#        log.warning(msg)
#        return 0
#    # shift has multiple columns (multiple measurements for the same time)
#    if shift.shape[1] > 1:
#        shift = np.delete(shift, np.arange(1, shift.shape[1]), axis=1)
#
#    # taper the reference matrix to avoid interpolation
#    taper = cosine_taper(mat.shape[1], 0.05)
#    mat *= np.tile(taper, [mat.shape[0], 1])
#
#    # find a suitable length for the FFT
#    N = nextpow2(2 * mat.shape[1])
#    w = np.zeros([1, N / 2 + 1])
#
#    # original and shifted phase
#    w[0, :] = np.linspace(0, np.pi, N / 2 + 1)
#    pha = np.exp(-1j * (shift) * w)
#
#    # Fourier Transform
#    F = np.fft.rfft(mat, N, 1)
#
#    # apply the phase shift
#    sF = F * pha
#
#    # transform to time domain
#    smat = np.fft.irfft(sF)
#
#    # cut to original size
#    shifted_mat = smat[:, 0:mat.shape[1]]
#    return shifted_mat
#
#
#if BC_UI:
#    class _time_shift_apply_view(HasTraits):
#        shift = List()
#        corr_data = List()
#        trait_view = View(Item('shift'), Item('corr_data'))


def time_stretch_estimate(corr_data, ref_trc=None, tw=None, stretch_range=0.1,
                          stretch_steps=100, sides='both', remove_nans=True):
    """ Time stretch estimate through stretch and comparison.
    This function estimates stretching of the time axis of traces as it can
    occur if the propagation velocity changes.
    Time stretching is estimated comparing each correlation function stored
    in the ``corr_data`` matrix (one for each row) with ``stretch_steps``
    stretched versions  of reference trace stored in ``ref_trc``.
    The maximum amount of stretching may be passed in ``stretch_range``. The
    time axis is multiplied by exp(stretch).
    The best match (stretching amount and corresponding correlation value) is
    calculated on different time windows. If ``tw = None`` the stretching is
    estimated on the whole trace.
    :type corr_data: :class:`~numpy.ndarray`
    :param corr_data: 2d ndarray containing the correlation functions.
        One for each row.
    :type ref_trc: :class:`~numpy.ndarray`
    :param ref_trc: 1D array containing the reference trace to be shifted and
        compared to the individual traces in ``mat``
    :type tw: list of :class:`~numpy.ndarray` of int
    :param tw: list of 1D ndarrays holding the indices of sampels in the time
        windows to be use in the time shift estimate. The sampels are counted
        from the zero lag time with the index of the first sample being 0. If
        ``tw = None`` the full time range is used.
    :type stretch_range: scalar
    :param stretch_range: Maximum amount of relative stretching.
        Stretching and compression is tested from ``-stretch_range`` to
        ``stretch_range``.
    :type stretch_steps: scalar`
    :param stretch_steps: Number of shifted version to be tested. The
        increment will be ``(2 * stretch_range) / stretch_steps``
    :type sides: str
    :param sides: Side of the reference matrix to be used for the stretching
        estimate ('both' | 'left' | 'right' | 'single') ``single`` is used for
        one-sided signals from active sources with zero lag time is on the
        first sample. Other options assume that the zero lag time is in the
        center of the traces.
    :type remove_nans: bool
    :param remove_nans: If `True` applay :func:`~numpy.nan_to_num` to the
        given correlation matrix before any other operation.
    :rtype: Dictionary
    :return: **dv**: Dictionary with the following keys
        *corr*: 2d ndarray containing the correlation value for the best
            match for each row of ``mat`` and for each time window.
            Its dimension is: :func:(len(tw),mat.shape[1])
        *value*: 2d ndarray containing the stretch amount corresponding to
            the best match for each row of ``mat`` and for each time window.
            Stretch is a relative value corresponding to the negative relative
            velocity change -dv/v.
            Its dimension is: :func:(len(tw),mat.shape[1])
        *sim_mat*: 3d ndarray containing the similarity matricies that
            indicate the correlation coefficient with the reference for the
            different time windows, different times and different amount of
            stretching.
            Its dimension is: :py:func:`(len(tw),mat.shape[1],len(strvec))`
        *second_axis*: It contains the stretch vector used for the velocity
            change estimate.
        *vale_type*: It is equal to 'stretch' and specify the content of
            the returned 'value'.
        *method*: It is equal to 'single_ref' and specify in which "way" the
            values have been obtained.
    """

    # Mat must be a 2d vector in every case so
    mat = np.atleast_2d(corr_data)

    # FIX: Trace back this problem to the original source and remove
    # this statement
    if remove_nans:
        mat = np.nan_to_num(mat)

    # generate the reference trace if not given (use the whole time span)
    if ref_trc is None:
        ref_trc = np.nansum(mat, axis=0) / mat.shape[0]

    # generate time window if not given (use the full length of the correlation
    # trace)
    if tw is None:
        tw = time_windows_creation([0], [int(np.floor(mat.shape[1] / 2.))])

    # taper and extend the reference trace to avoid interpolation
    # artefacts at the ends of the trace
    taper = cosine_taper(len(ref_trc), 0.05)
    ref_trc *= taper

    # different values of shifting to be tested
    stretchs = np.linspace(-stretch_range, stretch_range, stretch_steps)
    time_facs = np.exp(-stretchs)

    # time axis
    if sides is not 'single':
        time_idx = np.arange(len(ref_trc)) - (len(ref_trc) - 1.) / 2.
    else:
        time_idx = np.arange(len(ref_trc))

    # create the array to hold the shifted traces
    ref_stretch = np.zeros((len(stretchs), len(ref_trc)))

    # create a spline object for the reference trace
    ref_tr_spline = UnivariateSpline(time_idx, ref_trc, s=0)

    # evaluate the spline object at different points and put in the prepared
    # array
    for (k, this_fac) in enumerate(time_facs):
        ref_stretch[k, :] = ref_tr_spline(time_idx * this_fac)

    # search best fit of the crosscorrs to one of the stretched ref_traces
    dv = velocity_change_estimete(mat, tw, ref_stretch,
                                  stretchs, sides=sides,
                                  return_sim_mat=True,
                                  remove_nans=remove_nans)

    # TODO: It is not really clear why it it necessary to transpose here so
    # this is the fist point where to look in case of errors.
    #dv['corr'] = dv['corr'].T
    #dv['value'] = dv['value'].T
    # dv.update({'stretch_vec': stretchs})

    return dv


#if BC_UI:
#    class _time_stretch_estimate_view(HasTraits):
#
#        stretch_range = Float(0.01)
#        stretch_steps = Int(100)
#        sides = Enum('both', 'left', 'right', 'sinlge')
#
#        trait_view = View(Item('stretch_range'),
#                          Item('stretch_steps'),
#                          Item('sides'))
#
#
#def time_stretch_apply(corr_data, stretch, single_sided=False):
#    """ Apply time axis stretch to traces.
#    Stretch the time axis of traces e.g. to compensate a velocity shift in the
#    propagation medium.
#    Such shifts can occur in corrlation traces in case of a drifting clock.
#    This function ``applies`` the stretches. To correct for stretching
#    estimated with :class:`~miic.core.stretch_mod.time_stretch_estimate`you
#    need to apply negative stretching.
#    :type corr_data: :class:`~numpy.ndarray`
#    :param corr_data: 2d ndarray containing the correlation functions that are
#        to be shifted.
#        One for each row.
#    :type stretch: :class:`~numpy.ndarray`
#    :param stretch: ndarray with stretch.shape[0] = corr_data.shape[0]
#        containing the stretches relative units.
#    :rtype: :class:`~numpy.ndarray`
#    :return: **stretched_mat**: stretched version of the input matrix
#    """
#
#    # Mat must be a 2d vector in every case so
#    mat = np.atleast_2d(corr_data)
#    # check input
#    # stretch is just a 1d array
#    if len(stretch.shape) == 1:
#        t_stretch = np.zeros([stretch.shape[0], 1])
#        t_stretch[:, 0] = stretch
#        stretch = t_stretch
#    # stretch has the wrong length
#    elif stretch.shape[0] != mat.shape[0]:
#        msg = 'InputError: shift.shape[0] must be equal corr_data.shape[0]'
#        log.warning(msg)
#        return 0
#    # shift has multiple columns (multiple measurements for the same time)
#    if stretch.shape[1] > 1:
#        stretch = np.delete(stretch, np.arange(1, stretch.shape[1]), axis=1)
#
#    # taper and extend the reference trace to avoid interpolation
#    # artefacts at the ends of the trace
#    taper = cosine_taper(mat.shape[1], 0.05)
#    mat *= np.tile(taper, [mat.shape[0], 1])
#
#    # time axis
#    if single_sided:
#        time_idx = np.arange(mat.shape[1])
#    else:
#        time_idx = np.arange(mat.shape[1]) - (mat.shape[1] - 1.) / 2.
#
#    # allocate space for the result
#    stretched_mat = np.zeros_like(mat)
#
#    # stretch every line
#    for (ii, line) in enumerate(mat):
#        s = UnivariateSpline(time_idx, line, s=0)
#        stretched_mat[ii, :] = s(time_idx * np.exp(-stretch[ii]))
#
#    return stretched_mat
#
#
#if BC_UI:
#    class _time_stretch_apply_view(HasTraits):
#        shift = List()
#        corr_data = List()
#        trait_view = View(Item('shift'), Item('corr_data'))
#
#
#def multi_ref_creation(corr_mat,
#                       rtime,
#                       freq=30,
#                       use_break_point=False,
#                       break_point=None):
#    """ Create the multi-reference traces
#    This function creates multi-reference traces according with the given
#    frequency.
#    In case of a break-point is passed, the intervals to consider for the
#    reference traces creation are symmetric respect to it.
#    :type corr_mat: :class:`~numpy.ndarray`
#    :param corr_mat: Correlation matrix with one correlation function on each
#        row
#    :type rtime: :class:`~numpy.array` of :class:`~datetime.datetime` objects
#    :param rtime: Time vector associated to the given correlation matrix. Its
#        lenght must be equal to the numer of columns of the `corr_mat`
#        parameter
#    :type freq: int
#    :param freq: One reference trace every `freq` days
#    :type use_break_point: bool
#    :param use_break_point: If `True` the reference traces are calculated in
#        different intervals of `freq` days symmetric respect to the
#        `break_point`
#    :type break_point: string
#    :param break_point: Brake point expressed as "YYYY-MM-DD"
#    """
#
#    from pandas import DataFrame
#    # do a local import here to ensure compatibility with older pandas
#    # versions outside this function
#    from pandas import date_range
#
#    if use_break_point and break_point == None:
#        log.error("A break point must be passed")
#        return None
#
#    if use_break_point and type(break_point) != str:
#        log.warning("wrong break_point format")
#        return None
#
#    if use_break_point:
#        bp = from_str_to_datetime(break_point, datetimefmt=True)
#
#        f_frw = "%iD" % int(freq)
#        f_bck = "-%iD" % int(freq)
#
#        # Time "backward" intervals starting from the break_point and going
#        # back to the first day available in the data.
#        dr = date_range(rtime[rtime <= bp].max(),
#                        rtime[rtime < bp].min(),
#                        freq=f_bck)[::-1]
#
#        # Time "forward" intervals starting from the break_point and going
#        # ahead to the last day available in the data.
#        dr1 = date_range(bp, rtime.max(), freq=f_frw)[1:]  # break_point must
#                                                            # be removed
#
#        dr = dr.append(dr1)
#    else:
#        f = "%iD" % int(freq)
#        dr = date_range(np.min(rtime), np.max(rtime), freq=f)
#
#
#    # DataFrame creation
#    df = DataFrame(corr_mat, index=rtime)
#
#    # GroupBy the given intervals
#    dfg = df.groupby(lambda x: np.sum(np.ones(dr.shape)[dr <= x]))
#
#    # Calculate the reference traces averaging the traces on each interval
#    df_ref = dfg.mean()
#
#    # Take the raw output
#    ref_mat = df_ref.values
#
#    return ref_mat
#
#
#def multi_ref_vchange_and_align(corr_data,
#                      ref_trs,
#                      tw=None,
#                      stretch_range=0.1,
#                      stretch_steps=100,
#                      sides='both',
#                      return_sim_mat=False,
#                      remove_nans=True):
#    """ Multi-reference dv estimate and alignment
#    :type corr_data: :class:`~numpy.ndarray`
#    :param corr_data: 2d ndarray containing the correlation functions.
#        One for each row.
#    :type ref_trc: :class:`~numpy.ndarray`
#    :param ref_trc: 1D array containing the reference trace to be shifted and
#        compared to the individual traces in ``mat``
#    :type tw: list of :class:`~numpy.ndarray` of int
#    :param tw: list of 1D ndarrays holding the indices of sampels in the time
#        windows to be use in the time shift estimate. The sampels are counted
#        from the zero lag time with the index of the first sample being 0. If
#        ``tw = None`` the full time range is used.
#    :type stretch_range: scalar
#    :param stretch_range: Maximum amount of relative stretching.
#        Stretching and compression is tested from ``-stretch_range`` to
#        ``stretch_range``.
#    :type stretch_steps: scalar`
#    :param stretch_steps: Number of shifted version to be tested. The
#        increment will be ``(2 * stretch_range) / stretch_steps``
#    :type single_sided: bool
#    :param single_sided: if True zero lag time is on the first sample. If
#        False the zero lag time is in the center of the traces.
#    :type sides: str
#    :param sides: Side of the reference matrix to be used for the stretching
#        estimate ('both' | 'left' | 'right' | 'single') ``single`` is used for
#        one-sided signals from active sources with zero lag time is on the
#        first sample. Other options assume that the zero lag time is in the
#        center of the traces.
#    :type return_sim_mat: bool
#    :param return_sim_mat: If `True` the returning dictionary contains also the
#        similarity matrix `sim_mat'.
#    :type remove_nans: bool
#    :param remove_nans: If `True` applay :func:`~numpy.nan_to_num` to the
#        given correlation matrix before any other operation.
#    :rtype: Dictionary
#    :return: **dv**: Dictionary with the following keys
#        *corr*: 2d ndarray containing the correlation value for the best
#            match for each row of ``mat`` and for each time window.
#            Its dimension is: :func:(len(tw),mat.shape[1])
#        *value*: 2d ndarray containing the stretch amount corresponding to
#            the best match for each row of ``mat`` and for each time window.
#            Stretch is a relative value corresponding to the negative relative
#            velocity change -dv/v.
#            Its dimension is: :func:(len(tw),mat.shape[1])
#        *sim_mat*: 3d ndarray containing the similarity matricies that
#            indicate the correlation coefficient with the reference for the
#            different time windows, different times and different amount of
#            stretching.
#            Its dimension is: :py:func:`(len(tw),mat.shape[1],len(strvec))`
#        *second_axis*: It contains the stretch vector used for the velocity
#            change estimate.
#        *vale_type*: It is equal to 'stretch' and specify the content of
#            the returned 'value'.
#        *method*: It is equal to 'single_ref' and specify in which "way" the
#            values have been obtained.
#    """
#
#    # FIX: Trace back this problem to the original source and remove
#    # this statement
#    if remove_nans:
#        corr_data = np.nan_to_num(corr_data)
#        ref_trs = np.nan_to_num(ref_trs)
#
#    if tw and len(tw) > 1:
#        log.warning("The multi-reference vchange evaluation doesn't handle "
#                    "multiple time windows. Only the first time-window will "
#                    "be used")
#        tw = tw[0]
#
#    multi_ref_panel = multi_ref_vchange(corr_data,
#                                        ref_trs,
#                                        tw=tw,
#                                        stretch_range=stretch_range,
#                                        stretch_steps=stretch_steps,
#                                        sides=sides,
#                                        remove_nans=remove_nans)
#
#    n_ref = len(multi_ref_panel.keys())
#
#    if n_ref > 1:
#        dv = estimate_reftr_shifts_from_dt_corr(multi_ref_panel,
#                                                return_sim_mat=return_sim_mat)
#    else:
#        # changed key here
#        dv = multi_ref_panel['reftr_0']
#
#    return dv
#
#
#if BC_UI:
#    class _multi_ref_vchange_and_align_view(HasTraits):
#
#        stretch_range = Float(0.01)
#        stretch_steps = Int(100)
#        sides = Enum('both', 'left', 'right', 'sinlge')
#        return_sim_mat = Bool(False)
#
#        trait_view = View(Item('stretch_range'),
#                          Item('stretch_steps'),
#                          Item('sides'),
#                          Item('return_sim_mat'))
#
#
#def multi_ref_vchange(corr_data,
#                      ref_trs,
#                      tw=None,
#                      stretch_range=0.1,
#                      stretch_steps=100,
#                      sides='both',
#                      remove_nans=True):
#    """ Velocity change estimate with single or multiple reference traces.
#    This function estimates the velocity change corresponding to each row of
#    the ``corr_data`` matrix respect to the reference trace/s passed in
#    ``ref_trs``.
#    The velocity change is estimated comparing each correlation function stored
#    in the ``corr_data`` matrix (one for each row) with ``stretch_steps``
#    stretched versions of reference/s trace stored in ``ref_trs``.
#    The maximum amount of stretching may be passed in ``stretch_range``.
#    The best match (stretching amount and corresponding correlation value) is
#    calculated on different time windows. If ``tw = None`` the stretching is
#    estimated on the whole trace.
#    The output is a dictionary with keys of the form ``"reftr_%d" % i``: One
#    for each reference trace. The corresponding ``value`` is aslo a dictionary
#    that has a structure conforming with
#    :py:class:`~miic.core.stretch_mod.time_stretch_estimate` output.
#    :type corr_data: :class:`~numpy.ndarray`
#    :param corr_data: 2d ndarray containing the correlation functions.
#        One for each row.
#    :type ref_trc: :class:`~numpy.ndarray`
#    :param ref_trc: 1D array containing the reference trace to be shifted and
#        compared to the individual traces in ``mat``
#    :type tw: list of :class:`~numpy.ndarray` of int
#    :param tw: list of 1D ndarrays holding the indices of sampels in the time
#        windows to be use in the time shift estimate. The sampels are counted
#        from the zero lag time with the index of the first sample being 0. If
#        ``tw = None`` the full time range is used.
#    :type stretch_range: scalar
#    :param stretch_range: Maximum amount of relative stretching.
#        Stretching and compression is tested from ``-stretch_range`` to
#        ``stretch_range``.
#    :type stretch_steps: scalar`
#    :param stretch_steps: Number of shifted version to be tested. The
#        increment will be ``(2 * stretch_range) / stretch_steps``
#    :type single_sided: bool
#    :param single_sided: if True zero lag time is on the first sample. If
#        False the zero lag time is in the center of the traces.
#    :type sides: str
#    :param sides: Side of the reference matrix to be used for the stretching
#        estimate ('both' | 'left' | 'right' | 'single') ``single`` is used for
#        one-sided signals from active sources with zero lag time is on the
#        first sample. Other options assume that the zero lag time is in the
#        center of the traces.
#    :type remove_nans: bool
#    :param remove_nans: If `True` applay :func:`~numpy.nan_to_num` to the
#        given correlation matrix before any other operation.
#    :rtype: dictionary
#    :return: **multi_ref_panel**: It is a dictionary that contains as much
#        dictionaries as reference traces have been used. The key format is
#        ``"reftr_%d" % i`` and the corresponding value is also a dictionary
#        with the structure described in
#        :py:class:`~miic.core.stretch_mod.time_stretch_estimate`
#    """
#
#    # FIX: Trace back this problem to the original source and remove
#    # this statement
#    if remove_nans:
#        corr_data = np.nan_to_num(corr_data)
#        ref_trs = np.nan_to_num(ref_trs)
#
#    # remove 1-dimensions
#    ref_trs = np.squeeze(ref_trs)
#
#    # check how many reference traces have been passed
#    try:
#        reftr_count, _ = ref_trs.shape
#    except ValueError:  # An array is passed
#        reftr_count = 1
#
#    # Distionary that will hold all the results
#    multi_ref_panel = {}
#
#    # When there is just 1 reference trace no loop is necessary
#    if reftr_count == 1:
#        key = "reftr_0"
#        value = time_stretch_estimate(corr_data,
#                                      ref_trc=ref_trs,
#                                      tw=tw,
#                                      stretch_range=stretch_range,
#                                      stretch_steps=stretch_steps,
#                                      sides=sides,
#                                      remove_nans=remove_nans)
#        multi_ref_panel.update({key: value})
#    else:  # For multiple-traces loops
#        for i in range(reftr_count):
#            ref_trc = ref_trs[i]
#            key = "reftr_%d" % int(i)
#            value = time_stretch_estimate(corr_data,
#                                          ref_trc=ref_trc,
#                                          tw=tw,
#                                          stretch_range=stretch_range,
#                                          stretch_steps=stretch_steps,
#                                          sides=sides,
#                                          remove_nans=remove_nans)
#            multi_ref_panel.update({key: value})
#
#    return multi_ref_panel
#
#
#if BC_UI:
#    class _multi_ref_vchange_view(HasTraits):
#
#        stretch_range = Float(0.01)
#        stretch_steps = Int(100)
#        sides = Enum('both', 'left', 'right', 'sinlge')
#
#        trait_view = View(Item('stretch_range'),
#                          Item('stretch_steps'),
#                          Item('sides'))
#
#
#def est_shift_from_dt_corr(dt1, dt2, corr1, corr2):
#    """ Estimation of a baseline shift between velocity-change measurements
#    preformed with different references.
#    The use of different reference traces obtaind from different reference
#    periods will result in a shift of the velocity-change curves that ideally
#    characterizes the velocity variation between the two reference periods.
#    Instead of directly measuring this velocity change from the two reference
#    traces it is calulated here as the weighted average of the point
#    differences between the two velocity-change curves weighted by their
#    inverse variance according to Weaver et al. GJI 2011 (On the precision of
#    noise correlation interferometry)
#    Input vertors must all be of the same lenth.
#    :type dt1: :class:`~numpy.ndarray`
#    :pram dt1: Velocity variation measured for reference A
#    :type dt2: :class:`~numpy.ndarray`
#    :pram dt2: Velocity variation measured for reference B
#    :type corr1: :class:`~numpy.ndarray`
#    :pram corr1: Correlation between velocity corrected trace and reference A
#    :type corr2: :class:`~numpy.ndarray`
#    :pram corr2: Correlation between velocity corrected trace and reference B
#    """
#
#    # Create maked arrays so that NaNs are handled properly
#    dt1 = np.ma.masked_array(dt1, mask=(np.isnan(dt1) | np.isinf(dt1)),
#                             fill_value=0)
#    dt2 = np.ma.masked_array(dt2, mask=(np.isnan(dt2) | np.isinf(dt2)),
#                             fill_value=0)
#
#    corr1 = np.ma.masked_array(corr1, mask=(np.isnan(corr1) | np.isinf(corr1)),
#                               fill_value=0)
#    corr2 = np.ma.masked_array(corr2, mask=(np.isnan(corr2) | np.isinf(corr2)),
#                               fill_value=0)
#
#    # Remove the points where the correlation is 0
#    no_zero = (corr1 > 0) & (corr2 > 0)
#    corr1 = corr1[no_zero]
#    corr2 = corr2[no_zero]
#
#    # Estimate the point-variance for the two curves
#    var1 = (1 - corr1 ** 2) / (4 * corr1 ** 2)
#    var2 = (1 - corr2 ** 2) / (4 * corr2 ** 2)
#
#    # Calculate the point-weight
#    wgt = 1 / (var1 + var2)
#    mask = (corr1 > 0.999) & (corr2 > 0.999)
#    wgt = wgt[~mask]
#
#    # This saves from returning a masked value
#    try:
#        wgt_s = np.sum(wgt).filled(np.nan)
#    except:
#        wgt_s = np.sum(wgt)
#
#    # Calculate the shift and the total weight as a cumulative sum of the
#    # weighted average of the two curves
#    shift = np.sum((dt1[~mask] - dt2[~mask]) * wgt) / wgt_s
#    comb_corr = np.sum((corr1[~mask] + corr2[~mask]) * wgt) / wgt_s
#
#    # This saves from returning masked values
#    try:
#        shift = shift.filled(np.nan)
#    except:
#        pass
#
#    try:
#        comb_corr = comb_corr.filled(np.nan)
#    except:
#        pass
#
#    return comb_corr, shift
#
#
#def estimate_reftr_shifts_from_dt_corr(multi_ref_panel, return_sim_mat=False):
#    """ Combine velocity-change measurements of the same data performed with
#    different references to a single curve.
#    For a set of velocity-change measurements performed with different
#    references this function estimates the relative offsets between all pairs
#    of the measurements as a weighted average of their difference with the
#    function :py:class:`~miic.core.stretch_mod.est_shift_from_dt_corr`.
#    A least squares solution in computed that combines the pairwise
#    differences to a consistent set of reference shifts. These shifts should
#    be similar to the velocity variations measured between the reference
#    traces. The consistent set of reference shifts is used to correct i.e.
#    shift the similarity matricies to a common reference. Finally the
#    corrected similarity matrices are averaged resulting in a single matrix
#    that is interpreted as before. The position of the crest is the combined
#    velocity change and the height of the crest is the correlation value.
#    :type multi_ref_panel: dictionay
#    :param multi_ref_panel: It is a dictionary with one (key,value) pair
#        for each reference trace. Its structure is described
#        in :py:class:`~miic.core.stretch_mod.multi_ref_vchange`
#    :type return_sim_mat: bool
#    :param return_sim_mat: If `True` the returning dictionary contains also the
#        similarity matrix `sim_mat'.
#    :rtype: Dictionary
#    :return: **dv**: Dictionary with the following keys
#        *corr*: 2d ndarray containing the correlation value for the best
#            match for each row of ``mat`` and for each time window.
#            Its dimension is: :func:(len(tw),mat.shape[1])
#        *value*: 2d ndarray containing the stretch amount corresponding to
#            the best match for each row of ``mat`` and for each time window.
#            Stretch is a relative value corresponding to the negative relative
#            velocity change -dv/v.
#            Its dimension is: :func:(len(tw),mat.shape[1])
#        *sim_mat*: 3d ndarray containing the similarity matricies that
#            indicate the correlation coefficient with the reference for the
#            different time windows, different times and different amount of
#            stretching.
#            Its dimension is: :py:func:`(len(tw),mat.shape[1],len(strvec))`
#        *second_axis*: It contains the stretch vector used for the velocity
#            change estimate.
#        *vale_type*: It is equal to 'stretch' and specify the content of
#            the returned 'value'.
#        *method*: It is equal to 'single_ref' and specify in which "way" the
#            values have been obtained.
#    """
#
#    # Vector with the stretching amount
#    stretch_vect = multi_ref_panel['reftr_0']['second_axis']
#    delta = stretch_vect[1] - stretch_vect[0]
#
#    n_ref = len(multi_ref_panel.keys())
#
#    corr = []
#    shift = []
#
#    if n_ref > 1:
#
#        # Loop over reftr
#        for reftr1 in np.sort(multi_ref_panel.keys()):
#            ref_idx = [reftr for reftr in np.sort(multi_ref_panel.keys())
#                       if reftr != reftr1]
#            for reftr2 in ref_idx:
#                ccorr, sshift = est_shift_from_dt_corr(
#                            np.squeeze(multi_ref_panel[reftr1]['value']),
#                            np.squeeze(multi_ref_panel[reftr2]['value']),
#                            np.squeeze(multi_ref_panel[reftr1]['corr']),
#                            np.squeeze(multi_ref_panel[reftr2]['corr']))
#                corr.append(ccorr)
#                shift.append(sshift)
#
#        G = _create_G(len(multi_ref_panel.keys()))
#        W = np.diag(np.array(corr))
#        D = np.array(shift)
#
#        # This conversion is necessary to achive the same speed with diffrent
#        # .dot implementation on different numpy versions.
#        D = np.nan_to_num(D)
#        W = np.nan_to_num(W)
#
#        GTW = np.dot(G.T, W)
#
#        # to_invert = np.dot(G.T, np.dot(W, G))
#        to_invert = np.dot(GTW, G)
#
#        # TODO: Get rid of the matrix inversion
#        left_op = np.linalg.pinv(to_invert)
#
#        # This conversion is necessary to achive the same speed with diffrent
#        # .dot implementation on different numpy versions.
#        left_op = np.nan_to_num(left_op)
#
#        m = np.dot(left_op, np.dot(GTW, D))
#        m = np.hstack((0, m))
#        m = m - np.mean(m)
#
#        # How many samples (int) each sim matrix must be rolled
#        m = np.around(m / delta, out=np.zeros_like(m, dtype='int32'))
#
#        row, col = np.squeeze(multi_ref_panel['reftr_0']['sim_mat']).shape
#
#        stmp = np.zeros((row, col, n_ref))
#
#        # Roll the sim_mat
#        for (i, reftr) in enumerate(np.sort(multi_ref_panel.keys())):
#
#            stmp[:, :, i] = \
#                np.roll(np.squeeze(multi_ref_panel[reftr]['sim_mat']),
#                        m[i],
#                        axis=1)
#
#        # Create a msked array to evaluate the mean
#        stmp_masked = np.ma.masked_array(stmp,
#                                         mask=(np.isnan(stmp) | \
#                                               np.isinf(stmp)),
#                                         fill_value=0)
#
#        # Evaluate the sim_mat for the multi-ref approach as the mean
#        # of the rolled sim_mat corresponfing to the individual reference
#        # traces
#        # When operating with masked arrays this operation creates a new
#        # object so the default fill_values returns to be 1e20
#        bsimmat = np.mean(stmp_masked, axis=2)
#
#        corr = np.max(bsimmat, axis=1)
#        dt = np.argmax(bsimmat, axis=1)
#
#        dt = stretch_vect[dt]
#
#        # Remove the mask
#        bsimmat = bsimmat.filled(np.nan)
#
#        try:
#            corr = corr.filled(np.nan)
#        except:
#            pass
#
#        try:
#            dt = dt.filled(np.nan).astype('int')
#        except:
#            pass
#
#        ret_dict = {'corr': corr,
#                    'value': dt,
#                    'second_axis': stretch_vect,
#                    'value_type': np.array(['stretch']),
#                    'method': np.array(['multi_ref'])}
#
#        if return_sim_mat:
#            ret_dict.update({'sim_mat': bsimmat})
#
#        return ret_dict
#
#    else:
#        log.error("For a single reference trace use the appropriate funtion")
#        return None


def time_windows_creation(starting_list, t_width):
    """ Time windows creation.
    A matrix containing one time window for each row is created. The starting
    samples of each one of them are passed in the ``starting_list`` parameter.
    The windows length ``t_width`` can be scalar or a list of values.
    In the latter case both lists ``starting_list`` and ``t_width`` must
    have the same length.
    :type starting_list: list
    :param starting_list: List of starting points
    :type t_width: int or list of int
    :param t_width: Windows length
    :rtype: :class:`~numpy.ndarray`
    :return: **tw_mat**: 2d ndarray containing the indexes of one time window
        for each row
    """

    if not np.isscalar(starting_list):
        if not np.isscalar(t_width) and len(t_width) != len(starting_list):
            raise ValueError("t_width must be a scalar or list of scalars of\
                            the same length as starting_list")

    tw_list = []

    if np.isscalar(starting_list):
        if np.isscalar(t_width):
            wlen = t_width
        else:
            wlen = t_width[0]
        tw_list.append(np.arange(starting_list, starting_list + wlen, 1))
    else:
        for (ii, cstart) in enumerate(starting_list):
            if np.isscalar(t_width):
                wlen = t_width
            else:
                wlen = t_width[ii]
            tw_list.append(np.arange(cstart, cstart + wlen, 1))

    return tw_list


#if BC_UI:
#    class _time_windows_creation_view(HasTraits):
#        starting_list = List(Int)
#        t_width = List(Int(50))
#
#        trait_view = View(Tabbed(Item('starting_list',
#                                      height=150,
#                                      width=80),
#                                 Item('t_width',
#                                      label='Win width in samples',
#                                      height=150,
#                                      width=80)))
#
#
#def _create_G(n_ref):
#    """ Create the G matrix for the multi-trace alignment
#    """
#
#    G = None
#    for jj in range(n_ref):
#        line = range(n_ref)
#        tline = [i for i in line if i != jj]
#        tG = np.zeros((len(tline), len(tline)))
#        if jj > 0:
#            tG[:, jj - 1] = -1
#        for ii in range(len(tline)):
#            if tline[ii] > 0:
#                tG[ii, tline[ii] - 1] = 1
#        if G is None:
#            G = tG
#        else:
#            G = np.vstack((G, tG))
#    return G
#
#
#def _id_from_stats(stats, ret_id_dict=False):
#    id_dict = {}
#    stats = flatten_recarray(stats)
#    keywords = ['network', 'station', 'location', 'channel']
#    tr_id = '.'.join([stats[key].replace('-', '') for key in \
#                      keywords if isinstance(stats[key], str)])
#    if not ret_id_dict:
#        return tr_id
#    for key in keywords:
#        if stats[key] != []:
#            id_dict[key] = stats[key]
#        else:
#            id_dict[key] = ''
#    return tr_id, id_dict
#
#
#def dv_panel_creation(dv_list, save_dir='./save'):
#    """ Create :class:`pandas.Panel` objects recombining corr and dv curves
#    This function creates two :py:class:`pandas.Panel` object recombining
#    respectively all correlation `curve` and `dv/v` curve created for
#    a specific dataset and one :py:class:`pandas.DataFrame` containing all
#    the metainformation in each `stats` object associated with one stations
#    pair.
#    The function accept a list of dv dictionaries that could be created, e.g.,
#    with the function
#    :py:func:`~miic.core.stretch_mod.velocity_change_estimate` or with
#    :py:func:`~miic.core.stretch_mod.multi_ref_vchange_and_align`
#    that are checked for a consistent structure through
#    :py:func:`~miic.core.miic_utils.dv_check`.
#    Patterns that are processed without errors are logged in a file
#    ``success_patterns.npy`` saved in the `save_dir` directory.
#    :type dv_list: list of dv dictionaries
#    :param dv_list: List of dv dictionaries
#    :type save_dir: string
#    :param save_dir: Where the two Panels and the one DataFrame will be saved
#    :rtype: dict
#    :return: **panel_dict**: dictionary that has this structure:
#        *dv*: Velocity change panel
#        *corr*: Correlation panel
#        *stats*: Meta-information stored in a DataFrame
#    """
#    from pandas import DataFrame, Series, Panel
#    stats_df = DataFrame(columns=['evel',
#                                  'endtime',
#                                  'dist',
#                                  'network',
#                                  'channel',
#                                  'stlo',
#                                  'baz',
#                                  'stel',
#                                  'evla',
#                                  'npts',
#                                  'station',
#                                  'location',
#                                  'starttime',
#                                  'sampling_rate',
#                                  'stla',
#                                  'az',
#                                  'evlo'])
#
#    base_patterns = []
#
#    for (i, dv_dict) in enumerate(dv_list):
#
#        check_state = dv_check(dv_dict)
#
#        # Check if the dv dictionary is "correct"
#        if check_state['is_incomplete']:
#            log.error("Incomplete dv. Possible errors:")
#            for key in check_state:
#                if key is not 'is_incomplete':
#                    log.error("%s: %s", key, check_state[key])
#            continue
#
#        # Extract the stats obj and use it to generate the pattern
#        stats = flatten_recarray(dv_dict['stats'])
#        pattern = _id_from_stats(stats)
#
#        # Add the current stats obj to the stats dataframe
#        cdf = DataFrame(stats, index=[pattern])
#        stats_df = stats_df.append(cdf, verify_integrity=True)
#
#        # Generate the time index
#        time = convert_time(dv_dict['time'])
#
#        if time is None:
#            log.error("Time format error: skip pattern %s", pattern)
#            continue
#
#        # Extract the corr and dv arrays
#        corr = np.squeeze(dv_dict['corr'])
#        try:
#            dv = np.squeeze(dv_dict['value'])  # New style dictionaries
#        except KeyError:  # Try the old style format
#            try:
#                dv = np.squeeze(dv_dict['dv'])
#            except KeyError:
#                try:
#                    dv = np.squeeze(dv_dict['dt'])
#                except Exception:
#                    raise Exception
#
#        # Adapt dv and corr in case they are single curves
#        try:
#            _, _ = dv.shape
#        except ValueError:
#            dv = dv[np.newaxis, :]
#            corr = corr[np.newaxis, :]
#
#        if i == 0:
#            xyz = {}
#            pqr = {}
#            win_lab = []
#            for k in range(len(dv)):
#                lab = 'win-%i' % k
#                win_lab.append(lab)
#                xyz[lab] = {}
#                pqr[lab] = {}
#
#        for j in range(len(dv)):
#            cc = Series(corr[j], index=time)
#            # ss = Series(dv[j] - 1, index=time)
#            ss = Series(dv[j], index=time)
#            xyz[win_lab[j]].update({pattern: ss})
#            pqr[win_lab[j]].update({pattern: cc})
#
#        base_patterns.append(pattern)
#
#    dv = Panel(xyz)
#    corr = Panel(pqr)
#
#    dv.save(os.path.join(save_dir, 'dv.pickle'))
#    corr.save(os.path.join(save_dir, 'corr.pickle'))
#    if len(stats_df) > 0:
#        stats_df.save(os.path.join(save_dir, 'stats.pickle'))
#
#    np.save(os.path.join(save_dir, 'success_patterns.npy'), \
#        base_patterns)
#
#    panel_dict = {}
#    panel_dict['dv'] = dv
#    panel_dict['corr'] = corr
#    if len(stats_df) > 0:
#        panel_dict['stats'] = stats_df
#
#    return panel_dict

