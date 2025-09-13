# -*- coding: utf-8 -*-
__author__ = "Stefani Dimitrova & James Strong"
__email__ = "std31@aber.ac.uk"
__status__ = "Final version"
"""
Postprocessing for frames: detect brightness peaks and troughs + a combined peak→trough approach.
"""

import numpy as np

def findevents(data, threshold, fps):
    """
    detects peaks and troughs from the signal through +- threshold
    then applies per-FPS min duration (to) and min inter-event gap (tm)
    Returns:
        (troughstarts, troughends, peakstarts, peakends, strongest_troughs, strongest_peaks)
    intervals are inclsive [start, end].
    """

    # FPS-based timing rules (frames)
    if fps > 50:          # 60 fps
        to, tm = 6, 14
    elif 25 <= fps <= 50: # 30 fps
        to, tm = 2, 10
    else:                 # <25 fps
        to, tm = 2, 6

    troughstarts, troughends = [], []
    peakstarts, peakends = [], []
    inpeak = False
    intrough = False

    for i in range(1, len(data) - 1):
        if not intrough and not inpeak:
            # enter a trough (downward crossing of -T, -T/2)
            if data[i] < -threshold:
                slicedata = data[1:i]
                for j in range(len(slicedata) - 1, 0, -1):
                    if slicedata[j] >= -threshold / 2:
                        troughstarts.append(j + 1)
                        intrough = True
                        break
            # enter a peak (upward crossing of +T, +T/2)
            elif data[i] > threshold:
                slicedata = data[1:i]
                for j in range(len(slicedata) - 1, 0, -1):
                    if slicedata[j] <= threshold / 2:
                        peakstarts.append(j + 1)
                        inpeak = True
                        break
        elif intrough:
            # exit trough (up through -T/2)
            if data[i] >= -threshold / 2:
                intrough = False
                troughends.append(i)
        elif inpeak:
            # exit peak (down through +T/2)
            if data[i] <= threshold / 2:
                inpeak = False
                peakends.append(i)

    if inpeak:
        peakends.append(len(data) - 1)
    if intrough:
        troughends.append(len(data) - 1)

    def _filter_min_duration(starts, ends, min_len):
        fs, fe = [], []
        for s, e in zip(starts, ends):
            if (e - s + 1) >= min_len:
                fs.append(s); fe.append(e)
        return fs, fe

    def _enforce_min_gap(starts, ends, mingap):
        if not starts:
            return [], []
        fs = [starts[0]]; fe = [ends[0]]
        for s, e in zip(starts[1:], ends[1:]):
            if s - fe[-1] >= mingap:
                fs.append(s); fe.append(e)
        return fs, fe

    # applying tresholds
    peakstarts,   peakends   = _filter_min_duration(peakstarts,   peakends,   to)
    troughstarts, troughends = _filter_min_duration(troughstarts, troughends, to)

    peakstarts,   peakends   = _enforce_min_gap(peakstarts,   peakends,   tm)
    troughstarts, troughends = _enforce_min_gap(troughstarts, troughends, tm)

    # strongest point per interval
    def _strongest_indices(starts, ends, arr, mode="max"):
        out = []
        for s, e in zip(starts, ends):
            segment = arr[s:e + 1]
            if not segment:
                continue
            if mode == "max":
                v = max(segment)
            else:
                v = min(segment)
            out.append(s + segment.index(v))
        return out

    strongest_peaks   = _strongest_indices(peakstarts,   peakends,   data, mode="max")
    strongest_troughs = _strongest_indices(troughstarts, troughends, data, mode="min")

    return (
        troughstarts, troughends,
        peakstarts, peakends,
        strongest_troughs, strongest_peaks
    )

def _overlap_inclusive(ps, pe, gs, ge):
    """Return overlap length (frames) for inclusive intervals [ps,pe] and [gs,ge]."""
    return max(0, min(pe, ge) - max(ps, gs) + 1)

def prediction_overlaps_event(pred_start, pred_end, gt_start, gt_end):
    """True if inclusive intervals overlap by at least 1 frame."""
    return _overlap_inclusive(pred_start, pred_end, gt_start, gt_end) > 0


def compare_pred_blinks_to_GT_with_classes(data, gt, total_frames, brisignal_centered=None, strongest_indices=None):
    """
    Compare predicted intervals to GT and return:
      (TP, FP, FN,
       full_TP, half_TP,
       full_intensities, half_intensities,
       full_peak_indices, half_peak_indices,
       avg_full_duration, avg_half_duration)
      
    """
    # Build aligned GT lists using a single mask for starts/ends/classes
    if 'Eye closing Frame' in gt.columns and 'Eye Fully Open' in gt.columns:
        mask = gt['Eye closing Frame'].notna() & gt['Eye Fully Open'].notna()
        gtstart = gt.loc[mask, 'Eye closing Frame'].astype(int).to_list()
        gtend   = gt.loc[mask, 'Eye Fully Open'].astype(int).to_list()
        gtclass = gt.loc[mask, 'Class'].fillna('').astype(str).to_list()
    elif 'Eye closing Time' in gt.columns and 'Eye Fully Open Time' in gt.columns:
        mask = gt['Eye closing Time'].notna() & gt['Eye Fully Open Time'].notna()
        gtstart = gt.loc[mask, 'Eye closing Time'].astype(int).to_list()
        gtend   = gt.loc[mask, 'Eye Fully Open Time'].astype(int).to_list()
        gtclass = gt.loc[mask, 'Class'].fillna('').astype(str).to_list()
    else:
        return 0, 0, 0, 0, 0, [], [], [], [], None, None

    tpos = fpos = fneg = 0
    matched_gt = set()
    full_tp = half_tp = 0

    full_intensities, half_intensities = [], []
    full_peak_indices, half_peak_indices = [], []
    full_durations, half_durations = [], []

    start, end = data[0], data[1]

    for i in range(len(start)):
        matched = False
        for j in range(len(gtstart)):
            if j in matched_gt:
                continue
            if prediction_overlaps_event(start[i], end[i], gtstart[j], gtend[j]):
                tpos += 1
                matched = True
                matched_gt.add(j)

                # intensity sample at strongest point
                # intensity sample at strongest trough index
                if brisignal_centered is not None and strongest_indices is not None and i < len(strongest_indices):
                    idx = int(strongest_indices[i])  # always trust the trough index
                    if 0 <= idx < len(brisignal_centered):
                        val = brisignal_centered[idx]
                        if gtclass[j].upper() == 'F':
                            full_intensities.append(val); full_peak_indices.append(idx)
                        elif gtclass[j].upper() == 'H':
                            half_intensities.append(val); half_peak_indices.append(idx)


                # duration bucket (inclusive)
                dur = end[i] - start[i] + 1
                if gtclass[j].upper() == 'F':
                    full_tp += 1; full_durations.append(dur)
                elif gtclass[j].upper() == 'H':
                    half_tp += 1; half_durations.append(dur)
                break

        if not matched:
            fpos += 1

    fneg = len(gtstart) - tpos

    avg_full_duration = (sum(full_durations) / len(full_durations)) if full_durations else None
    avg_half_duration = (sum(half_durations) / len(half_durations)) if half_durations else None

    return (
        tpos, fpos, fneg,
        full_tp, half_tp,
        full_intensities, half_intensities,
        full_peak_indices, half_peak_indices,
        avg_full_duration, avg_half_duration
    )

def combine_peak_trough_within_window(
    peakstarts, peakends,
    troughstarts, troughends,
    strongest_peaks=None,        
    brisignal_centered=None,     # signal to recompute trough strongest index
    window_frames: int = 15,
    bin_unpaired: bool = True,
):
    """
    Pair peaks and troughs within window_frames.
    - Always recomputes strongest trough index from `brisignal_centered` for each combined interval.
    """

    npk, ntr = len(peakstarts), len(troughstarts)

    carry_peaks = strongest_peaks is not None

    i, j = 0, 0
    c_starts, c_ends = [], []
    c_str_peaks   = [] if carry_peaks else None
    c_str_troughs = []

    def gap_between(pi, ti):
        """Non-negative gap between peak and trough; 0 if they overlap."""
        ps, pe = peakstarts[pi], peakends[pi]
        ts, te = troughstarts[ti], troughends[ti]
        if pe >= ts and te >= ps:
            return 0
        if pe < ts:  # peak ends before trough starts
            return ts - pe
        return ps - te  # trough ends before peak starts

    def recompute_trough_index(start, end, signal):
        if signal is None:
            return None
        s0, e0 = max(0, int(start)), min(int(end), len(signal) - 1)
        if e0 >= s0:
            seg = signal[s0:e0+1]
            if len(seg) == 0:
                return None
            off = int(np.argmin(seg))
            return s0 + off
        return None

    while i < npk and j < ntr:
        g = gap_between(i, j)

        if g <= window_frames:
            # Pair current peak i with trough j
            ps, pe = peakstarts[i], peakends[i]
            ts, te = troughstarts[j], troughends[j]

            c_starts.append(min(ps, ts))
            c_ends.append(max(pe, te))

            if carry_peaks:   
                c_str_peaks.append(strongest_peaks[i])
            c_str_troughs.append(recompute_trough_index(min(ps, ts), max(pe, te), brisignal_centered))

            i += 1
            j += 1

        else:
            # Too far apart: advance the one that ends first
            ps, pe = peakstarts[i], peakends[i]
            ts, te = troughstarts[j], troughends[j]
            peak_ends_first = (pe < ts)

            if bin_unpaired:
                if peak_ends_first:
                    i += 1
                else:
                    j += 1
            else:
                if peak_ends_first:
                    c_starts.append(ps)
                    c_ends.append(pe)
                    if carry_peaks: c_str_peaks.append(strongest_peaks[i])
                    c_str_troughs.append(recompute_trough_index(ps, pe, brisignal_centered))
                    i += 1
                else:
                    c_starts.append(ts)
                    c_ends.append(te)
                    if carry_peaks: c_str_peaks.append(None)
                    c_str_troughs.append(recompute_trough_index(ts, te, brisignal_centered))
                    j += 1

    if not bin_unpaired:
        while i < npk:
            ps, pe = peakstarts[i], peakends[i]
            c_starts.append(ps); c_ends.append(pe)
            if carry_peaks: c_str_peaks.append(strongest_peaks[i])
            c_str_troughs.append(recompute_trough_index(ps, pe, brisignal_centered))
            i += 1
        while j < ntr:
            ts, te = troughstarts[j], troughends[j]
            c_starts.append(ts); c_ends.append(te)
            if carry_peaks: c_str_peaks.append(None)
            c_str_troughs.append(recompute_trough_index(ts, te, brisignal_centered))
            j += 1

    return c_starts, c_ends, c_str_peaks, c_str_troughs
