# -*- coding: utf-8 -*-
__author__ = "Stefani Dimitrova & James Strong"
__email__ = "std31@aber.ac.uk"
__status__ = "Final version"

"""
Main for the horseblink project
"""

import sys
import math
import gc
from pathlib import Path

import numpy as np
import cv2

import getdata
import videoanalysis as va
import visualisation2 as vis
import postprocess as post
import config


# everything that would print to console is saved in file
if config.save:
    try:
        sys.stdout = open(config.savedir / config.savefoldername / 'output.txt', 'w')
    except:
        print("error placing console output into file")
        
def recompute_trough_indices(c_starts, c_ends, signal):
    out = []
    L = len(signal)
    for s, e in zip(c_starts, c_ends):
        s0, e0 = max(0, int(s)), min(int(e), L - 1)
        if e0 >= s0:
            seg = signal[s0:e0+1]
            off = int(np.argmin(seg))
            out.append(s0 + off)
        else:
            out.append((s0 + e0)//2)
    return out

def safe_mean(vals):
    """Return mean of values or '' if empty/invalid."""
    if vals and all(math.isfinite(v) for v in vals):
        return sum(vals) / len(vals)
    return ""

def fmt(x):
    if isinstance(x, str) or x is None:
        return ""
    return f"{x:.3f}" if math.isfinite(x) else ""


# Defaults for combined method if not present in config
if not hasattr(config, "combine_window_frames"):
    setattr(config, "combine_window_frames", 20)
if not hasattr(config, "combine_bin_unpaired"):
    setattr(config, "combine_bin_unpaired", True)

path = getdata.getHorseVideoPath()
video_id = Path(path).stem

gt = getdata.getgroundTruthData(video_id)
vid = getdata.getHorseVideoFile(path)
fps = vid.get(cv2.CAP_PROP_FPS)
vidfcount = va.getVidFrameCount(vid)

eyebrightness, roibrightness, eyesaturation, roisaturation, eyehue = va.extractVideoData(
    vid, vidfcount, video_id=video_id
)

(peakstarts, peakends), (troughstarts, troughends), strongest_peaks, strongest_troughs, blink_vector, brisignal_centered = vis.plotsignals(
    eyebrightness, roibrightness, eyesaturation, roisaturation, eyehue, fps, gt
)

# METRICS

# Peaks
(tp_p, fp_p, fn_p, full_tp_p, half_tp_p,
 full_bri_p, half_bri_p,
 full_peak_indices, half_peak_indices,
 avg_full_duration_p, avg_half_duration_p) = post.compare_pred_blinks_to_GT_with_classes(
    (peakstarts, peakends), gt, vidfcount, brisignal_centered, strongest_peaks
)
mean_full_p = safe_mean(full_bri_p)
mean_half_p = safe_mean(half_bri_p)

# Troughs
(tp_t, fp_t, fn_t, full_tp_t, half_tp_t,
 full_bri_t, half_bri_t,
 full_trough_indices, half_trough_indices,
 avg_full_duration_t, avg_half_duration_t) = post.compare_pred_blinks_to_GT_with_classes(
    (troughstarts, troughends), gt, vidfcount, brisignal_centered, strongest_troughs
)
mean_full_t = safe_mean(full_bri_t)
mean_half_t = safe_mean(half_bri_t)

# Combined, then recompute trough-strongest on residual
cw_starts, cw_ends, cw_str_peaks, cw_str_troughs = post.combine_peak_trough_within_window(
    peakstarts, peakends, troughstarts, troughends,
    window_frames=getattr(config, "combine_window_frames", 20),
    bin_unpaired=getattr(config, "combine_bin_unpaired", True)
)
cw_str_troughs = recompute_trough_indices(cw_starts, cw_ends, brisignal_centered)

(cw_tp, cw_fp, cw_fn, cw_full_tp, cw_half_tp,
 cw_full_bri, cw_half_bri,
 cw_idx_f, cw_idx_h,
 cw_avg_full_dur, cw_avg_half_dur) = post.compare_pred_blinks_to_GT_with_classes(
    (cw_starts, cw_ends), gt, vidfcount, brisignal_centered, cw_str_troughs
)
mean_full_cw = safe_mean(cw_full_bri)
mean_half_cw = safe_mean(cw_half_bri)

# Tracking percentage 
eb = np.asarray(eyebrightness, dtype=float)
tracked_pct = float(np.isfinite(eb).mean() * 100.0) if eb.size else 0.0


print(f"\n=== {video_id} ===")
print(f"Percentage of frames where eye was tracked {tracked_pct:.2f}%")

print("\n===== Peaks only =====")
print(f"TP: {tp_p}, FP: {fp_p}, FN: {fn_p}")
print(f"Full blinks: {full_tp_p}, Half blinks: {half_tp_p}")
print(f"Mean Full Brightness (peak strongest): {fmt(mean_full_p)}")
print(f"Mean Half Brightness (peak strongest): {fmt(mean_half_p)}")

print("\n===== Troughs only =====")
print(f"TP: {tp_t}, FP: {fp_t}, FN: {fn_t}")
print(f"Full blinks: {full_tp_t}, Half blinks: {half_tp_t}")
print(f"Mean Full Brightness (trough strongest): {fmt(mean_full_t)}")
print(f"Mean Half Brightness (trough strongest): {fmt(mean_half_t)}")

print("\n===== Combined (within-window) =====")
print(f"TP: {cw_tp}, FP: {cw_fp}, FN: {cw_fn}")
print(f"Full blinks: {cw_full_tp}, Half blinks: {cw_half_tp}")
print(f"Mean Full Brightness (trough strongest): {fmt(mean_full_cw)}")
print(f"Mean Half Brightness (trough strongest): {fmt(mean_half_cw)}")

cv2.destroyAllWindows()
gc.collect()

if config.save:
    sys.stdout = sys.__stdout__
