# -*- coding: utf-8 -*-
__author__ = "Stefani Dimitrova & James Strong"
__version__ = "1"
__email__ = "std31@aber.ac.uk"
__status__ = "Final version"
"""
08/09/23

Visualisation for Horseblinks (Brightness Peak Events Only)
"""
import os
os.environ.pop("MPLBACKEND", None)

import matplotlib
import matplotlib.pyplot as plt
matplotlib.use("TkAgg", force=True)   
from scipy.signal import savgol_filter
import numpy as np
import config
import postprocess as post


 # figure not showing, probably python settings
def _bring_to_front(fig, title=None):
    m = fig.canvas.manager

    if title and hasattr(m, "set_window_title"):
        m.set_window_title(title)

    w = getattr(m, "window", None)
    if w is None:
        return

    if title:
        if hasattr(w, "wm_title"):          
            w.wm_title(title)
        elif hasattr(w, "setWindowTitle"):  
            w.setWindowTitle(title)

    if hasattr(w, "raise_"):                
        w.raise_()
    elif hasattr(w, "lift"):                
        w.lift()

    if hasattr(w, "activateWindow"):        
        w.activateWindow()

 # used for plotting Figure 2
def _crossings(signal, starts, ends, level, mode):
    """Return fractional-frame start/end crossings for each [start,end] window."""
    N = len(signal)
    xs, xe = [], []
    up = (mode == 'peak')  

    def interp(i):
        y0, y1 = signal[i], signal[i + 1]
        dy = y1 - y0
        return float(i) if dy == 0 else float(i) + (level - y0) / dy

    for s, e in zip(starts, ends):
        a = max(0, s - 1)         
        b = min(N - 2, e - 1)     

        x_start = float(s)
        for i in range(a, b + 1):
            y0, y1 = signal[i], signal[i + 1]
            if (up and (y0 < level <= y1)) or (not up and (y0 > level >= y1)):
                x_start = interp(i)
                break

        x_end = float(e)
        for i in range(b, a - 1, -1):
            y0, y1 = signal[i], signal[i + 1]
            if (up and (y0 >= level > y1)) or (not up and (y0 <= level < y1)):
                x_end = interp(i)
                break

        xs.append(x_start); xe.append(x_end)

    return xs, xe


def plotsignals(
    eyebrightness, roibrightness, eyesaturation, roisaturation, eyehue, fps,
    gt=None, plot=True
):
    """Build residual brightness signal, threshold, find events, plot (with GT if given)."""

    brisnormalised = [r - e for r, e in zip(roibrightness, eyebrightness)]

    brifilter = savgol_filter(brisnormalised, config.brifilterwindow, 1)

    brisignal = [r - f for r, f in zip(brisnormalised, brifilter)]

    # center to baseline (median)
    offset = float(np.median(brisignal))
    brisignal_centered = [x - offset for x in brisignal]

    # median-based
    # median_k = getattr(config, "median_k", 6)  # expose in config; typical 1.0–3.0
    # bri_threshold = median_k * float(np.median(np.abs(brisignal_centered)))
    # t_hard = bri_threshold
    # t_soft = bri_threshold / 2.0

    # std-based 
    std_k = 1
    bri_threshold = std_k * float(np.std(brisignal_centered))
    t_hard = bri_threshold
    t_soft = bri_threshold / 2.0

    # debug readout (visible by design)
    print(f"[DBG] T={t_hard:.3f}, T/2={t_soft:.3f}, "
          f"residual range={min(brisignal_centered):.3f}..{max(brisignal_centered):.3f}")

    # detect events (peaks+troughs) on the centered signal
    (troughstarts, troughends,
     peakstarts,   peakends,
     strongest_troughs, strongest_peaks) = post.findevents(brisignal_centered, bri_threshold, fps)

    # blink vector from peaks (primary detector)
    blink_vector = [0.0] * len(brisignal_centered)
    for s, e in zip(peakstarts, peakends):
        for i in range(max(0, s), min(e + 1, len(blink_vector))):
            blink_vector[i] = 0.15

    # init GT-dependent outputs
    full_peaks, half_peaks, full_troughs, half_troughs = [], [], [], []
    gt_blink_vector = [0.0] * len(brisignal_centered)

    # classify predictions vs GT
    def _classify(starts, ends, strongest):
        (_tp,_fp,_fn,_full_tp,_half_tp,
         _full_int,_half_int,
         full_idx, half_idx,
         _avg_full_dur,_avg_half_dur) = post.compare_pred_blinks_to_GT_with_classes(
            (starts, ends), gt, len(brisignal_centered), brisignal_centered, strongest
        )
        return full_idx, half_idx

    # if GT is present, compute class markers + GT blink vector
    if gt is not None:
        # peaks vs GT (split strongest indices by GT class)
        full_peaks,  half_peaks   = _classify(peakstarts,   peakends,   strongest_peaks)
        # troughs vs GT (split strongest indices by GT class)
        full_troughs, half_troughs = _classify(troughstarts, troughends, strongest_troughs)

        # GT blink vector from Eye closing .. Eye Fully Open
        if 'Eye closing Frame' in gt.columns and 'Eye Fully Open' in gt.columns:
            gts = gt['Eye closing Frame'].dropna().astype(int).to_list()
            gte = gt['Eye Fully Open'].dropna().astype(int).to_list()
            for s, e in zip(gts, gte):
                for i in range(max(0, s), min(e + 1, len(gt_blink_vector))):
                    gt_blink_vector[i] = 0.08

    # plotting block (two figures)
    if plot:
        # x-axis in frames (only needed for plots)
        steps = list(range(len(eyebrightness)))

        # Figure 1: 6-panel summary
        fig, axs = plt.subplots(6, 1, figsize=(8.27, 13), sharex=True)
        fig.patch.set_facecolor('white')

        # (1) raw brightness
        axs[0].plot(eyebrightness, color='black', linewidth=1, label='Eye Brightness')
        axs[0].plot(roibrightness, color='gray',  linewidth=1, label='ROI Brightness')
        axs[0].set_title('1. Raw Brightness'); axs[0].legend(); axs[0].tick_params(labelbottom=False)

        # (2) normalized residual (ROI−Eye)
        axs[1].plot(brisnormalised, color='black', linewidth=1)
        axs[1].set_title('2. Normalized Brightness'); axs[1].tick_params(labelbottom=False)

        # (3) smoothed trend
        axs[2].plot(brifilter, color='black', linewidth=1)
        axs[2].set_title('3. Smoothed Trend'); axs[2].tick_params(labelbottom=False)

        # (4) centered residual + thresholds (+ optional GT markers)
        axs[3].plot(brisignal_centered, color='black', linewidth=1)
        axs[3].axhline(+t_hard, linestyle='--', color='gray', linewidth=1, label='Hard ±T')
        axs[3].axhline(-t_hard, linestyle='--', color='gray', linewidth=1)
        axs[3].axhline(+t_soft, linestyle=':',  color='gray', linewidth=1, label='Soft ±T/2')
        axs[3].axhline(-t_soft, linestyle=':',  color='gray', linewidth=1)
        axs[3].axhline(0,       linestyle='dotted', color='red', linewidth=1, label='Centered Baseline')
        axs[3].set_title(f'4. Final Signal with Threshold (T={t_hard:.2f}, T/2={t_soft:.2f})')

        # GT markers at strongest points
        if full_peaks:   axs[3].scatter(full_peaks,   [brisignal_centered[i] for i in full_peaks],   color='blue',   s=10, label='Full Blink Peak')
        if half_peaks:   axs[3].scatter(half_peaks,   [brisignal_centered[i] for i in half_peaks],   color='green',  s=10, label='Half Blink Peak')
        if full_troughs: axs[3].scatter(full_troughs, [brisignal_centered[i] for i in full_troughs], color='orange', s=10, label='Full Trough')
        if half_troughs: axs[3].scatter(half_troughs, [brisignal_centered[i] for i in half_troughs], color='purple', s=10, label='Half Trough')

        axs[3].legend(); axs[3].tick_params(labelbottom=False)

        # (5) binary blink vector (pred)
        axs[4].plot(steps, blink_vector, color='black', linewidth=1)
        if strongest_peaks:
            axs[4].scatter(strongest_peaks, [0.15] * len(strongest_peaks), color='red', s=10)
        axs[4].set_title('5. Binary Blink Prediction (Peaks)')
        axs[4].set_ylim(-0.02, 0.2); axs[4].set_yticks([0, 0.15]); axs[4].set_yticklabels(['0', '1'])
        axs[4].tick_params(labelbottom=False)

        # (6) ground truth blink vector
        axs[5].plot(steps, gt_blink_vector, color='black', linewidth=1)
        axs[5].set_title('6. Ground Truth Blink Vector')
        axs[5].set_ylim(-0.02, 0.1); axs[5].set_yticks([0, 0.08]); axs[5].set_yticklabels(['0', '1'])
        axs[5].set_xlabel('Frame')

        # layout + save
        plt.tight_layout(pad=2.0)
        # fig.savefig("blink_signals.svg", format='svg', dpi=300)
        _bring_to_front(fig, "Blink signals — 6 panels")
        plt.show(block=True)
        plt.close(fig)

        # Figure 2: Full residual + thresholds + start/end crossings
        fig_full_residual, ax = plt.subplots(figsize=(10, 4))
        ax.plot(steps, brisignal_centered, color='black', linewidth=1, label='Centered Residual Signal')

        # thresholds
        ax.axhline(+t_hard, linestyle='--', color='gray', linewidth=1, label='+T (Hard)')
        ax.axhline(-t_hard, linestyle='--', color='gray', linewidth=1, label='−T (Hard)')
        ax.axhline(+t_soft, linestyle=':',  color='gray', linewidth=1, label='+T/2 (Soft)')
        ax.axhline(-t_soft, linestyle=':',  color='gray', linewidth=1, label='−T/2 (Soft)')
        ax.axhline(0,       linestyle='dotted', color='red', linewidth=1, label='Baseline')

        # choose which threshold level to visualize crossings at
        marker_level_peaks   = +t_soft
        marker_level_troughs = -t_soft

        # fractional-frame crossings (start/end of each predicted event)
        peak_xs,   peak_xe   = _crossings(brisignal_centered, peakstarts,   peakends,   marker_level_peaks,   mode='peak')
        trough_xs, trough_xe = _crossings(brisignal_centered, troughstarts, troughends, marker_level_troughs, mode='trough')

        # peak start/end markers at +T/2
        if peak_xs:
            ax.scatter(peak_xs, [marker_level_peaks] * len(peak_xs), s=30, facecolors='none',
                       edgecolors='blue', linewidths=1.4, label='Peak start (T/2)')
        if peak_xe:
            ax.scatter(peak_xe, [marker_level_peaks] * len(peak_xe), s=30, color='blue',
                       label='Peak end (T/2)')

        # trough start/end markers at −T/2
        if trough_xs:
            ax.scatter(trough_xs, [marker_level_troughs] * len(trough_xs), s=30, facecolors='none',
                       edgecolors='orange', linewidths=1.4, label='Trough start (−T/2)')
        if trough_xe:
            ax.scatter(trough_xe, [marker_level_troughs] * len(trough_xe), s=30, color='orange',
                       label='Trough end (−T/2)')

        # strongest points overlays
        if strongest_peaks:
            ax.scatter(strongest_peaks,
                       [brisignal_centered[i] for i in strongest_peaks],
                       color='red', marker='x', s=40, linewidths=1.5, label='Strongest Peak')
            for i in strongest_peaks:
                ax.annotate(str(i), (i, brisignal_centered[i]), fontsize=8, color='red')

        if strongest_troughs:
            ax.scatter(strongest_troughs,
                       [brisignal_centered[i] for i in strongest_troughs],
                       color='purple', marker='x', s=40, linewidths=1.5, label='Strongest Trough')
            for i in strongest_troughs:
                ax.annotate(str(i), (i, brisignal_centered[i]), fontsize=8, color='purple')

        # labels
        ax.set_title('Residual Signal with Start and Stop Markers')
        ax.set_xlabel('Frame'); ax.set_ylabel('Centered Brightness Residual')
        ax.legend(loc='upper right', fontsize='small')

        fig_full_residual.tight_layout()
        # fig_full_residual.savefig("residual_signal_with_thresholds.svg", format='svg', dpi=300)
        _bring_to_front(fig_full_residual, "Residual signal — full length")
        plt.show(block=True)
        plt.close(fig_full_residual)

    return (peakstarts, peakends), (troughstarts, troughends), strongest_peaks, strongest_troughs, blink_vector, brisignal_centered
