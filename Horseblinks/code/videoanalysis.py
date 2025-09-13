# -*- coding: utf-8 -*-
__author__ = "Stefani Dimitrova & James Strong"
__email__ = "std31@aber.ac.uk"
__status__ = "Final version"

"""
Video analysis for the horseblink project. Contains all methods for video analysis
"""

import cv2
from sys import exit
import config
import numpy as np
import eyetracker
import preprocess
from pathlib import Path
import json


def getVidFrameRate(vid):
    """returns FPS for video - useful for fps-based temporal tresholds"""
    return vid.get(cv2.CAP_PROP_FPS)

def getVidFrameCount(vid):
    """get video frame count"""
    return int(vid.get(cv2.CAP_PROP_FRAME_COUNT))

def _roi_cache_path(video_id: str) -> Path:
    """path builder - returns the absolute path to a saved roi by adding .json to the videofile name"""
    return (config.roi_cache_dir / f"{video_id}.json").resolve()


# loads cached roi from the json file, 
def _load_cached_roi(video_id: str, frame_w: int, frame_h: int):
    """ loads roi from the json file if roi is found """
    try:
        # looks for the path where roi should be found
        p = _roi_cache_path(video_id)
        # if path not found then return none
        if not p.exists():
            return None
        with open(p, "r") as f:
            # turns the json into dict
            data = json.load(f)
        x, y, w, h = map(int, data["roi"])
        return (x, y, w, h)
    except Exception:
        return None


def _save_cached_roi(video_id: str, roi, frame_w: int, frame_h: int):
    """ writes or rewrites the roi - writes the ROI box and the frame size to the json file"""
    try:
        # config.roi_cache_dir.mkdir(parents=True, exist_ok=True) - only comment out if running on a new pc
        with open(_roi_cache_path(video_id), "w") as f:
            json.dump({"roi": list(map(int, roi)), "frame_wh": [int(frame_w), int(frame_h)]}, f)
        print(f"Saved ROI cache: {_roi_cache_path(video_id).name}")
    except Exception as e:
        print(f"Could not save ROI cache: {e}")
        

def extractVideoData(vid, framecount, video_id: str = None):
    """
    Used to stream video, track, compute and return metrics for each frame
    Uses the cached roi if one is available and if not, prompts roi selection
    """

    roibrightness = []
    eyebrightness = []
    eyesaturation = []
    roisaturation = []
    averageeyehue = []

    # used to keep track of what frame we're on
    framenum = 0
    # how many frames we successfully track
    foundframes = 0
    # saves last successfull roi box, it is displayed in red when the tracker fails for observational purposes
    lastfoundroi = None
    # how long the tracker failed for (if more than 5 frames - reinitializes)
    losttracking = 0
    tracker = None
    # roi is cached once per run
    saved_cache_once = False
    def _to_gray(img):
        return img if img.ndim == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


    while vid.isOpened():
        framenum += 1
        ret, frame = vid.read()
        if not ret:
            if framenum < framecount:
                print("not able to open frame", framenum)
                continue
            else:
                print("end of video")
                break

        #  pre-processing
        if config.frameresize:
            frame = preprocess.frameresize(frame)
        if config.frameblur:
            frame = preprocess.frameblur(frame)

        fh, fw = frame.shape[:2]

        # One-time tracker init
        if tracker is None:
            cached = None
            if getattr(config, "use_cached_roi", False) and video_id:
                cached = _load_cached_roi(video_id, fw, fh)
            if cached is not None:
                tracker = cv2.TrackerKCF_create()
                tracker.init(frame, tuple(cached))
                lastfoundroi = cached
                print(f"Initialized KCF from cached ROI {tuple(cached)} for '{video_id}'")
            else:
                print("tracking started, press q to quit")
                tracker = eyetracker.initializeEyeTracker(frame)  # creates KCF after selectROI
            if getattr(config, "use_cached_roi", False) and video_id and (not saved_cache_once):
                ok0, roi0 = tracker.update(frame)
                if ok0:
                    _save_cached_roi(video_id, roi0, fw, fh)
                    lastfoundroi = roi0
                    saved_cache_once = True
                    print(f"Cached ROI immediately on init for '{video_id}': {tuple(map(int, roi0))}")
                else:
                    print("Initial tracker.update() failed on init frame; ROI not cached yet.")

        # Track this frame
        ok, roi = tracker.update(frame)

        if ok:
            losttracking = 0
            foundframes += 1
            lastfoundroi = roi

            # Save ROI to cache once 
            if (not saved_cache_once) and getattr(config, "use_cached_roi", False) and video_id:
                _save_cached_roi(video_id, roi, fw, fh)
                saved_cache_once = True

            x, y, w, h = map(int, roi)
            roiframe = frame[y:y+h, x:x+w, :]
            height, width = roiframe.shape[:2]

            # Inner eye crop -  remove a border percentage
            border_height = int(height * config.border_size_percent / 100 / 2)
            border_width  = int(width  * config.border_size_percent / 100 / 2)
            eyeframe = roiframe[
                border_height : height - border_height,
                border_width  : width  - border_width
            ]

            # Metrics
            gray_eye = cv2.cvtColor(eyeframe, cv2.COLOR_BGR2GRAY)
            eyebrightness.append(float(np.mean(gray_eye)))

            gray_roi = cv2.cvtColor(roiframe, cv2.COLOR_BGR2GRAY)
            roibrightness.append(float(np.mean(gray_roi)))

            eye_hsv = cv2.cvtColor(eyeframe, cv2.COLOR_BGR2HSV)
            eyesaturation.append(float(np.mean(eye_hsv[:, :, 1])))
            averageeyehue.append(float(np.mean(eye_hsv[:, :, 0])))

            roi_hsv = cv2.cvtColor(roiframe, cv2.COLOR_BGR2HSV)
            roisaturation.append(float(np.mean(roi_hsv[:, :, 1])))

            # Optional display
            if config.show_video:
                frame_disp = frame.copy()
                cv2.rectangle(frame_disp, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.namedWindow("eyeFrame", cv2.WINDOW_AUTOSIZE)
                
                gray_eye = cv2.cvtColor(eyeframe, cv2.COLOR_BGR2GRAY)
                heatmap_eye = cv2.applyColorMap(gray_eye, cv2.COLORMAP_JET)
                
                cv2.imshow('eyeFrame', heatmap_eye)  # for brightness purposes
                cv2.moveWindow("eyeFrame", 600, 600)

                cv2.imshow('Frame', frame_disp)
                cv2.moveWindow("Frame", 200, 200)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("process exited by user early")
                exit()


        else:
            print("lost tracking")
            losttracking += 1
            if lastfoundroi is not None and config.show_video:
                x, y, w, h = map(int, lastfoundroi)
                frame_disp = frame.copy()
                cv2.rectangle(frame_disp, (x, y), (x + w, y + h), (0, 0, 255), 2)
                cv2.imshow('Frame', frame_disp)
                cv2.moveWindow("Frame", 200, 200)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    print("process exited by user early")
                    exit()

            # Re-init tracker if lost too long
            if losttracking > config.reinitTrackingThreshold:
                print("eye has been lost, please select ROI again")
                tracker = eyetracker.initializeEyeTracker(frame)
                losttracking = 0

    vid.release()
    cv2.destroyAllWindows()

    print("Tracking completed")
    if framecount > 0:
        print("Percentage of frames where eye was tracked", foundframes / framecount * 100)

    # We do not keep frames; return empty list for compatibility
    return eyebrightness, roibrightness, eyesaturation, roisaturation, averageeyehue

def saveStrongestEventFrames(video_path, strongest_peaks, strongest_troughs, outdir=None):
    """
    Used to save frames corresponding to highest peaks and lowest troughs

    variables:
        video_path: path to the source video.
        strongest_peaks : frame indices for strongest peaks.
        strongest_troughs : frame indices for strongest troughs.
    """
    from pathlib import Path

    video_path = Path(video_path)
    video_name = video_path.stem

    outdir = Path(config.strongest_frames_dir) / video_name if outdir is None else Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    peak_set = set(int(i) for i in strongest_peaks) if strongest_peaks else set()
    trough_set = set(int(i) for i in strongest_troughs) if strongest_troughs else set()
    wanted = sorted(peak_set | trough_set)

    saved = {'peak': [], 'trough': []}

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"Could not open video for saving strongest frames: {video_path}")
        return saved

    def _save_one(idx: int):
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = cap.read()
        if not ok:
            print(f"Could not seek/read frame {idx}")
            return None, None

        # Decide label (peak wins if in both)
        if idx in peak_set:
            label = 'peak'
        elif idx in trough_set:
            label = 'trough'
        else:
            print(f"Frame {idx} not in peaks or troughs — skipping.")
            return None, None

        outpath = outdir / f"{label}_{idx:06d}.png"
        cv2.imwrite(str(outpath), frame)  # save full frame
        return label, str(outpath)

    for idx in wanted:
        label, path_str = _save_one(idx)
        if label and path_str:
            saved[label].append(path_str)

    cap.release()
    return saved
