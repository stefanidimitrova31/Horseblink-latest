# -*- coding: utf-8 -*-
__author__ = "Stefani Dimitrova & James Strong"
__version__ = "1"
__email__ = "std31@aber.ac.uk"
__status__ = "Final version"

"""
07/09/23

Tracker for tracking eyes in Horseblinks.

TODO
could benefit from some defensive programing related to not selecting a roi etc.

"""
import cv2 


def initializeEyeTracker(frame):
    """
    initalises tracker for eye tracking, more robust to allow half of window around eye and allow for this border
    when processing. see config border removal.

    """
    
                
    cv2.namedWindow("roiframe", cv2.WINDOW_AUTOSIZE)
    cv2.setWindowProperty("roiframe", cv2.WND_PROP_TOPMOST, 1)
    roi = cv2.selectROI("roiframe", frame)
    eyetracker = cv2.TrackerKCF_create()
    eyetracker.init(frame, roi)
    cv2.destroyWindow("roiframe")
    
    return eyetracker