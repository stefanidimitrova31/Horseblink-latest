# -*- coding: utf-8 -*-
__author__ = "Stefani Dimitrova & James Strong"
__email__ = "std31@aber.ac.uk"
__status__ = "Final version"

"""
Config file for the horseblink project. Contains all variables that need to be modified and their explanations.
"""

from pathlib import Path


# OUTPUT file settings, don't change unless you want info to be saved in a different directory, currently saved in the output folder (txt with results saved in subfolder EXP00)
save = True
savefolder = 'output'
savefoldername = 'EXP00'
configfilename = 'config'

# the project's root folder
basedir = Path(__file__).parent.parent
videofolder = 'videos'
datafolder = 'groundtruth'
# full paths to the video folder, datafolders and output folder
videodir = basedir / videofolder
groundtruthdir = basedir / datafolder
savedir = basedir / savefolder

strongest_frames_dir = r"C:\Users\Stefani\Desktop\Horseblinks (1)\Horseblinks\output\strongestFrames"


# saving ROI across videos so we can test with the same ROI selection on many thresholds
# if both are set to False always requires manual selection and doesn't cache it
use_cached_roi = True          # check if there is a cached roi for this video; if true load_cached_roi in videoanalysis runs ; if false it prompts to select a roi and _save_cached_roi in videoanalysis runs
fixed_roi_mode = True          # if past roi exists and this is true - reuse that past roi, if false doesnt cache
roi_cache_dir = savedir / "roi"


#frame visualisation options
frameresize = True
framesizeratio = True  # If False, use fixed size below
resizeratio = 0.4
newframewidth = 1920
newframeheight = 1080

# show live video during tracking (memory consuming for longer vids)
show_video = False

# frame blur settings
frameblur = True
blurkernsize = 15

reinitTrackingThreshold = 5  # frames lost before reinit
border_size_percent = 49      # % of frame edge to crop for ROI

# brightness filter window sizes
brifilterwindow = 30

# Combined-events pairing window
combine_gap_frames = 20  # frames; trough must start within this many frames of the preceding peak end
