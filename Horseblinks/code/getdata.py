# -*- coding: utf-8 -*-
__author__ = "Stefani Dimitrova & James Strong"
__version__ = "1"
__email__ = "std31@aber.ac.uk"
__status__ = "Final version"

import tkinter as tk
from tkinter import filedialog
from pathlib import Path
from cv2 import VideoCapture
from sys import exit
import pandas as pd
import config

def getHorseVideoPath():
    """
    user selects video file.
    returns the path to the selected video.
    """
    print("Select a horse video file")

    window = tk.Tk()
    window.withdraw()
    window.lift()
    window.wm_attributes('-topmost', True)

    file_path = filedialog.askopenfilename(
        parent=window,
        initialdir=config.videodir,
        title="Select Horse Video File",
        filetypes=[("Video files", "*.mp4 *.avi *.mov *.mkv")]
    )
    window.destroy()

    file_path = Path(file_path)
    print(f" Selected video path: {file_path}")
    return file_path


def getHorseVideoFile(videoFilePath):
    """
    opens video by given path.

    Parameters
    ----------
    videoFilePath : Path
        Path to the video file.

    Returns
    -------
    vid : OpenCV VideoCapture object
    """
    print(f"Trying to open video at: {videoFilePath}")
    vid = VideoCapture(str(videoFilePath))

    if not vid.isOpened():
        print(" Error: Could not open video file")
        exit()

    return vid


def getgroundTruthData(GTFileName):
    """
    Loads the ground truth CSV file that matches the given video filename stem.

    Parameters
    ----------
    GTFileName : str
        The stem of the video filename (e.g., '2022-06-041').

    Returns
    -------
    data : pandas DataFrame
        The ground truth data.
    """
    GTFileName = GTFileName + ".csv"
    gt_path = config.groundtruthdir / GTFileName
    print(f" Loading ground truth from: {gt_path}")

    if not gt_path.exists():
        print(f" Error: Ground truth file not found: {gt_path}")
        exit()

    data = pd.read_csv(gt_path)
    return data
