# -*- coding: utf-8 -*-
__author__ = "Stefani Dimitrova & James Strong"
__version__ = "1"
__email__ = "std31@aber.ac.uk"
__status__ = "Final version"

import config
from cv2 import resize, GaussianBlur

def frameresize(frame):
    """
    resize frame based on config settings

    Parameters
    ----------
    frame : TYPE
        Frame to be resized.

    Returns
    -------
    frame : TYPE
        resized frame

    """
    
    if config.framesizeratio:
        height = int(frame.shape[0] * config.resizeratio)
        width = int(frame.shape[1] * config.resizeratio)
        newdim = (width,height)
        frame = resize(frame, newdim )
    else:    
        frame = resize(frame, (config.newframewidth, config.newframeheight))
    
    return frame   


def frameblur(frame):
    """
    add blur to frame, may help with removing noise in terms of brightness spots

    Parameters
    ----------
    frame : TYPE
        frame to be blurred.

    Returns
    -------
    frame : TYPE
        blurred frame.

    """
        
    frame = GaussianBlur(frame, (config.blurkernsize, config.blurkernsize), 0)
    
    return frame