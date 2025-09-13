# -*- coding: utf-8 -*-
__author__ = "Stefani Dimitrova & James Strong"
__version__ = "1"
__email__ = "std31@aber.ac.uk"
__status__ = "Final version"
"""
22/09/23

Save blinks, config, roi, results

"""
import config
import os
import cv2

def saveblinkimages():
    
    pass
    
def savedata(frames, events):
    
    savepath = config.savedir / config.savefoldername
    # record copy of all variables in config.
    
    configdata = config.getallvar()
    
    try:
        config_file = open(savepath / config.configfilename, 'wt')
        config_file.write(str(configdata))
        config_file.close
        
    except:
        print("Error writing config file to save path")
        
        
        
    #save middle of events
    blinkfolder = savepath / "blinkImages"
    
    if not os.path.exists(blinkfolder):
        try:
            os.mkdir(blinkfolder)
    
        except OSError as error:
            print(error)
    
    
    for i in range(len(events[0])):
        
        mid = int((events[0][i] + events[1][i]) / 2)
        filename = str(mid) + ".jpg"
        savepath = str(blinkfolder / filename)
        print(savepath)        
        cv2.imwrite(savepath, frames[mid+1])
    
    print("save completed")
