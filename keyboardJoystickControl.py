#!/usr/bin/env python

import kbhit
import os
from pyparrot.Minidrone import Mambo
import cv2
import edgeDetection
import threading
from datetime import datetime 
import time
import cmath
import numpy as np


def sessionName():
    now = datetime.now()
    return now.strftime("%Y-%m-%d-%H%M%S")

sessionPath = os.path.join("sessions",sessionName())
contourFrameToShow = None
polarCoordsToTarget = None
autoTrackModeEnabled = False


def polarToCartesian(rho, phi):
    x = rho * np.cos(phi)
    y = rho * np.sin(phi)
    return x, y

def scaleTragectory(x, y):
    return 20.0 * x / 320.0, -20.0 * y / 240.0

def autoTrackTarget():
    global polarCoordsToTarget
    global autoTrackModeEnabled
    global mambo
    while True:
        mambo.smart_sleep(0.1)
        print(f"autoTrackTarget {polarCoordsToTarget}, {autoTrackModeEnabled}")
        if polarCoordsToTarget is not None and autoTrackModeEnabled:
            x, y = polarToCartesian(polarCoordsToTarget[0], polarCoordsToTarget[1]) # positive y is down in the image
            if (polarCoordsToTarget[0] < 75):
                print("target acquired !!!!!!!")
            else:
                roll, pitch = scaleTragectory(x, y)
                print(f"tracking target roll {roll}, pitch {pitch}")
                polarCoordsToTarget = None
                mambo.fly_direct(roll, pitch, 0, 0, 0.5)
        

def keyboardControl():
    global mambo
    global autoTrackModeEnabled
    kb = kbhit.KBHit()
    while True:
        if kb.kbhit():
            c = kb.getch()
            if ord(c) == 27: # ESC
                break

            if ord(c) == 119: #w
                mambo.fly_direct(0, 20, 0, 0)
                print("go forward")
            if ord(c) == 97: #a
                mambo.fly_direct(-20, 0, 0, 0)
                print("go left")
            if ord(c) == 115: #s
                mambo.fly_direct(0, -20, 0, 0)
                print("go backward")
            if ord(c) == 116: #t
                print("takeoff")
                mambo.safe_takeoff(5)
            if ord(c) == 100: #d
                mambo.fly_direct(20, 0, 0, 0)
                print("go right")
            if ord(c) == 108: #l
                print("land")
                result = mambo.land()
                print(f"land {result}")
            if ord(c) == 102: #f
                print("enable auto track mode")
                autoTrackModeEnabled = True
            if ord(c) == 103: #g
                print("disable auto track mode")
                autoTrackModeEnabled = False

    kb.set_normal_term()

t1 = threading.Thread(target=keyboardControl)
t2 = threading.Thread(target=autoTrackTarget)


def download_images_and_take_image(mambo):
    global contourFrameToShow
    global polarCoordsToTarget
    if mambo.sensors.picture_event == "taken":
        mambo.take_picture()
        picture_names = mambo.groundcam.get_groundcam_pictures_names() #get list of availible files
        print(f"There are {len(picture_names)} pictures")
        for picture in picture_names:
            frame = mambo.groundcam.get_groundcam_picture(picture,True)
            if frame is not None and frame is not False:
                success = cv2.imwrite(os.path.join(sessionPath, picture), frame)
                mambo.groundcam._delete_file(picture)
                beforeContours = time.time()
                frameContours = frame.copy()
                polarCoordsToTarget = edgeDetection.getContoursOfImage(frame, frameContours)
                if polarCoordsToTarget is not None:
                    print(f"polar coords to target rho:{polarCoordsToTarget[0]}, phi{np.rad2deg(polarCoordsToTarget[1])}")
                cv2.imwrite(os.path.join(sessionPath, "contours_" + picture), frameContours)
                contourFrameToShow = frameContours

if __name__ == "__main__":
    print('Hit any key, or ESC to exit')
    os.mkdir(sessionPath)
    mambo = Mambo(None, use_wifi=True) 
    print("trying to connect to mambo now")
    success = mambo.connect(num_retries=3)
    print("connected: %s" % success)

    mambo.set_user_picture_event_callback(download_images_and_take_image, mambo)
    mambo.ask_for_state_update()
    mambo.smart_sleep(1)
    mambo.take_picture()
    mambo.smart_sleep(1)

    t1.start()
    t2.start()

    while True:
        if contourFrameToShow is not None:
            cv2.imshow("Contour", contourFrameToShow)
            cv2.waitKey(1)

    cv2.destroyAllWindows()