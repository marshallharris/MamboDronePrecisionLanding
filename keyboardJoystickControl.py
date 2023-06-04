#!/usr/bin/env python

import kbhit
import os
from pyparrot.Minidrone import Mambo
import cv2
import edgeDetection
import threading
from datetime import datetime 


def sessionName():
    now = datetime.now()
    return now.strftime("%Y-%m-%d-%H%M%S")

sessionPath = os.path.join("sessions",sessionName())
contourFrameToShow = None

def keyboardControl():
    global mambo
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
                pass
    kb.set_normal_term()

t1 = threading.Thread(target=keyboardControl)


def download_images_and_take_image(mambo):
    global contourFrameToShow
    if mambo.sensors.picture_state == "ready":
        picture_names = mambo.groundcam.get_groundcam_pictures_names() #get list of availible files
        print(f"There are {len(picture_names)} pictures")
        for picture in picture_names:
            frame = mambo.groundcam.get_groundcam_picture(picture,True)
            if frame is not None and frame is not False:
                print(f"downloading picture {picture}")
                success = cv2.imwrite(os.path.join(sessionPath, picture), frame)
                print(f"success? {success}")
                mambo.groundcam._delete_file(picture)
                frameContours = frame.copy()
                edgeDetection.getContoursOfImage(frame, frameContours)
                cv2.imwrite(os.path.join(sessionPath, "contours_" + picture), frameContours)
                contourFrameToShow = frameContours
        mambo.take_picture()
        mambo.smart_sleep(0.2)



if __name__ == "__main__":
    print('Hit any key, or ESC to exit')
    os.mkdir(sessionPath)
    mambo = Mambo(None, use_wifi=True) 
    print("trying to connect to mambo now")
    success = mambo.connect(num_retries=3)
    print("connected: %s" % success)

    

    mambo.ask_for_state_update()
    mambo.smart_sleep(1)
    mambo.set_user_sensor_callback(download_images_and_take_image, mambo)

    t1.start()

    while True:
        if contourFrameToShow is not None:
            cv2.imshow("Contour", contourFrameToShow)
            cv2.waitKey(1)

    cv2.destroyAllWindows()