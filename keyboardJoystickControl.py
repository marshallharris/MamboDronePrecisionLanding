#!/usr/bin/env python

import kbhit
import csv
import os
from pyparrot.Minidrone import Mambo
import cv2
import edgeDetection
import threading
from datetime import datetime 
import time
import cmath
import logging
import numpy as np
from dataclasses import dataclass


def empty(a):
    pass

def sessionName():
    now = datetime.now()
    return now.strftime("%Y-%m-%d-%H%M%S")

sessionPath = os.path.join("sessions",sessionName())
contourFrameToShow = None
polarCoordsToTarget = None
autoTrackModeEnabled = False
pictureTimestampQueue = []
csvFile = None
mostRecentTarget = None


def polarToCartesian(rho, phi):
    x = rho * np.cos(phi)
    y = rho * np.sin(phi)
    return x, y


def autoTrackTargetPID():
    global mostRecentTarget
    global autoTrackModeEnabled
    global mambo
    print("PID loop running sanity check")
    x_errorPrior = 0
    y_errorPrior = 0
    x_integralPrior = 0
    y_integralPrior = 0
    timestampPrior = None
    Kp = 20.0/320.0

    Ki = 0
    Kd = 0
    bias = 0

    while True:
        mambo.smart_sleep(0.1)
        T = cv2.getTrackbarPos("T", "Command_duration") / 100
        if mostRecentTarget is not None:
            print("got most recent frame")
            if mostRecentTarget.rho < 75 and mostRecentTarget.percentOfImage() > 5:
                print("target acquired!!! Landing")    
                mambo.land()
            elif timestampPrior is not None:
                timestamp = mostRecentTarget.timestamp
                iterationTime = timestamp - timestampPrior
                x_error = mostRecentTarget.x_center - 320
                y_error = mostRecentTarget.y_center - 240
                x_integral = x_integralPrior + x_error * iterationTime
                y_integral = y_integralPrior + y_error * iterationTime
                x_derivative = (x_error - x_errorPrior) / iterationTime
                y_derivative = (y_error - y_errorPrior) / iterationTime
                
                x_output = Kp * x_error + Ki * x_integral + Kd * x_derivative + bias
                y_output = Kp * y_error + Ki * y_integral + Kd * y_derivative + bias
                
                roll = x_output
                pitch = -y_output #invert because Y is positive in down direction of image

                x_errorPrior = x_error
                y_errorPrior = y_error
                x_integralPrior = x_integral
                y_integralPrior = y_integral

                timestampPrior = timestamp
                
                if autoTrackModeEnabled:
                    command_time = 1.0
                    height_scale_factor = mambo.sensors.position_z * -1 / 150.0 # make this more intelligent by looking at size of target
                    data = [timestamp, iterationTime, Kp, Ki, Kd, x_error, x_integral, x_derivative, x_output, y_error, y_integral, y_derivative, y_output, command_time]
                    with open(os.path.join(sessionPath, "PID_Data.csv"), 'a') as f:
                        writer = csv.writer(f)
                        print("write PID data to CSV")
                        writer.writerow(data)
                    timeComponent = 100/mostRecentTarget.percentOfImage()
                    horizontalComponent = -20
                    if mostRecentTarget.percentOfImage() > 2:
                        horizontalComponent = -10
                    if mostRecentTarget.percentOfImage() > 4:
                        horizontalComponent = -5
                    if mostRecentTarget.percentOfImage() > 8 :
                        horizontalComponent = 0

                    if timeComponent > 30:
                        timeComponent = 30
                    print(f"tracking target roll {roll}, pitch {pitch} percent {mostRecentTarget.percentOfImage()}, time {timeComponent/100 * 3.0}")
                    mambo.fly_direct(roll, pitch, 0, horizontalComponent, timeComponent/100 * 3.0)
                
            else:
                print("setting timestamp prior")
                timestampPrior = mostRecentTarget.timestamp
            mostRecentTarget = None
        elif autoTrackModeEnabled:
            mambo.fly_direct(0,0,0,0, 0.1)


def log_position(mambo):
    while True:
        mambo.smart_sleep(0.1)
        data = [datetime.timestamp(datetime.now()), mambo.sensors.position_x, mambo.sensors.position_y, mambo.sensors.position_z, mambo.sensors.altitude]
        with open(os.path.join(sessionPath, "Position_Data.csv"), 'a') as f:
            writer = csv.writer(f)
            writer.writerow(data)


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
            if ord(c) == 101: #e
                mambo.fly_direct(0, 0, 0, 20)
                print("go up")
            if ord(c) == 114: #r
                mambo.fly_direct(0,0,0, -20)
                print("go down")
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

keyboardControlThread = threading.Thread(target=keyboardControl)
pidAutoTrackThread = threading.Thread(target=autoTrackTargetPID)

loggingThread = None


def download_images_and_take_image(mambo):
    global contourFrameToShow
    global polarCoordsToTarget
    global pictureTimestampQueue
    global mostRecentTarget
    if mambo.sensors.picture_event == "taken":
        logging.info("take picture")
        mambo.take_picture()
        pictureTimestampQueue.append(datetime.timestamp(datetime.now()))
        picture_names = mambo.groundcam.get_groundcam_pictures_names() #get list of availible files
        print(f"There are {len(picture_names)} pictures")
        for picture in picture_names:
            logging.info("got picture")
            frame = mambo.groundcam.get_groundcam_picture(picture,True)
            if frame is not None and frame is not False:
                success = cv2.imwrite(os.path.join(sessionPath, picture), frame)
                mambo.groundcam._delete_file(picture)
                beforeContours = time.time()
                frameContours = frame.copy()
                # 230, 245 are canny values for patterned carpet
                target = edgeDetection.getContoursOfImage(frame, frameContours, 230, 245)
                if target is not None:
                    target.timestamp = pictureTimestampQueue.pop(0)
                    mostRecentTarget = target
                    print(f"Target x: {target.x_center}, y: {target.y_center}, width: {target.width}, height: {target.height}")
                else:
                    pictureTimestampQueue.pop(0) # remove timestampe where contour can't be extracted

                cv2.imwrite(os.path.join(sessionPath, "contours_" + picture), frameContours)
                contourFrameToShow = frameContours

def clear_all_images(mambo):
    picture_names = mambo.groundcam.get_groundcam_pictures_names() #get list of availible files
    for picture in picture_names:
        mambo.groundcam._delete_file(picture)

if __name__ == "__main__":
    print('Hit any key, or ESC to exit')
    os.mkdir(sessionPath)
    logging.basicConfig(filename=os.path.join(sessionPath, "event.log"), level=logging.INFO, format='%(asctime)s %(message)s')
    with open(os.path.join(sessionPath, "PID_Data.csv"), 'a') as f:
        data = ["timestamp", "iterationTime", "Kp", "Ki", "Kd", "x_error","X_integral", "x_derivative", "x_output", "y_error","y_integral", "y_derivative", "y_output", "command_time"]
        writer = csv.writer(f)
        print("write data to CSV")
        writer.writerow(data)
    with open(os.path.join(sessionPath, "Position_Data.csv"), 'a') as f:
        data = ["timestamp", "pos_x", "pos_y", "pos_z", "altitude"]
        writer = csv.writer(f)
        print("write data to CSV")
        writer.writerow(data)
    logging.info('starting log file')
    mambo = Mambo(None, use_wifi=True) 
    print("trying to connect to mambo now")
    success = mambo.connect(num_retries=3)
    print("connected: %s" % success)

    cv2.namedWindow("Command_duration")
    cv2.resizeWindow("Command_duration", 640, 240)
    cv2.createTrackbar("T", "Command_duration", 0, 100, empty)

    clear_all_images(mambo)
    mambo.set_user_picture_event_callback(download_images_and_take_image, mambo)
    mambo.ask_for_state_update()
    mambo.smart_sleep(1)
    logging.info("take picture")
    mambo.take_picture()
    pictureTimestampQueue.append(datetime.timestamp(datetime.now()))
    mambo.smart_sleep(1)

    keyboardControlThread.start()
    pidAutoTrackThread.start()
    loggingThread = threading.Thread(target=log_position, args=(mambo,))
    loggingThread.start()

    while True:
        if contourFrameToShow is not None:
            cv2.imshow("Contour", contourFrameToShow)
            cv2.waitKey(1)

    cv2.destroyAllWindows()