# MamboDronePrecisionLanding

This project is an attempt the get a parrot Mambo FPV minidrone to autonomously land on a target.

# Notes for improvements
- Proper Logging
- experiment to try and find out when the image was taken (how old is it by the time it's downloaded)
- try to use the accelerometer data do determine how the vehicle has moved since the last image
- try proper PID control for commanding guiding the vehicle to the target
- mirror on FPV camera to look down but actually get a decent frame rate


# Testing measurements 

- mambo.fly_direct(0, 20, 0, 0, 1) fly forward for about 1s results in about 40cm-42cm of travel, according to accelerometer data
- mambo.fly_direct(0, -20, 0, 0, 1) fly backwards for about 1s results in about 35cm-40cm of travel, according to accelerometer data
- mambo.fly_direct(-20, 0, 0, 0, 1) fly left for about 1s results in about 30-35cm of travel, according to accelerometer data
- mambo.fly_direct(20, 0, 0, 0, 1) fly right for about 1s results in about 30-35cm of travel, according to accelerometer data