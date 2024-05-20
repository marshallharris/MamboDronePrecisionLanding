import edgeDetection
import cv2
import argparse

def empty(a):
    pass

def parseArgs():
    parser = argparse.ArgumentParser(description="Filename of image to do edge detection on")
    parser.add_argument("filename")
    parser.add_argument("--plot", action="store_true", help="Plot each convex hull")
    return parser.parse_args()

if __name__ == "__main__":

    args = parseArgs()
    cv2.namedWindow("Parameters")
    cv2.resizeWindow("Parameters", 640, 240)
    cv2.createTrackbar("Threshold1", "Parameters", 150, 255, empty)
    cv2.createTrackbar("Threshold2", "Parameters", 200, 255, empty)

    image = cv2.imread(args.filename)

    while True:
        threshold1 = cv2.getTrackbarPos("Threshold1", "Parameters")
        threshold2 = cv2.getTrackbarPos("Threshold2", "Parameters")
        imageContour = image.copy()
        edgeDetection.getContoursOfImage(image, imageContour, threshold1, threshold2, args.plot)

        cv2.imshow("Groundcam", imageContour)
        if cv2.waitKey(3000) == ord('q'):
            break