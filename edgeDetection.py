import cv2
import numpy as np

def getLargestContourCenter(img, imgContour):
    contours, hierarchy = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(imgContour, contours, -1, (255, 0, 255),7)
    contoursSorted = sorted(contours, key=lambda x: cv2.contourArea(x))
    for contour in reversed(contoursSorted):
        perimeter = cv2.arcLength(contoursSorted[-1], True)
        approx = cv2.approxPolyDP(contoursSorted[-1], 0.05 * perimeter, True)
        if len(approx) == 4:
            x, y, w, h = cv2.boundingRect(approx)
            cv2.rectangle(imgContour, (x, y), (x +w, y + h), (0, 255, 0), 5)
            return (x + w/2, y + h/2)
    return None


def getPolarCoordsFromMiddleOfImage(middleOfContour):
    imageCenterXCoordinate = 320
    imageCenterYCoordinate = 240
    x = middleOfContour[0] - imageCenterXCoordinate 
    y = middleOfContour[1] - imageCenterYCoordinate

    rho = np.linalg.norm([x, y])
    phi = np.arctan2(y, x)
    return (rho, phi)

def writeTextOnImage(text, image):
    font = cv2.FONT_HERSHEY_SIMPLEX
    org = (50, 50)
    fontScale = 0.5
    color = (255, 0, 0)
    thickness = 2
    image = cv2.putText(image, text, org, font, 
                   fontScale, color, thickness, cv2.LINE_AA)

def getContoursOfImage(image, imgContour, cannyLowerThreshold=25, cannyUpperThreshold=90):
    imgBlur = cv2.GaussianBlur(image, (7, 7), 1)
    grayscale = cv2.cvtColor(imgBlur, cv2.COLOR_BGR2GRAY)
    edge_image = cv2.Canny(grayscale, cannyLowerThreshold, cannyUpperThreshold)
    kernel = np.ones((5, 5))
    imgDil = cv2.dilate(edge_image, kernel, iterations=1)
    contourCenter = getLargestContourCenter(imgDil, imgContour)
    if contourCenter is not None:
        polarCoords = getPolarCoordsFromMiddleOfImage(contourCenter)
        polarCoordsString = f"rho {polarCoords[0]:.3f}, phi {np.rad2deg(polarCoords[1]):.3f}, cL: {cannyLowerThreshold}, cU: {cannyUpperThreshold}"
        writeTextOnImage(polarCoordsString, imgContour)
        return polarCoords
    return None
