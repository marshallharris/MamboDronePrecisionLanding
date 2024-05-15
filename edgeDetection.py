import cv2
import numpy as np
from scipy.spatial import ConvexHull, convex_hull_plot_2d
import matplotlib.pyplot as plt
from Target import Target

def getTargetContourCenterAndSize(img, grayscaleImage, imgContour):
    contours, hierarchy = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(imgContour, contours, -1, (255, 0, 255),2)
    contoursSorted = sorted(contours, key=lambda x: cv2.contourArea(x))
    largestArea = 0

    setOfPointsRepresentingLargestApproxContour = None
    roiPoints = None
    for contour in contoursSorted:
        perimeter = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.05 * perimeter, True)
        
        reshaped_approx = np.vstack(approx).squeeze()
        if(len(reshaped_approx) < 3):
            continue
        hull = ConvexHull(reshaped_approx)
        # graphConvexHull(hull, reshaped_approx)
        # hull.volume actually gives the area when dimension is 2D
        if(hull.volume > largestArea):
            largestArea = hull.volume
            setOfPointsRepresentingLargestApproxContour = approx
            roiPoints = approx[hull.vertices]

    brightnessThresholdForWhitePaper = 130
    if largestArea > 0 and averageBrightnessAcrossConvexHull(grayscaleImage, roiPoints) > brightnessThresholdForWhitePaper:
        x, y, w, h = cv2.boundingRect(setOfPointsRepresentingLargestApproxContour)
        cv2.rectangle(imgContour, (x, y), (x +w, y + h), (0, 255, 0), 5)
        rhoAndPhi = getPolarCoordsFromMiddleOfImage(x + w/2, y + h/2)
        return Target(x_center=x + w/2, y_center=y + h/2, area=largestArea, width=w, height=h, rho=rhoAndPhi[0], phi=rhoAndPhi[1], timestamp=0.0)
    return None

def averageBrightnessAcrossConvexHull(image, roi_corners):
    mask = np.zeros(image.shape, dtype=np.uint8)
    # fill the ROI so it doesn't get wiped out when the mask is applied
    ignore_mask_color = (255)
    cv2.fillConvexPoly(mask, roi_corners, ignore_mask_color)

    # apply the mask
    masked_image = cv2.bitwise_and(image, mask)
    average_color = getAverageColor(masked_image)
    print(f"non rect avg color{getAverageColor(masked_image)}")
    # cv2.imshow("masked", masked_image)
    return average_color

def getAverageColor(grayscaleImage):
    return np.average(grayscaleImage, weights=grayscaleImage>10)

def graphConvexHull(hull, points):
    closedVerticesList  = np.append(hull.vertices, hull.vertices[0])
    plt.plot(points[:, 0], -points[:,1], 'o')
    plt.plot(points[closedVerticesList,0], -points[closedVerticesList,1], 'r--', lw=2)
    plt.plot(points[hull.vertices[0],0], -points[hull.vertices[0],1], 'ro')
    plt.show()

def getPolarCoordsFromMiddleOfImage(target_center_x, target_center_y):
    imageCenterXCoordinate = 320
    imageCenterYCoordinate = 240
    x = target_center_x - imageCenterXCoordinate 
    y = target_center_y - imageCenterYCoordinate

    rho = np.linalg.norm([x, y])
    phi = np.arctan2(y, x)
    return (rho, phi)

def writeTextOnImage(text, image, org):
    font = cv2.FONT_HERSHEY_SIMPLEX
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
    # cv2.imshow("imgDil", imgDil)

    target = getTargetContourCenterAndSize(imgDil, grayscale,imgContour)
    if target is not None:
        dimensionString = f"{target.stringForImage()}"
        cannyThresholdsString= f"area {target.area},cL: {cannyLowerThreshold}, cU: {cannyUpperThreshold}"
        writeTextOnImage(dimensionString, imgContour, (50, 50))
        writeTextOnImage(cannyThresholdsString, imgContour, (50, 80))
        return target
    return None
