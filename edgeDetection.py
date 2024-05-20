import cv2
import numpy as np
from scipy.spatial import ConvexHull, convex_hull_plot_2d
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import math
from Target import Target
from TargetCandidate import TargetCandidate
from operator import attrgetter

def getTargetContourCenterAndSize(imgOriginal, imgDil, grayscaleImage, imgContour, plotConvexHulls):
    contours, hierarchy = cv2.findContours(imgDil, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(imgContour, contours, -1, (255, 0, 255),2)
    contoursSorted = sorted(contours, key=lambda x: cv2.contourArea(x))
    largestArea = 0

    targetCandidates = []
    for contour in contoursSorted:
        perimeter = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.05 * perimeter, True)
        
        reshaped_contour = np.vstack(contour).squeeze()
        reshaped_approx = np.vstack(approx).squeeze()
        if(len(reshaped_approx) < 3):
            continue
        hull = ConvexHull(reshaped_approx)        
        roiPoints = approx[hull.vertices]
        masked_image = maskImageOutsideConvexHull(grayscaleImage, roiPoints)
        avg_brightness = getAverageBrightness(masked_image)
        if plotConvexHulls:
            graphConvexHull(hull, reshaped_approx, reshaped_contour, masked_image, imgOriginal.copy(), avg_brightness)
        targetCandidates.append(TargetCandidate(points=roiPoints, avgBrightness=avg_brightness, cornerAngles=anglesOfConvexHull(roiPoints.squeeze()), area=hull.volume))

    brightnessThresholdForWhitePaper = 180
    targetCandidates = sorted(targetCandidates, key=attrgetter('avgBrightness'), reverse=True)
    for candidate in targetCandidates:
        if candidate.containsRightAngles() and candidate.avgBrightness > brightnessThresholdForWhitePaper:
            x, y, w, h = cv2.boundingRect(candidate.points)
            cv2.rectangle(imgContour, (x, y), (x +w, y + h), (0, 255, 0), 5)
            rhoAndPhi = getPolarCoordsFromMiddleOfImage(x + w/2, y + h/2)
            return Target(x_center=x + w/2, y_center=y + h/2, area=candidate.area, width=w, height=h, rho=rhoAndPhi[0], phi=rhoAndPhi[1], timestamp=0.0)
    return None

def anglesOfConvexHull(roi_corners):
    if len(roi_corners) < 3:
        return
    angles  = []
    for i in range(0, len(roi_corners)):

        index_1 = i 
        index_2 = (i + 1) % len(roi_corners)
        index_3 = (i + 2) % len(roi_corners) 
        angles.append(angleBetweenPoints(roi_corners[index_1], roi_corners[index_2], roi_corners[index_3]))
    return angles

def angleBetweenPoints(P1, P2, P3):
    # law of cosines, find angle at P2 given the lines between points P1, to P2, to P3
    # angle123 = arccos((P12^2 + P23^2 - P23^2) / (2 * P12 * P23))
    P12 = distance(P1, P2)
    P13 = distance(P1, P3)
    P23 = distance(P2, P3)
    return np.arccos((P12**2 + P23**2 - P13**2)/ (2 * P12 * P23))

def distance(point_a, point_b):
    return np.sqrt((point_a[0] - point_b[0])**2 + (point_a[1] - point_b[1])**2)


def maskImageOutsideConvexHull(image, roi_corners):
    mask = np.zeros(image.shape, dtype=np.uint8)
    # fill the ROI so it doesn't get wiped out when the mask is applied
    ignore_mask_color = (255)
    cv2.fillConvexPoly(mask, roi_corners, ignore_mask_color)

    # apply the mask
    masked_image = cv2.bitwise_and(image, mask)
    return masked_image

def getAverageBrightness(masked_image):
    return np.average(masked_image, weights=masked_image>10)

def graphConvexHull(hull, points, full_contour, masked_image, original_image, avg_brightness):
    fig = plt.figure(figsize=(10, 7.5))
    gs = GridSpec(2, 2, figure=fig)

    ax1 = fig.add_subplot(gs[0, :])
    ax2 = fig.add_subplot(gs[1, 0])
    ax3 = fig.add_subplot(gs[1, 1])

    fig.suptitle(f"The average brightness is {avg_brightness}")

    closedVerticesList  = np.append(hull.vertices, hull.vertices[0])
    ax1.plot(full_contour[:, 0], -full_contour[:, 1], '+' )
    ax1.plot(points[:, 0], -points[:,1], 'o')
    ax1.plot(points[closedVerticesList,0], -points[closedVerticesList,1], 'r--', lw=2)
    ax1.plot(points[hull.vertices[0],0], -points[hull.vertices[0],1], 'ro')
    
    ax2.imshow(masked_image, cmap='gray' )
    ax2.axis('off')

    cv2.drawContours(original_image, [full_contour], -1, (255, 0, 255),2)
    ax3.imshow(cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB))
    ax3.axis('off')

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

def getContoursOfImage(image, imgContour, cannyLowerThreshold=25, cannyUpperThreshold=90, plotConvexHulls=False):
    imgBlur = cv2.GaussianBlur(image, (7, 7), 1)
    grayscale = cv2.cvtColor(imgBlur, cv2.COLOR_BGR2GRAY)
    edge_image = cv2.Canny(grayscale, cannyLowerThreshold, cannyUpperThreshold)
    kernel = np.ones((5, 5))
    imgDil = cv2.dilate(edge_image, kernel, iterations=1)
    # cv2.imshow("imgDil", imgDil)

    target = getTargetContourCenterAndSize(image, imgDil, grayscale,imgContour, plotConvexHulls)
    if target is not None:
        dimensionString = f"{target.stringForImage()}"
        cannyThresholdsString= f"area {target.area},cL: {cannyLowerThreshold}, cU: {cannyUpperThreshold}"
        writeTextOnImage(dimensionString, imgContour, (50, 50))
        writeTextOnImage(cannyThresholdsString, imgContour, (50, 80))
        return target
    return None
