import cv2
import numpy as np
import math
import time
from firebase import firebase
import json as simplejson

import _thread

globaldata = []

def senddataThread(ThreadName):
    from firebase import firebase
    application = firebase.FirebaseApplication('firebaseurl', None)

    while True:
        time.sleep(0.2)
        #try:
        if len(globaldata)!=0:
                post = application.post(data=globaldata[0], url='firebaseurl')
                print(globaldata)
                print('success')
                globaldata.remove(globaldata[0])
                for i in range(0,10):
                    try:
                        globaldata.pop()
                    except:
                        break

        #except:
            #pass



def lazyCut(img, point_A, point_B, palm_point):
    height, width, _ = img.shape
    x1, y1 = point_A[0], point_A[1]
    x2, y2 = point_B[0], point_B[1]
    k = (y2 - y1) / (x2 - x1)
    b = y1 - k * x1
    side = k * palm_point[0] + b - palm_point[1]
    crosspoint = []
    contourpoints = []
    crosspoint.append(point_A)
    crosspoint.append(point_B)

    crosspoint.append((0, height))

    crosspoint.append((width, height))

    for point in crosspoint:
        pointSide = k * point[0] + b - point[1]
        contourpoints.append([point[0], point[1]])
        cv2.circle(img, point, 2, [255, 255, 255], 3)
    contourpoints.append([point_A[0], point_A[1]])
    contourpoints.append([point_B[0], point_B[1]])
    # print(contourpoints)
    return contourpoints


def cut(img, point_A, point_B, palm_point):
    height, width, _ = img.shape
    x1, y1 = point_A[0], point_A[1]
    x2, y2 = point_B[0], point_B[1]
    k = (y2 - y1) / (x2 - x1)
    b = y1 - k * x1
    side = k * palm_point[0] + b - palm_point[1]
    crosspoint = []
    contourpoints = []
    crosspoint.append(point_A)
    crosspoint.append(point_B)
    crosspoint.append((0, 0))
    crosspoint.append((0, height))
    crosspoint.append((width, 0))
    crosspoint.append((width, height))
    if 0 <= int(b) <= height:
        crosspoint.append((0, int(b)))
    if 0 <= int(width * k + b) <= height:
        crosspoint.append((width, int(width * k + b)))
    if 0 <= (int(-1 * b / k)) <= width:
        crosspoint.append((int(-1 * b / k), 0))
    if 0 <= int((height - b) / k) <= width:
        crosspoint.append((int((height - b) / k), height))
    for point in crosspoint:
        pointSide = k * point[0] + b - point[1]
        if pointSide * side < 0:
            contourpoints.append([point[0], point[1]])
            cv2.circle(img, point, 2, [255, 255, 255], 3)
    contourpoints.append([point_A[0], point_A[1]])
    contourpoints.append([point_B[0], point_B[1]])
    # print(contourpoints)
    return contourpoints


def findBoundaries(mask, maxLoc, maxVal):
    radiusStep = 1
    initialRadius = maxVal
    initlaAngle = 0
    angleStep = math.pi / 12
    pointSamples = []
    angle = 0
    boundaries = []
    boundariesTuple = []
    for i in range(0, 24):
        samplePoint = (int(maxLoc[0] + maxVal * math.cos(angle)), int(maxLoc[1] + maxVal * math.sin(angle)))
        angle += angleStep
        pointSamples.append(samplePoint)
    for sample in pointSamples:
        radius = 20
        findBoundary = False
        while (radius <= maxVal and not findBoundary):
            angle = 0
            for i in range(0, 24):
                point = (int(sample[0] + radius * math.cos(angle)), int(sample[1] + radius * math.sin(angle)))
                angle += angleStep
                try:
                    sum = mask[point[1], point[0]] + \
                          mask[point[1], point[0] + 1] + \
                          mask[point[1], point[0] - 1] + \
                          mask[point[1] + 1, point[0]] + \
                          mask[point[1] + 1, point[0] + 1] + \
                          mask[point[1] + 1, point[0] - 1] + \
                          mask[point[1] - 1, point[0]] + \
                          mask[point[1] - 1, point[0] + 1] + \
                          mask[point[1] - 1, point[0] - 1]
                    if sum == 0:
                        findBoundary = True
                        validBoundary = [point[0], point[1]]
                        boundariesTuple.append(point)
                        boundaries.append(validBoundary)
                        break
                except:
                    findBoundary = False
            radius += 5
    return boundaries, boundariesTuple


def crossangle(p1, p2, p3, p4):
    m1 = (p2[1] - p1[1]) / (p2[0] - p1[0])
    m2 = (p3[1] - p4[1]) / (p3[0] - p4[0])
    try:
        angle = np.arctan((m1 - m2) / (1 + m1 * m2))
    except:
        angle = np.pi / 2
    return angle


def searchPalmLine(img_nobackground_wb, wpt0, wpt1, palm_point, thumb=False):
    height, width = img_nobackground_wb.shape
    blank_image = np.zeros((height, width, 1), np.uint8)
    k = (wpt1[1] - wpt0[1]) / (wpt1[0] - wpt0[0])
    b0 = wpt1[1] - k * wpt1[0]

    b1 = palm_point[1] - k * palm_point[0]
    b2 = 2.1 * b1 - b0
    pt3 = (0, int(b2))
    pt4 = (width, int(k * width + b2))
    cv2.line(blank_image, pt3, pt4, (255, 255, 255), 4)
    temp_img = cv2.bitwise_and(img_nobackground_wb, img_nobackground_wb, mask=blank_image)

    vaildContours, closed, hull = getMaxContour(temp_img)
    c = vaildContours[0]
    extLeft = tuple(c[c[:, :, 0].argmin()][0])
    # print(extLeft)
    extRight = tuple(c[c[:, :, 0].argmax()][0])
    # print(extRight)
    # print(vaildContours)
    cv2.drawContours(temp_img, vaildContours, 0, (255, 255, 255), 1)
    '''
    pt5 = (vaildContours[0][0][0], vaildContours[0][0][1])
    pt6 = (vaildContours[0][len(vaildContours)-1][0], vaildContours[0][len(vaildContours)-1][1])
    '''
    cv2.line(temp_img, extLeft, extRight, (255, 255, 255), 20)

    return vaildContours, extLeft, extRight


def crosspoint_cal(p1, p2, p3, p4):
    k1 = (p2[1] - p1[1]) / (p2[0] - p1[0])
    k2 = (p3[1] - p4[1]) / (p3[0] - p4[0])
    b1 = p2[1] - k1 * p2[0]
    b2 = p3[1] - k2 * p3[0]
    x = (b2 - b1) / (k1 - k2)
    y = k1 * x + b1
    left_percent = (x - p3[0]) / (p4[0] - p3[0])
    return left_percent


def detectfingers(img):
    height, width, _ = img.shape
    skinMask = HSVBin(img)
    skinMask_copy = skinMask.copy()
    # contours, closed, fake_hull= getContours(skinMask)
    distimage = cv2.distanceTransform(src=skinMask_copy, distanceType=cv2.DIST_L2, maskSize=5)

    minVal, maxVal, minLoc, maxLoc = cv2.minMaxLoc(distimage, mask=None)
    cv2.normalize(distimage, distimage, 0, 1.0, cv2.NORM_MINMAX)
    cv2.imshow('distimg', distimage)
    palm_point = maxLoc

    skinMaskColor = cv2.cvtColor(skinMask, cv2.COLOR_BAYER_GR2RGB)
    # cv2.circle(distimage,maxLoc, int(maxVal),(255, 0, 0),1)
    # cv2.circle(skinMaskColor, maxLoc, int(maxVal), (255, 0, 0), 1)
    # cv2.circle(skinMaskColor, maxLoc, int(maxVal * 1.2), (255, 0, 0), 1)
    boundaries, boundarytuples = findBoundaries(skinMask, maxLoc, maxVal * 1.3)
    # wristPoints
    wpt0 = ()
    wpt1 = ()
    maxDistance = 0
    for i in range(0, len(boundarytuples) - 1):
        pt0 = boundarytuples[i]
        pt1 = boundarytuples[i + 1]
        distance = np.square(pt0[0] - pt1[0]) + np.square(pt0[1] - pt1[1])
        if distance > maxDistance:
            wpt0 = pt0
            wpt1 = pt1
            maxDistance = distance
    pt0 = boundarytuples[len(boundarytuples) - 1]
    pt1 = boundarytuples[0]
    distance = np.square(pt0[0] - pt1[0]) + np.square(pt0[1] - pt1[1])
    if distance > maxDistance:
        wpt0 = pt0
        wpt1 = pt1
        maxDistance = distance
    cv2.line(skinMaskColor, wpt0, wpt1, (0, 0, 255), 4)

    mycontour = np.array(boundaries)
    cv2.drawContours(skinMaskColor, [mycontour], 0, (255, 255, 255), 1)
    cv2.fillPoly(skinMaskColor, [mycontour], [0, 0, 0])
    cv2.imshow('palmMask', skinMaskColor)
    wristCountourPoints = lazyCut(skinMaskColor, wpt0, wpt1, palm_point)
    # print(len(wristCountourPoints))
    wristContour = np.array(wristCountourPoints)
    # wristContour = np.array([[480, 0], [480, 630], [350, 386], [281, 256]])
    cv2.drawContours(skinMaskColor,[wristContour],0,(255,255,255),7)

    cv2.fillPoly(skinMaskColor, [wristContour], [0, 0, 0])
    # cv2.circle(skinMaskColor,palm_point, 3, (255,255,255), 3)

    img_nobackground = cv2.bitwise_and(img, img, mask=skinMask)
    img_nobackground_wb = cv2.cvtColor(img_nobackground, cv2.COLOR_BGR2GRAY)
    (t, img_nobackground_wb) = cv2.threshold(img_nobackground_wb, 1, 255, cv2.THRESH_BINARY)
    new_img = img
    palm_line_cnt, leftpt, rightpt = searchPalmLine(img_nobackground_wb, wpt0, wpt1, palm_point)

    fingerImg = cv2.cvtColor(skinMaskColor, cv2.COLOR_BGR2GRAY)
    kernel = np.ones((1, 3), np.uint8)
    fingerImg = cv2.erode(fingerImg, kernel, iterations=10)
    # fingerImg = cv2.dilate(fingerImg, kernel, iterations=10)
    fingerCnt = getContours(fingerImg)
    if len(fingerCnt) >= 5:
        fingerCnt = fingerCnt[0:5]
    skinMaskColor = cv2.cvtColor(fingerImg, cv2.COLOR_BAYER_GR2RGB)
    cv2.drawContours(skinMaskColor, fingerCnt, -1, (0, 0, 255), 2)
    contours_poly = [None] * len(fingerCnt)
    boundRect = [None] * len(fingerCnt)
    tops = [None] * len(fingerCnt)
    centers = []
    crosspoints = []
    rectarea = []
    length = []
    thumb = 6
    forefinger = 6
    midfinger = 6
    ringfinger = 6
    pinky = 6
    radius = [None] * len(fingerCnt)
    for i, c in enumerate(fingerCnt):
        contours_poly[i] = cv2.approxPolyDP(c, 3, True)
        tops[i] = tuple(c[c[:, :, 1].argmin()][0])
        boundRect[i] = cv2.boundingRect(contours_poly[i])
        # print(i)
        # print(boundRect[i])
        cv2.rectangle(skinMaskColor, (int(boundRect[i][0]), int(boundRect[i][1])),
                      (int(boundRect[i][0] + boundRect[i][2]), int(boundRect[i][1] + boundRect[i][3])), (255, 0, 0), 2)
        centers.append((int(boundRect[i][0] + boundRect[i][2] / 2), int(boundRect[i][1] + boundRect[i][3] / 2)))
        crosspoints.append(crosspoint_cal(palm_point, centers[i], leftpt, rightpt))
        rectarea.append(boundRect[i][2] * boundRect[i][3])
        length.append(boundRect[i][3])

        cv2.line(skinMaskColor, centers[i], palm_point, (0, 0, 255), 1)
        if -0.6 < crossangle(palm_point, centers[i], wpt0, wpt1) < 0.6:
            thumb = i
        elif 0.3 > crosspoints[i] > 0:
            forefinger = i
        elif 0.5 > crosspoints[i] > 0.3:
            midfinger = i
        elif 0.7 > crosspoints[i] > 0.5:
            ringfinger = i
        elif crosspoints[i] > 0.7:
            pinky = i
    # print(rectarea)
    if thumb != 6:
        cv2.circle(skinMaskColor, centers[thumb], 3, (0, 0, 255), 3)
        cv2.circle(skinMaskColor, tops[thumb], 3, (0, 0, 255), 3)
        thumb_area = cv2.contourArea(fingerCnt[thumb])
    if forefinger != 6:
        cv2.circle(skinMaskColor, centers[forefinger], 3, (0, 0, 255), 3)
        cv2.circle(skinMaskColor, tops[forefinger], 3, (0, 0, 255), 3)
        thumb_area = cv2.contourArea(fingerCnt[forefinger])
    if midfinger != 6:
        cv2.circle(skinMaskColor, centers[midfinger], 3, (0, 0, 255), 3)
        cv2.circle(skinMaskColor, tops[midfinger], 3, (0, 0, 255), 3)
        thumb_area = cv2.contourArea(fingerCnt[midfinger])
    if ringfinger != 6:
        cv2.circle(skinMaskColor, centers[ringfinger], 3, (0, 0, 255), 3)
        cv2.circle(skinMaskColor, tops[ringfinger], 3, (0, 0, 255), 3)
        thumb_area = cv2.contourArea(fingerCnt[ringfinger])
    if pinky != 6:
        cv2.circle(skinMaskColor, centers[pinky], 3, (0, 0, 255), 3)
        cv2.circle(skinMaskColor, tops[pinky], 3, (0, 0, 255), 3)
        thumb_area = cv2.contourArea(fingerCnt[pinky])
    cv2.circle(skinMaskColor, palm_point, 3, (0, 0, 255), 3)
    cv2.line(skinMaskColor, wpt1, wpt0, (0, 0, 255), 2)
    #print(crosspoints)

    # cv2.drawContours(skinMaskColor, palm_line_cnt, 0, (255,255,255), 2)
    cv2.line(skinMaskColor, leftpt, rightpt, (255, 255, 255), 2)
    drawGrids(skinMaskColor, 4, False, False)
    cv2.imshow('results', skinMaskColor)

    # drawGrids(img, 8, True)
    # cv2.imshow('capture',img)
    #cv2.imshow('nobackground', img_nobackground)

    #print(thumb, forefinger, midfinger, ringfinger, pinky, rectarea)
    #print(time.time())
    return thumb, forefinger, midfinger, ringfinger, pinky, rectarea, length,tops




def main(arg):
    print('Start camera')
    from firebase import firebase
    cap = cv2.VideoCapture(0)
    state = [0, 0, 0, 0, 0]  # fivefingers, 0 miss, 1 straight, 2 bent
    area = [0, 0, 0, 0, 0]
    length = [0, 0, 0, 0, 0]
    application = firebase.FirebaseApplication('https://iotproj-510ee.firebaseio.com', None)
    runtime = 0
    while (cap.isOpened()):
        try:
            ret, sample = cap.read()
            tops = []
            imgs = []
            blank_image = np.zeros((sample.shape[0],sample.shape[1], 3), np.float32)
            for i in range(0,3):

                ret, img = cap.read()
                imgs.append(img.astype(np.float32))
            for img in imgs:
                blank_image = img/3 + blank_image
            aveimg = blank_image.astype(np.uint8)
            img = aveimg
            width = img.shape[1]
            height = img.shape[0]
            thumb, forefinger, midfinger, ringfinger, pinky, rectarea, length_c, tops = detectfingers(img)

            fingers = [thumb, forefinger, midfinger, ringfinger, pinky]

            for i, finger in enumerate(fingers):

                if finger != 6:
                    #print(length_c[finger], length[i])
                    if state[i] == 0:
                        state[i] = 1
                    elif state[i] == 2 and length_c[finger] >= 1.2 * length[i]:
                        #if length_c[finger] >= 1.2 * length[i] or rectarea[finger] >= 1.2 * area[i]:
                        state[i] = 1
                    elif state[i] == 1 and length_c[finger] <= 0.86 * length[i]:
                        #if length_c[finger] <= 0.8 * length[i] or rectarea[finger] >= 1.2 * area[i]:
                        #print(length_c[finger], length[i])


                        x = int(tops[finger][0] * 408 / width)
                        #print(width, tops[finger][0], x)
                        #print(height)
                        y = int((height - tops[finger][1]) * 408 / height)
                        data = {'x': x, 'y': y}
                        print(data)
                        globaldata.append(data)
                        state[i] = 2
                    length[i] = length_c[finger]

                    #print('333')
                    area[i] = rectarea[finger]
                else:
                    state[i] = 0
           # print(state)
            #print(length)
            k = cv2.waitKey(10)
            if k == 27:
                break
        except:
            pass




def drawGrids(img, gridNum, hand, thumb):
    '''
    hand: True for right, False for left
    '''
    height, width, channels = img.shape
    if  thumb:
        if not hand:
            thumb_point = int(width * 1 / 4)
            horizonPoints = [thumb_point]
            horizonPoints.append(int(thumb_point + (width - thumb_point) * 1 / gridNum))
            for i in range(2, gridNum):
                horizonPoints.append(int(horizonPoints[i - 1] + (width - thumb_point) * 1 / gridNum))
            # print(horizonPoints)

            verticalPoints = [int(height * 1 / gridNum)]
            for i in range(1, gridNum - 1):
                verticalPoints.append(int(verticalPoints[i - 1] + height * 1 / gridNum))
            # print(verticalPoints)

            for point in horizonPoints:
                cv2.line(img, (point, 0), (point, height), (255, 0, 0), 1)
            for point in verticalPoints:
                cv2.line(img, (thumb_point, point), (width, point), (255, 0, 0), 1)
        else:
            thumb_point = int(width * 3 / 4)
            horizonPoints = [int(thumb_point * 1 / gridNum)]
            for i in range(1, gridNum - 1):
                horizonPoints.append(int(horizonPoints[i - 1] + thumb_point * 1 / gridNum))
            horizonPoints.append(thumb_point)

            verticalPoints = [int(height * 1 / gridNum)]
            for i in range(1, gridNum - 1):
                verticalPoints.append(int(verticalPoints[i - 1] + height * 1 / gridNum))

            for point in horizonPoints:
                cv2.line(img, (point, 0), (point, height), (255, 0, 0), 1)
            for point in verticalPoints:
                cv2.line(img, (0, point), (thumb_point, point), (255, 0, 0), 1)
    else:
        horizonPoints = [0]
        for i in range(1, gridNum):
            horizonPoints.append(int(horizonPoints[i - 1] + width * 1 / gridNum))
        verticalPoints = [0]
        for i in range(1, gridNum):
            verticalPoints.append(int(verticalPoints[i - 1] + height * 1 / gridNum))
        for point in horizonPoints:
            cv2.line(img, (point, 0), (point, height), (255, 0, 0), 1)
        for point in verticalPoints:
            cv2.line(img, (0, point), (width, point), (255, 0, 0), 1)

def getMaxContour(img):
    kernel = np.ones((3, 3), np.uint8)
    closed = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
    closed = cv2.morphologyEx(closed, cv2.MORPH_CLOSE, kernel)
    _, contours, h = cv2.findContours(closed, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    vaildContours = []
    hulls = []
    try:
        cnt = max(contours, key=lambda x: cv2.contourArea(x))
        hull = cv2.convexHull(cnt)
        vaildContours = [cnt]
    except:
        hull = []
        vaildContours = []
    defects = []

    return vaildContours, closed, hull


def getContours(img):
    kernel = np.ones((3, 3), np.uint8)
    closed = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
    closed = cv2.morphologyEx(closed, cv2.MORPH_CLOSE, kernel)
    _, contours, h = cv2.findContours(closed, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    vaildContours = []
    hulls = []
    for cnt in contours:
        if cv2.contourArea(cnt) >= 700:
            vaildContours.append(cnt)

    return vaildContours


def HSVBin(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    kernel = np.ones((5, 5), np.uint8)
    lower_skin = np.array([100, 50, 0])
    upper_skin = np.array([125, 255, 255])
    lower_skin = np.array([50, 37, 0])
    upper_skin = np.array([179, 255, 255])
    mask = cv2.inRange(hsv, lower_skin, upper_skin)
    # mask = cv2.erode(mask, kernel, iterations=1)
    mask = cv2.GaussianBlur(mask, (3, 3), 100)
    # res = cv2.bitwise_and(img,img,mask=mask)
    return mask


if __name__ == '__main__':
    _thread.start_new_thread(senddataThread, (1, ))
    main(1)

