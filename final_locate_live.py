import numpy as np
import cv2
import math
import random
from networktables import NetworkTable
table = NetworkTable.getTable('vision')

### GLOBALS ###

displayThreshold = False
### END GLOBALS ###
### CALIBRATION ###

# camera calibration
cameraRMS = 0.283286598231
cameraMatrix = np.float32([[1.12033194e+03, 0.0, 6.49786694e+02],
                           [0.0, 1.11455896e+03, 3.80918277e+02],
                           [0.0, 0.0, 1.0]])
cameraDistortion = np.float32([0.15190902, -0.78835469, 0.00402702, -0.00291226, -1.00032999])

# color calibration
calibrationTuple = ((60, 208, 25), (71, 255, 124), (0, 25, 0), (36, 124, 16))
calibrationTuple = ((60, 136, 35), (76, 255, 248), (0, 35, 0), (138, 248, 89))
calLowHSV, calHighHSV, calLowBGR, calHighBGR = calibrationTuple

# exposure
exposure = -9

# angle function values
angleFunc1A = -90.535724570955
angleFunc1B = 45.247456281206
angleFunc2A = -5.511621418651
angleFunc2B = 24.318446167847

# distance function values
distFunc1A = 0.08748759788249
distFunc1B = -0.5189968934245

distFunc2A = -0.1414458147376
distFunc2B = -0.0675841969269

cameraWidth = 580

### END CALIBRATION ###
### FUNCTIONS ###

# a clicking function, simply toggles the binary displayThreshold variable
def clickFunc(evt,x,y,flags,param):
    global displayThreshold

    if evt == cv2.EVENT_LBUTTONDOWN:
        displayThreshold = not displayThreshold

def findCorners2(contour):
    global w,h
    
    topLeftDistance = 9999999
    topRightDistance = 9999999
    bottomRightDistance = 9999999
    bottomLeftDistance = 9999999
    topLeft = topLeftOrigin = (0,0)
    topRight = topRightOrigin = (0,w)
    bottomRight = bottomRightOrigin = (h,w)
    bottomLeft = bottomLeftOrigin = (h,0)

    for pt2 in contour:
        y,x = pt2
        pt = (y,x)

        if distance(topLeftOrigin,pt) < topLeftDistance:
            topLeftDistance = distance(topLeftOrigin,pt)
            topLeft = pt

        if distance(topRightOrigin,pt) < topRightDistance:
            topRightDistance = distance(topRightOrigin,pt)
            topRight = pt

        if distance(bottomRightOrigin,pt) < bottomRightDistance:
            bottomRightDistance = distance(bottomRightOrigin,pt)
            bottomRight = pt

        if distance(bottomLeftOrigin,pt) < bottomLeftDistance:
            bottomLeftDistance = distance(bottomLeftOrigin,pt)
            bottomLeft = pt

    return (topLeft,topRight,bottomRight,bottomLeft)

def calculateAngle(before, point, after):
    a = distance(point,after)
    b = distance(before,after)
    c = distance(before,point)

    a2 = a**2
    b2 = b**2
    c2 = c**2

    n2ac = -2 * a * c

    b2ma2mc2 = b2 - a2 - c2

    B = 1/math.cos(b2ma2mc2/n2ac)

    return math.degrees(B)

def calculateCenter(centerX, width):
    widthCenter = width / 2.0
    centerX = centerX * 1.0
    return (centerX - widthCenter) / widthCenter

def simplifyContour(contour):
    out = [None] * len(contour)

    for i in xrange(len(contour)):
        before = (contour[len(contour)-1][0] if i==0 else contour[i-1][0])
        point = contour[i][0]
        after = (contour[0][0] if i == len(contour)-1 else contour[i+1][0])

        angle = calculateAngle(before, point, after)
        out[i] = contour[i][0].tolist() if angle < 90 else None

    def remFunc(item):
        return not item is None

    out = filter(remFunc, out)

    # copy out
    cout = list(out)

    # remove similar points
    # TODO: rewrite this to be more intelligent
    for i in xrange(len(out)):
        if i == 0:
            continue;

        oy,ox = out[i-1]
        y,x = out[i]

        if(abs(oy-y) < 40 and abs(ox-x) < 40):
            cout[i] = None
    
    return np.array(filter(remFunc, cout))

# draws a point on screen
def drawPoint(img,pt,color=(0,255,0)):
    # pt is (y,x)
    cv2.circle(img,pt,4,color,-1)

# mathematical distance
def distance(p0,p1):
    return math.sqrt((p0[0] - p1[0])**2 + (p0[1] - p1[1])**2)   

# computes the 2d transform matrix
def findTransform(contour,corners):
    global w,h
    
    # now that we have our rectangle of points, let's compute
    # the width of our new image
    (tl, tr, br, bl) = corners
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
 
    # ...and now for the height of our new image
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
 
    # take the maximum of the width and height values to reach
    # our final dimensions
    maxWidth = max(int(widthA), int(widthB))
    maxHeight = max(int(heightA), int(heightB))
 
    # construct our destination points
    dst = np.array([
	[0, 0],
	[maxWidth - 1, 0],
	[maxWidth - 1, maxHeight - 1],
	[0, maxHeight - 1]], dtype = "float32")
 
    # calculate the perspective transform matrix and warp
    # the perspective to grab the screen
    M = cv2.getPerspectiveTransform(np.array(corners, dtype="float32"), dst)
    
    return (M,maxWidth,maxHeight)

# gives a vector of x,y,z rotations from a 3x3 rotation matrix
def mat2euler(M, cy_thresh=None):
    ''' Discover Euler angle vector from 3x3 matrix

    Uses the conventions above.

    Parameters
    ----------
    M : array-like, shape (3,3)
    cy_thresh : None or scalar, optional
       threshold below which to give up on straightforward arctan for
       estimating x rotation.  If None (default), estimate from
       precision of input.

    Returns
    -------
    z : scalar
    y : scalar
    x : scalar
       Rotations in radians around z, y, x axes, respectively

    Notes
    -----
    If there was no numerical error, the routine could be derived using
    Sympy expression for z then y then x rotation matrix, which is::

      [                       cos(y)*cos(z),                       -cos(y)*sin(z),         sin(y)],
      [cos(x)*sin(z) + cos(z)*sin(x)*sin(y), cos(x)*cos(z) - sin(x)*sin(y)*sin(z), -cos(y)*sin(x)],
      [sin(x)*sin(z) - cos(x)*cos(z)*sin(y), cos(z)*sin(x) + cos(x)*sin(y)*sin(z),  cos(x)*cos(y)]

    with the obvious derivations for z, y, and x

       z = atan2(-r12, r11)
       y = asin(r13)
       x = atan2(-r23, r33)

    Problems arise when cos(y) is close to zero, because both of::

       z = atan2(cos(y)*sin(z), cos(y)*cos(z))
       x = atan2(cos(y)*sin(x), cos(x)*cos(y))

    will be close to atan2(0, 0), and highly unstable.

    The ``cy`` fix for numerical instability below is from: *Graphics
    Gems IV*, Paul Heckbert (editor), Academic Press, 1994, ISBN:
    0123361559.  Specifically it comes from EulerAngles.c by Ken
    Shoemake, and deals with the case where cos(y) is close to zero:

    See: http://www.graphicsgems.org/

    The code appears to be licensed (from the website) as "can be used
    without restrictions".
    '''
    M = np.asarray(M)
    if cy_thresh is None:
        try:
            cy_thresh = np.finfo(M.dtype).eps * 4
        except ValueError:
            cy_thresh = _FLOAT_EPS_4
    r11, r12, r13, r21, r22, r23, r31, r32, r33 = M.flat
    # cy: sqrt((cos(y)*cos(z))**2 + (cos(x)*cos(y))**2)
    cy = math.sqrt(r33*r33 + r23*r23)
    if cy > cy_thresh: # cos(y) not close to zero, standard form
        z = math.atan2(-r12,  r11) # atan2(cos(y)*sin(z), cos(y)*cos(z))
        y = math.atan2(r13,  cy) # atan2(sin(y), cy)
        x = math.atan2(-r23, r33) # atan2(cos(y)*sin(x), cos(x)*cos(y))
    else: # cos(y) (close to) zero, so x -> 0.0 (see above)
        # so r21 -> sin(z), r22 -> cos(z) and
        z = math.atan2(r21,  r22)
        y = math.atan2(r13,  cy) # atan2(sin(y), cy)
        x = 0.0
    return z, y, x

# estimates angle when board tilted far end left
def estimateAngleFunction1(thetaY):
    result = 0

    result += angleFunc1A

    try:
        result += angleFunc1B*math.log(thetaY)
    except ValueError:
        result = -999
    
    return result

# estimates angle when board tilted far end right
def estimateAngleFunction2(thetaY):
    thetaY = -thetaY
    thetaY += 7

    result = 0

    result += angleFunc2A

    try:
        result += angleFunc2B*math.log(thetaY)
    except ValueError:
        result = -999

    # subtract for correction
    try:
        result -= math.log(thetaY)
    except ValueError:
        result = -999
    
    return result

def estimateDistanceFunction1(translationZ):
    result = 0
    result += distFunc1A * translationZ
    result += distFunc1B
    return result

def estimateDistanceFunction2(translationY):
    result = 0
    result += distFunc2A * translationY
    result += distFunc2B
    return result


def countCameras():
    ret = 5
    for i in range(0,5):
        tempCam = cv2.VideoCapture(i)
        res = tempCam.isOpened()
        tempCam.release()
        print i
        if res is True:
            ret = i-1
    print ret
    return ret

def arbitrateValue(v1,v2):
    if v1 == -999:
        return v2
    if v2 == -999:
        return -v1
    if v2 > v1:
        return -v1
    if v2 > v2:
        return v2
    print "arbitrate failure:",v1,v2
    return -999

### END FUNCTIONS ###

# instantiate the video capture object
cap = cv2.VideoCapture(countCameras())

# get the width and height
w = cap.get(3)
h = cap.get(4)

# set the window as a named window so the click function can be bound
cv2.namedWindow("frame")
# bind the click function
cv2.setMouseCallback("frame",clickFunc)

#displayThreshold = False

# print instructions
print "Calibration program started..."
print "Left click to include that value in calibration,"
print "Each left click expands the range to include that value."
print "Right click to toggle seeing what the mask looks like."
print "After each click, the coordinates and hsv values are printed, then the current range."
print "Press Q to exit."
print ""

# infinite loop until brokwnk
while(True):
    # set exposure
    cap.set(15,exposure)

    # capture each frame
    ret, frame = cap.read()

    # flip the frame (optional)
    #frame = cv2.flip(frame,1)

    # Our operations on the frame come here
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # get the calibration values in
    hsvLow = np.array(list(calLowHSV))
    hsvHigh = np.array(list(calHighHSV))
    bgrLow = np.array(list(calLowBGR))
    bgrHigh = np.array(list(calHighBGR))

    # use the calibration values to mask out what we want
    mask = cv2.inRange(hsv, hsvLow, hsvHigh)
    mask2 = cv2.inRange(frame, bgrLow, bgrHigh)

    # combine the masks
    bw = cv2.bitwise_and(mask,mask2)

    # dilate the image to simplify small black bits
    bw = cv2.dilate(bw, None, None, None, 3)

    # contour it
    bw, contours, hierarchy = cv2.findContours(bw,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    # find the largest area contour
    largestArea = 0
    largestAreaIndex = -1
    for i in xrange(len(contours)):
        contours[i] = cv2.convexHull(contours[i])
        area = cv2.contourArea(contours[i])
        if area > largestArea:
            largestArea = area
            largestAreaIndex = i

    # if we have a largest area
    if(largestAreaIndex > -1):
        # get the contour
        contour = contours[largestAreaIndex]
        # simplify it (try to get it to 4 corners)
        contour = simplifyContour(contour)
        #contour = simplifyContour(contour)

        # draw it on screen
        cv2.drawContours(frame, [contour], -1, (0,0,255), 3)
        # if it has 4 corners
        if len(contour) == 4:
            # find the corners
            # fc2 was just the most reliable IMO -sam
            a,b,c,d = findCorners2(contour)
            #print (a,b,c,d)
            # draw the corners
            drawPoint(frame, a, (0,255,0))
            drawPoint(frame, b, (255,255,0))
            drawPoint(frame, c, (0,255,255))
            drawPoint(frame, d, (255,255,255))

            # find the 2D transform
            M,mw,mh = findTransform(contour,(a,b,c,d))

            # transform and get the transformed image
            bw = cv2.warpPerspective(frame,M,(mw,mh))#(int(w),int(h)))

            # convert tuple corners into lists
            # for compatability with numpy
            a2 = list(a)
            #a2.append(0.0)
            b2 = list(b)
            #b2.append(0.0)
            c2 = list(c)
            #c2.append(0.0)
            d2 = list(d)
            #d2.append(0.0)

            centerX = (a[0] + b[0] + c[0] + d[0]) / 4

            # 2d points representation of the object on screen in pixels
            # (y,x)
            imagePoints = np.array([a2,b2,c2,d2],dtype = "float32")

            # 3d points representation of the object in (y,x,z)
            objectPoints = np.float32([[6,-10,0],[6,10,0],[-6,10,0],[-6,-10,0]])

            #print objectPoints
            #print imagePoints

            # the hardest math: find the 3d rotation and translation vectors
            # of a known size 3d plane based on screen coordinates
            ret,rvec,tvec = cv2.solvePnP(objectPoints,imagePoints,cameraMatrix,cameraDistortion)

            # calculate to rotation matrix
            # don't ask me these weird names, we only want rM
            rM, jacobian = cv2.Rodrigues(rvec)

            # get the xtheta, ytheta, and ztheta values
            xTheta,yTheta,zTheta = mat2euler(rM)

            # move to degrees
            xTheta = math.degrees(xTheta)
            yTheta = math.degrees(yTheta)
            zTheta = math.degrees(zTheta)

            # print them out
            #print (xTheta,yTheta,zTheta)

            # attempt at calculating
            #print "Angle Function 1=",estimateAngleFunction1(yTheta)
            #print "Angle Function 2=",estimateAngleFunction2(yTheta)
            
            # calculate horizontal ppi and vertical ppi
            # NOT ACCURATE: NOT USED.
            #horizontalPPI = distance(a,b)/20; #20 inches width
            #verticalPPI = distance(a,d)/12; #12 inches height

            #width = mh
            #height = mw
            
            #print (width, height)
            #size = (width + (height * 5 / 3)) / 2
            #print tvec

            distZ = estimateDistanceFunction1(tvec[2][0])
            distY = estimateDistanceFunction2(tvec[1][0])
            distFt = math.sqrt(distZ ** 2 + distY ** 2)
            #print "dist z:", distZ
            #print "dist y:", distY
            #print "dist ft:", distFt

            angle = arbitrateValue(estimateAngleFunction1(yTheta),estimateAngleFunction2(yTheta))
            dist = distFt * 12

            centerValue = calculateCenter(centerX, cameraWidth)
            
            
            # publish
            table.putNumber('distance',dist)
            table.putNumber('angle',angle)
            table.putNumber('center',centerValue)
            print "dist:", dist
            print "angle:", angle
            print "centerValue:", centerValue
        else:
            table.putNumber('distance',-999)
            table.putNumber('angle',-999)
            table.putNumber('center',-999)

    else:
        table.putNumber('distance',-999)
        table.putNumber('angle',-999)
        table.putNumber('center',-999)
    
    # Display the resulting frame
    if not displayThreshold:
        cv2.imshow('frame',frame)
    else:
        cv2.imshow('frame',bw)

    # waitkey waits for x ms before continuing, zero means wait for a key
    # indefinitely. If key=Q, quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
