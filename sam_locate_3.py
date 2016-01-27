import numpy as np
import cv2
import math
import random

### GLOBALS ###

displayThreshold = False

### END GLOBALS ###
### FUNCTIONS ###

def clickFunc(evt,x,y,flags,param):
    global displayThreshold

    if evt == cv2.EVENT_LBUTTONDOWN:
        displayThreshold = not displayThreshold

'''
returns in order: (topleft,topright,bottomright,bottomleft)
'''
def findCorners(contour):
    minx = 999999999999999
    miny = 999999999999999
    maxx = 0
    maxy = 0
    
    # topleft
    for pt in contour:
        y,x = pt
        if y < miny and x < minx:
            miny = y
            minx = x

    topleft = (miny,minx)

    miny = 999999999999999
    minx = 999999999999999

    for pt in contour:
        y,x = pt
        if y < miny and x > maxx:
            miny = y
            maxx = x

    topright = (miny,maxx)

    miny = 999999999999999
    maxx = 0

    for pt in contour:
        y,x = pt
        if y > maxy and x > maxx:
            maxy = y
            maxx = x

    bottomright = (maxy,maxx)

    maxy = 0
    maxx = 0

    for pt in contour:
        y,x = pt
        if y > maxy and x < minx:
            maxy = y
            minx = x

    bottomleft = (maxy,minx)

    return (topleft,topright,bottomright,bottomleft)

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

    a2 = a**2;
    b2 = b**2;
    c2 = c**2;

    n2ac = -2 * a * c;

    b2ma2mc2 = b2 - a2 - c2;

    B = 1/math.cos(b2ma2mc2/n2ac)

    return math.degrees(B)

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

def findCorners3(contour):
    global w,h
    
    topLeftFitness = 1
    topRightFitness = 1
    bottomRightFitness = 1
    bottomLeftFitness = 1
    topLeft = topLeftOrigin = (0,0)
    topRight = topRightOrigin = (0,w)
    bottomRight = bottomRightOrigin = (h,w)
    bottomLeft = bottomLeftOrigin = (h,0)

    def fitnessFunc(corner,point):
        global w,h
        
        disy = abs(corner[0]-point[0])
        disx = abs(corner[1]-point[1])

        return (disy * disx) / (w*h)

    for pt2 in contour:
        y,x = pt2[0]
        pt = (y,x)

        if fitnessFunc(topLeftOrigin,pt) < topLeftFitness:
            topLeftFitness = fitnessFunc(topLeftOrigin,pt)
            topLeft = pt

        if fitnessFunc(topRightOrigin,pt) < topRightFitness:
            topRightFitness = fitnessFunc(topRightOrigin,pt)
            topRight = pt

        if fitnessFunc(bottomRightOrigin,pt) < bottomRightFitness:
            bottomRightFitness = fitnessFunc(bottomRightOrigin,pt)
            bottomRight = pt

        if fitnessFunc(bottomLeftOrigin,pt) < bottomLeftFitness:
            bottomLeftFitness = fitnessFunc(bottomLeftOrigin,pt)
            bottomLeft = pt

    return (topLeft,topRight,bottomRight,bottomLeft)

def drawPoint(img,pt,color=(0,255,0)):
    # pt is (y,x)
    cv2.circle(img,pt,4,color,-1)

def distance(p0,p1):
    return math.sqrt((p0[0] - p1[0])**2 + (p0[1] - p1[1])**2)

def findOrientation(corners):
    # returns -1 for small end left, 1 for small end right, and 0 if perfectly square
    # returns false if it cant determine orientation
    tl,tr,br,bl = corners
    if tr[0] == tl[0] and br[0] == bl[0]:
        return 0
    if tl[0] >= tr[0] and bl[0] <= br[0]:
        return -1
    if tr[0] >= tl[0] and br[0] <= bl[0]:
        return 1
    return False        

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
 
    # construct our destination points which will be used to
    # map the screen to a top-down, "birds eye" view
    dst = np.array([
	[0, 0],
	[maxWidth - 1, 0],
	[maxWidth - 1, maxHeight - 1],
	[0, maxHeight - 1]], dtype = "float32")
 
    # calculate the perspective transform matrix and warp
    # the perspective to grab the screen
    M = cv2.getPerspectiveTransform(np.array(corners, dtype="float32"), dst)
    
    return (M,maxWidth,maxHeight)
    

### END FUNCTIONS ###

# instantiate the video capture object
cap = cv2.VideoCapture("video.avi")

w = cap.get(3)
h = cap.get(4)

# set exposure
#cap.set(15, 0)

cv2.namedWindow("frame")
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

while(True):
    #print cap.get(15)
    #cap.set(15,-15);

    # capture each frame
    ret, frame = cap.read()

    cap.set(cv2.CAP_PROP_FPS,12)

    if frame is None:
        cap.set(cv2.CAP_PROP_POS_FRAMES,0)
        continue;

    # Our operations on the frame come here
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    hsvLow = np.array([76,208,78])
    hsvHigh = np.array([81,255,232])
    bgrLow = np.array([54,78,0])
    bgrHigh = np.array([146,232,23])

    mask = cv2.inRange(hsv, hsvLow, hsvHigh)
    mask2 = cv2.inRange(frame, bgrLow, bgrHigh)

    bw = cv2.bitwise_and(mask,mask2)

    # erode the image
    bw = cv2.dilate(bw, None, None, None, 3)

    # contour it
    bw, contours, hierarchy = cv2.findContours(bw,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    largestArea = 0
    largestAreaIndex = -1
    for i in xrange(len(contours)):
        contours[i] = cv2.convexHull(contours[i])
        area = cv2.contourArea(contours[i])
        if area > largestArea:
            largestArea = area
            largestAreaIndex = i

    if(largestAreaIndex > -1):
        contour = contours[largestAreaIndex]
        contour = simplifyContour(contour)
        #contour = simplifyContour(contour)

        cv2.drawContours(frame, [contour], -1, (0,0,255), 3)
        #print contours[largestAreaIndex]
        if len(contour) == 4:
            a,b,c,d = findCorners2(contour)
            #print (a,b,c,d)
            drawPoint(frame, a, (0,255,0))
            drawPoint(frame, b, (255,255,0))
            drawPoint(frame, c, (0,255,255))
            drawPoint(frame, d, (255,255,255))

            M,mw,mh = findTransform(contour,(a,b,c,d))

            print M

            bw = cv2.warpPerspective(frame,M,(mw,mh))#(int(w),int(h)))

            # calculate horizontal ppi and vertical ppi
            horizontalPPI = distance(a,b)/20; #20 inches width
            verticalPPI = distance(a,d)/14; #14 inches height
            
            

    
    # Display the resulting frame
    if not displayThreshold:
        cv2.imshow('frame',frame)
    else:
        cv2.imshow('frame',bw)
    
    if cv2.waitKey(0) & 0xFF == ord('q'):
        break
    

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
