import numpy as np
import cv2
import math

###
#
# sam_locate_2.py | Locate and determine the vision target
#
# This program works with the recorded image, and it finds the target and
# highlights it in the image.
#
###

### GLOBALS ###

# This instantiates the variables we need to access globally.

displayThreshold = False # a boolean for whether we are displaying the threshold
                         # mask on the screen

### END GLOBALS ###
### FUNCTIONS ###

# this function is what is passed as our click handling function
# it receives an event code, the x and y where on the screen it occurred, and
# other useless stuff
def clickFunc(evt,x,y,flags,param):
    global displayThreshold

    # simply toggle the mask when left click
    if evt == cv2.EVENT_LBUTTONDOWN:
        displayThreshold = not displayThreshold

# find corners of the contour (attempt 1)
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

# find corners in the contour (attempt 2)
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

    # for each point, calculate the distance from each corner of the screen.
    # if it is smaller than the smallest currently found, it will be
    # considered the corner for that corner of the screen
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

# calculates the angle at a point by supplying the before point,
# the intersection, and the after point. returns in degrees
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

# simplifies a contour into (ideally) 4 lines
def simplifyContour(contour):
    # create an array filled with nones, with the length of the contour long
    out = [None] * len(contour)

    # loop through the contour points
    for i in xrange(len(contour)):
        # find before, point in question, and after points
        before = (contour[len(contour)-1][0] if i==0 else contour[i-1][0])
        point = contour[i][0]
        after = (contour[0][0] if i == len(contour)-1 else contour[i+1][0])

        # calculate the angle, and add it to the list if the angle is less than 90.
        angle = calculateAngle(before, point, after)
        out[i] = contour[i][0].tolist() if angle < 90 else None

    # define the function to remove nones
    def remFunc(item):
        return not item is None

    # remove nones
    out = filter(remFunc, out)

    # copy out to a new list called cout
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

    # return the simplified contour
    return np.array(filter(remFunc, cout))

# find corners of a contour (attempt 3)
# this uses a method similar to the distance method, but it uses
# a fitness value, calculated by factoring in the distances from /both/
# edges of the screen.
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

    # define the fitness function
    def fitnessFunc(corner,point):
        global w,h
        
        disy = abs(corner[0]-point[0])
        disx = abs(corner[1]-point[1])

        return (disy * disx) / (w*h)

    # same as attempt 2, the lower the fitness, the better
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

# draw a point on the screen
def drawPoint(img,pt,color=(0,255,0)):
    # pt is (y,x)
    cv2.circle(img,pt,4,color,-1)

# the distance formula
# can accept as tuples or arrays
def distance(p0,p1):
    return math.sqrt((p0[0] - p1[0])**2 + (p0[1] - p1[1])**2)

### END FUNCTIONS ###

# instantiate the video capture object
cap = cv2.VideoCapture("video.avi")

# get width and height
w = cap.get(3)
h = cap.get(4)

# set a named window up, so we can bind click functions
cv2.namedWindow("frame")
# bind the click function
cv2.setMouseCallback("frame",clickFunc)

# print instructions
print "Calibration program started..."
print "Left click to include that value in calibration,"
print "Each left click expands the range to include that value."
print "Right click to toggle seeing what the mask looks like."
print "After each click, the coordinates and hsv values are printed, then the current range."
print "Press Q to exit."
print ""

# enter into the frame-fetching loop
while(True):
    # capture each frame
    ret, frame = cap.read()

    # set the frames per second... doesn't have an effect
    cap.set(cv2.CAP_PROP_FPS,12)

    # loop over when we hit the last frame
    if frame is None:
        cap.set(cv2.CAP_PROP_POS_FRAMES,0)
        continue;

    # convert to hsv and save that as a new image in the global var
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # set the threshold values
    # these are predefined calibration values
    hsvLow = np.array([76,208,78])
    hsvHigh = np.array([81,255,232])
    bgrLow = np.array([54,78,0])
    bgrHigh = np.array([146,232,23])

    # mask the hsv and frame images...
    # that means, make a binary (b&w) image
    # only showing white where the hsv (or bgr) values, are within the range
    mask = cv2.inRange(hsv, hsvLow, hsvHigh)
    mask2 = cv2.inRange(frame, bgrLow, bgrHigh)

    # do a bitwise and, meaning, only keep the white where both masks are white
    bw = cv2.bitwise_and(mask,mask2)

    # erode the image
    bw = cv2.dilate(bw, None, None, None, 3)

    # contour it
    # find the contours
    bw, contours, hierarchy = cv2.findContours(bw,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    largestArea = 0
    largestAreaIndex = -1
    for i in xrange(len(contours)):
        # make them convex hulls, and keep the one with the largest area
        contours[i] = cv2.convexHull(contours[i])
        area = cv2.contourArea(contours[i])
        if area > largestArea:
            largestArea = area
            largestAreaIndex = i

    # if we found a contour
    if(largestAreaIndex > -1):
        # grab it, and simplify it
        contour = contours[largestAreaIndex]
        contour = simplifyContour(contour)

        # draw it on screen
        cv2.drawContours(frame, [contour], -1, (0,0,255), 3)

        # if it has 4 points (a rectangle)
        if len(contour) == 4:
            # find the corners
            # I determined method 2 is the most efficient
            a,b,c,d = findCorners2(contour)
            #print (a,b,c,d)
            # draw the points on screen
            drawPoint(frame, a, (0,255,0))
            drawPoint(frame, b, (255,255,0))
            drawPoint(frame, c, (0,255,255))
            drawPoint(frame, d, (255,255,255))
            
            # NOTUSED
            # calculate horizontal ppi and vertical ppi
            horizontalPPI = distance(a,b)/20; #20 inches width
            verticalPPI = distance(a,d)/14; #14 inches height
            
    # Display the resulting frame
    # if displaythreshold is true, it will display the mask, otherwise
    # it will display the actual frame
    if not displayThreshold:
        cv2.imshow('frame',frame)
    else:
        cv2.imshow('frame',bw)

    # wait for the q key to quit
    # when zero is passed to waitKey, it waits indefinitely
    # for the next frame to display
    # you can press any key to advance to the next frame
    if cv2.waitKey(0) & 0xFF == ord('q'):
        break
    
# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
