import numpy as np
import cv2

###
# 
# sam_calibrate_2.py | Second iteration of the calibration program
# 
# This is designed to calibrate the HSV (hue saturation value) and BGR (rgb backwards) thresholds of a camera image.
# The instructions to use it are printed to the console, and documentation of
# what code does is listed in the code in comments such as these.
# 
###

### GLOBALS ###

# This instantiates the variables we need to access globally.

frame = None # the regular bgr camera frame
hsv = None   # the hsv (hue saturation value) camera frame

lowh = None  # low value for the h in hsv
highh = None # high value for the h in hsv
lows = None  # low value for the s in hsv
highs = None # you're catching on...
lowv = None
highv = None
lowb = None  # bgr thresholds down from here
highb = None
lowg = None
highg = None
lowr = None
highr = None # and we're done

displayThreshold = False # a boolean for whether we are displaying the threshold
                         # mask on the screen

clickedYet = False       # a boolean for whether we have yet clicked the screen


### END GLOBALS ###
### FUNCTIONS ###

# this function is designed to return the integer of the last attached camera,
# the highest integer
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

# this function is what is passed as our click handling function
# it receives an event code, the x and y where on the screen it occurred, and
# other useless stuff
def clickFunction(event, x, y, flags, param):

    global frame,hsv
    global lowh,highh,lows,highs,lowv,highv
    global lowb,highb,lowg,highg,lowr,highr
    global displayThreshold
    global clickedYet

    # toggle the mask when right clicked
    if event == cv2.EVENT_RBUTTONUP:
        displayThreshold = not displayThreshold

    # when left clicked
    if event == cv2.EVENT_LBUTTONUP:
        # get the hsv at the location
        h,s,v = hsv[y,x]
        # get the bgr at the location
        b,g,r = frame[y,x]
        print ((y,x),(h,s,v))
        # all these statements store the hsvbgr values to the high and low
        # if they are out of the current range
        if h < lowh:
            lowh = h
        if h > highh:
            highh = h
        if s < lows:
            lows = s
        if s > highs:
            highs = s
        if v < lowv:
            lowv = v
        if v > highv:
            highv = v
        if b < lowb:
            lowb = b
        if b > highb:
            highb = b
        if g < lowg:
            lowg = g
        if g > highg:
            highg = g
        if r < lowr:
            lowr = r
        if r > highr:
            highr = r
        # if not yet clicked, save the values for the first time
        if not clickedYet:
            clickedYet = True
            lowh = h
            highh = h
            lows = s
            highs = s
            lowv = v
            highv = v
            lowb = b
            highb = b
            lowg = g
            highg = g
            lowr = r
            highr = r
        # print the calibration
        print "Current bounds low(h,s,v) high(h,s,v) low(b,g,r) high(b,g,r)"
        print ((lowh,lows,lowv),(highh,highs,highv),(lowb,lowg,lowr),(highb,highg,highr))

### END FUNCTIONS ###

# instantiate the video capture object
cap = cv2.VideoCapture(countCameras())

# set a named window up, so we can bind click functions
cv2.namedWindow("frame")
# bind the click function
cv2.setMouseCallback("frame",clickFunction)

# ensure that we are set not to display the threshold
displayThreshold = False

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

    # set the exposure of the camera
    # -11 is the max
    cap.set(15,-9)

    # capture each frame
    ret, frame = cap.read()

    # convert to hsv and save that as a new image in the global var
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # if the threshold values have been set, package them
    # into arrays, otherwise use arrays filled with zero.
    if not(lowh == None or lows == None or lowv == None or highh == None or highs == None or highv == None):
        hsvLow = np.array([lowh,lows,lowv])
        hsvHigh = np.array([highh,highs,highv])
    else:
        hsvLow = np.array([0,0,0])
        hsvHigh = np.array([0,0,0])

    # same as above if
    if not(lowb == None or lowg == None or lowr == None or highb == None or highg == None or highr == None):
        bgrLow = np.array([lowb,lowg,lowr])
        bgrHigh = np.array([highb,highg,highr])
    else:
        bgrLow = np.array([0,0,0])
        bgrHigh = np.array([0,0,0])

    # mask the hsv and frame images...
    # that means, make a binary (b&w) image
    # only showing white where the hsv (or bgr) values, are within the range
    mask = cv2.inRange(hsv, hsvLow, hsvHigh)
    mask2 = cv2.inRange(frame, bgrLow, bgrHigh)

    # do a bitwise and, meaning, only keep the white where both masks are white
    bw = cv2.bitwise_and(mask,mask2)

    # Display the resulting frame
    # if displaythreshold is true, it will display the mask, otherwise
    # it will display the actual frame
    if not displayThreshold:
        cv2.imshow('frame',frame)
    else:
        cv2.imshow('frame',bw)

    # wait for the q key to quit
    # only wait for 1 ms for the next frame
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything is done, release the capture
cap.release()
cv2.destroyAllWindows()
