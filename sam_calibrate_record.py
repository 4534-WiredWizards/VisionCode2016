import numpy as np
import cv2

###
#
# sam_calibrate_record.py | A calibration recording program
#
# This program doesn't really have to do with calibration, it records
# the camera frames to a video file. It is not actually the .avi format, but
# you select a codec when you run it.
#
# Not important anymore, we have our test video.
#
###

### GLOBALS ###

frame = None
hsv = None
lowh = None
highh = None
lows = None
highs = None
lowv = None
highv = None
lowb = None
highb = None
lowg = None
highg = None
lowr = None
highr = None
displayThreshold = False
clickedYet = False


### END GLOBALS ###
### FUNCTIONS ###

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

def clickFunction(event, x, y, flags, param):

    global frame,hsv
    global lowh,highh,lows,highs,lowv,highv
    global lowb,highb,lowg,highg,lowr,highr
    global displayThreshold
    global clickedYet

    if event == cv2.EVENT_RBUTTONUP:
        displayThreshold = not displayThreshold

    if event == cv2.EVENT_LBUTTONUP:
        # get the hsv at the location
        h,s,v = hsv[y,x]
        b,g,r = frame[y,x]
        print ((y,x),(h,s,v))
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
        print "Current bounds low(h,s,v) high(h,s,v) low(b,g,r) high(b,g,r)"
        print ((lowh,lows,lowv),(highh,highs,highv),(lowb,lowg,lowr),(highb,highg,highr))

### END FUNCTIONS ###

# instantiate the video capture object
cap = cv2.VideoCapture(countCameras())

# set exposure
#cap.set(15, 0)

writer = cv2.VideoWriter();
size = (int(cap.get(3)),int(cap.get(4)))
print size
# cv2.VideoWriter_fourcc(*"XVID")
writer.open("C:/Users/Samuel/Documents/VisionCode2016/angles2.avi",-1,24.0,size)

cv2.namedWindow("frame")
cv2.setMouseCallback("frame",clickFunction)

displayThreshold = False

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
    print writer.isOpened()
    cap.set(15,-9)

    # capture each frame
    ret, frame = cap.read()

    writer.write(frame)

    # Our operations on the frame come here
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    if not(lowh == None or lows == None or lowv == None or highh == None or highs == None or highv == None):
        hsvLow = np.array([lowh,lows,lowv])
        hsvHigh = np.array([highh,highs,highv])
    else:
        hsvLow = np.array([0,0,0])
        hsvHigh = np.array([0,0,0])

    if not(lowb == None or lowg == None or lowr == None or highb == None or highg == None or highr == None):
        bgrLow = np.array([lowb,lowg,lowr])
        bgrHigh = np.array([highb,highg,highr])
    else:
        bgrLow = np.array([0,0,0])
        bgrHigh = np.array([0,0,0])

    mask = cv2.inRange(hsv, hsvLow, hsvHigh)
    mask2 = cv2.inRange(frame, bgrLow, bgrHigh)

    bw = cv2.bitwise_and(mask,mask2)

    # Display the resulting frame
    if not displayThreshold:
        cv2.imshow('frame',frame)
    else:
        cv2.imshow('frame',bw)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
writer.release()
cv2.destroyAllWindows()
