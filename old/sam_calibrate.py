import numpy as np
import cv2

### GLOBALS ###

frame = None
hsv = None
lowh = None
highh = None
lows = None
highs = None
lowv = None
highv = None
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
    global displayThreshold
    global clickedYet

    if event == cv2.EVENT_RBUTTONUP:
        displayThreshold = not displayThreshold

    if event == cv2.EVENT_LBUTTONUP:
        # get the hsv at the location
        h,s,v = hsv[y,x]
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
        if not clickedYet:
            clickedYet = True
            lowh = h
            highh = h
            lows = s
            highs = s
            lowv = v
            highv = v
        print "Current bounds (lowh,lows,lowv,highh,highs,highv)"
        print (lowh,lows,lowv,highh,highs,highv)

### END FUNCTIONS ###

# instantiate the video capture object
cap = cv2.VideoCapture(countCameras())
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
    # capture each frame
    ret, frame = cap.read()

    # Our operations on the frame come here
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    if not(lowh == None or lows == None or lowv == None or highh == None or highs == None or highv == None):
        low = np.array([lowh,lows,lowv])
        high = np.array([highh,highs,highv])
    else:
        low = np.array([0,0,0])
        high = np.array([0,0,0])

    mask = cv2.inRange(hsv, low, high);

    bw = mask

    # Display the resulting frame
    if not displayThreshold:
        cv2.imshow('frame',hsv)
    else:
        cv2.imshow('frame',bw)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
