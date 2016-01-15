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

    if event == cv2.EVENT_RBUTTONUP:
        displayThreshold = not displayThreshold

    if event == cv2.EVENT_LBUTTONUP:
        # get the hsv at the location
        h,s,v = hsv[x,y]
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
        if not lowh == None:
            lowh = h
        if not highh == None:
            highh = h
        if not lows == None:
            lows = s
        if not highs == None:
            highs = s
        if not lowv == None:
            lowv = v
        if not highv == None:
            highv = v
        print (lowh,lows,lowv,highh,highs,highv)

### END FUNCTIONS ###

# instantiate the video capture object
cap = cv2.VideoCapture(countCameras())
cv2.namedWindow("frame")
cv2.setMouseCallback("frame",clickFunction)

displayThreshold = False

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
