import numpy as np
import cv2

### GLOBALS ###

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

def clickFunc(evt,x,y,flags,param):
    global displayThreshold

    if evt == cv2.EVENT_LBUTTONDOWN:
        displayThreshold = not displayThreshold

### END FUNCTIONS ###

# instantiate the video capture object
cap = cv2.VideoCapture(countCameras())

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
    cap.set(15,-15);

    # capture each frame
    ret, frame = cap.read()

    #cap.set(cv2.CAP_PROP_FPS,12)
    #print cap.get(cv2.CAP_PROP_FPS)

    '''
    if frame is None:
        cap.set(cv2.CAP_PROP_POS_FRAMES,0)
        continue;
    '''

    # Our operations on the frame come here
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    hsvLow = np.array([60,255,10])
    hsvHigh = np.array([68,255,64])
    bgrLow = np.array([0,10,0])
    bgrHigh = np.array([7,64,0])

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
        cv2.drawContours(frame, [contours[largestAreaIndex]], -1, (255,0,0), 3)
    
    # Display the resulting frame
    if not displayThreshold:
        cv2.imshow('frame',frame)
    else:
        cv2.imshow('frame',bw)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
