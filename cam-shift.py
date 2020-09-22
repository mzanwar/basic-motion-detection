import imutils
import numpy as np
import cv2

cap = cv2.VideoCapture('./videos/tennis-trimmed.mp4')

# take first frame of the video
ret,frame = cap.read()


def threshold_image(frame, lastGrayFrame, threshold, iterations):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gaussian_blur = 3
    gray = cv2.GaussianBlur(gray, (gaussian_blur, gaussian_blur), 0)
    frameDelta = cv2.absdiff(lastGrayFrame, gray)
    thresh = cv2.threshold(frameDelta,
                           threshold,  # how big the squares are
                           255,
                           cv2.THRESH_BINARY)[1]
    thresh = cv2.dilate(thresh, None, iterations=iterations)
    return thresh

def click_and_crop(event, x, y, flags, param):
    # grab references to the global variables
    global refPt, cropping
    # if the left mouse button was clicked, record the starting
    # (x, y) coordinates and indicate that cropping is being
    # performed
    if event == cv2.EVENT_LBUTTONDOWN:
        refPt = [(x, y)]
        cropping = True
    # check to see if the left mouse button was released
    elif event == cv2.EVENT_LBUTTONUP:
        # record the ending (x, y) coordinates and indicate that
        # the cropping operation is finished
        refPt.append((x, y))
        cropping = False
        # draw a rectangle around the region of interest
        cv2.rectangle(frame, refPt[0], refPt[1], (0, 255, 0), 1)
        print(refPt[0], refPt[1])
        cv2.imshow("image", frame)


# setup initial location of window
r,h,c,w = 180,70,999,50  # simply hardcoded the values
track_window = (c,r,w,h)

# set up the ROI for tracking
roi = frame[r:r+h, c:c+w]
hsv_roi =  cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
mask = cv2.inRange(hsv_roi, np.array((0., 60.,32.)), np.array((180.,255.,255.)))
roi_hist = cv2.calcHist([hsv_roi],[0],mask,[180],[0,180])
cv2.normalize(roi_hist,roi_hist,0,255,cv2.NORM_MINMAX)
cv2.imshow("roi_hist", roi_hist)

# Setup the termination criteria, either 10 iteration or move by atleast 1 pt
term_crit = ( cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1 )


while(1):
    ret ,frame = cap.read()

    if ret == True:
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        dst = cv2.calcBackProject([hsv],[0],roi_hist,[0,180],1)
        cv2.imshow("dst", dst)
        # apply meanshift to get the new location
        ret, track_window = cv2.meanShift(dst, track_window, term_crit)

        # Draw it on image
        x,y,w,h = track_window
        img2 = cv2.rectangle(frame, (x,y), (x+w,y+h), 255,1)
        cv2.imshow('img2',img2)
        cv2.namedWindow("img2")
        cv2.setMouseCallback("img2", click_and_crop)

        k = cv2.waitKey() & 0xff
        if k == 27:
            break
        else:
            cv2.imwrite(chr(k)+".jpg",img2)

    else:
        break

cv2.destroyAllWindows()
cap.release()