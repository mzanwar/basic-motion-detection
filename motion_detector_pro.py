# USAGE
# python motion_detector.py
# python motion_detector.py --video videos/example_01.mp4


from imutils.video import VideoStream
from imutils import contours
import argparse
import datetime
import imutils
import time
import cv2
import numpy as np

from utils import ObjectTracker, region_of_interest, VIDEO_SIZE, compute_difference_with_dilation_erosion, \
    find_all_contours

# Tweak these when ball found, or when search for ball
STARTING_THRESHOLD = 25
STARTING_INTERATIONS = 16

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video", help="path to the video file")
ap.add_argument("-a", "--min-area", type=int, default=500, help="minimum area size")
args = vars(ap.parse_args())

# if the video argument is None, then we are reading from webcam
if args.get("video", None) is None:
    vs = VideoStream(src=0).start()
    time.sleep(2.0)

# otherwise, we are reading from a video file
else:
    vs = cv2.VideoCapture(args["video"])

# initialize the first frame in the video stream
lastFrame = None


###############track bars ##########

trackbars = np.zeros((300, 512, 3), np.uint8)
cv2.namedWindow('Tennis')

def nothing(x):
    pass

cv2.createTrackbar('Threshold', 'Tennis', STARTING_THRESHOLD, 255, nothing)
cv2.createTrackbar('Iterations', 'Tennis', STARTING_INTERATIONS, 255, nothing)

object_tracker = ObjectTracker()
ball_found = False

while True:
    frame = vs.read()
    frame = frame if args.get("video", None) is None else frame[1]
    text = "Not Found"

    # if the frame could not be grabbed, then we have reached the end
    if frame is None:
        break

    # resize the frame, convert it to grayscale, and blur it
    frame = imutils.resize(frame, width=VIDEO_SIZE)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # crop out the top half of our image, hardcoded triangle
    gray = region_of_interest(gray)

    # blur gray
    gaussian_blur = 3
    gray = cv2.GaussianBlur(gray, (gaussian_blur, gaussian_blur), 0)

    # initialize last frame
    if lastFrame is None:
        lastFrame = gray

    threshold = cv2.getTrackbarPos('Threshold', 'Tennis')
    iterations = cv2.getTrackbarPos('Iterations', 'Tennis')
    thresh = compute_difference_with_dilation_erosion(currentFrame=gray, lastFrame=lastFrame, threshold=threshold, iterations=iterations)

    cnts = find_all_contours(thresh)

    # loop over the contours
    for idx, c in enumerate(cnts):

        object_tracker.add(c)
        contours.label_contour(frame, c, idx) # so much better than the circles

        (x, y, w, h) = cv2.boundingRect(c)
        moment = cv2.moments(c)
        area = cv2.contourArea(c)
        perimeter = cv2.arcLength(c, True)
        hull = str(cv2.convexHull(c, returnPoints=True))
        convex = cv2.isContourConvex(c)
        ratio = cv2.contourArea(c) / cv2.arcLength(c, True)

    cv2.putText(frame, "Ball Status: {}".format(text), (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    cv2.putText(frame, datetime.datetime.now().strftime("%A %d %B %Y %I:%M:%S%p"), (10, frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 255), 1)

    cv2.imshow("Gray", gray)
    cv2.imshow("Tennis", frame)
    cv2.imshow("Thresh", thresh)

    cv2.moveWindow("Tennis", VIDEO_SIZE, 0)
    cv2.moveWindow("Gray", 0, VIDEO_SIZE)
    cv2.moveWindow("Thresh", VIDEO_SIZE, VIDEO_SIZE)

    key = cv2.waitKey() & 0xFF

    # if the `q` key is pressed, break from the lop
    if key == ord("q"):
        break

    lastFrame = gray
    object_tracker.cleanup_stale_objects()
    object_tracker.find_ball()

# cleanup the camera and close any open windows
vs.stop() if args.get("video", None) is None else vs.release()
cv2.destroyAllWindows()

