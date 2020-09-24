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
from statemachine import StateMachine, State

from utils import ObjectTracker, region_of_interest, VIDEO_SIZE, compute_difference_with_dilation_erosion, \
     find_contours_around_ball

# Tweak these when ball found, or when search for ball
STARTING_THRESHOLD = 25
STARTING_INTERATIONS = 16

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video", help="path to the video file")
ap.add_argument("-a", "--min-area", type=int, default=500, help="minimum area size")
args = vars(ap.parse_args())


class BallMachine(StateMachine):
    text = "Searching"

    # States
    searching = State('Searching', initial=True)
    tracking = State('Track')
    scanning = State('Scanning') # find recently lost ball

    # Transitions
    ball_lost = scanning.to(searching)

    found = searching.to(tracking) | scanning.to(tracking)
    not_found = searching.to(searching) | tracking.to(scanning)

    def on_enter_searching(self):
        self.text = "Searching"

    def on_enter_tracking(self):
        self.text = "Tracking"

    def on_enter_scanning(self):
        self.text = "Scanning"

# if the video argument is None, then we are reading from webcam
if args.get("video", None) is None:
    vs = VideoStream(src=0).start()
    time.sleep(2.0)

# otherwise, we are reading from a video file
else:
    vs = cv2.VideoCapture(args["video"])

# initialize the lastFrame
lastFrame = None

def nothing(x):
    pass


trackbars = np.zeros((300, 512, 3), np.uint8)
cv2.namedWindow('Tennis')
cv2.createTrackbar('Threshold', 'Tennis', STARTING_THRESHOLD, 255, nothing)
cv2.createTrackbar('Iterations', 'Tennis', STARTING_INTERATIONS, 255, nothing)

object_tracker = ObjectTracker()

bstm = BallMachine()

while True:
    frame = vs.read()
    frame = frame if args.get("video", None) is None else frame[1]

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

    cnts = None

    if bstm.is_searching:
        object_tracker.search_for_ball(where=thresh, frame=frame)
    elif bstm.is_scanning:
        object_tracker.search_for_missing_ball(thresh, frame)
    elif bstm.is_tracking:
        cnts = find_contours_around_ball(object_tracker.ball, thresh)
        object_tracker.track_found_ball_per_frame(cnts)

    for obj in object_tracker.objects:
        c = obj.contour
        (x, y), radius = cv2.minEnclosingCircle(c)
        center = (int(x), int(y))
        radius = int(radius)
        img = cv2.circle(frame, center, radius, (0, 255, 255), 1)
        cv2.putText(frame, str(obj.id), (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

    if object_tracker.ball:
        c = object_tracker.ball.contour
        (x, y), radius = cv2.minEnclosingCircle(c)
        center = (int(x), int(y))
        radius = int(radius)
        img = cv2.circle(frame, center, radius, (0, 0, 255), -1)
        cv2.putText(frame, "ball", (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 0)

    cv2.putText(frame, "Ball Status: {}".format(bstm.text), (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
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
    if not bstm.is_tracking:
        object_tracker.make_unmatched_contours_objects()
        object_tracker.try_find_ball_from_objects(bstm)


# cleanup the camera and close any open windows
vs.stop() if args.get("video", None) is None else vs.release()
cv2.destroyAllWindows()

