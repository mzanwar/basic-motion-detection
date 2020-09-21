# USAGE
# python motion_detector.py
# python motion_detector.py --video videos/example_01.mp4

# import the necessary packages
import math

from imutils.video import VideoStream
import argparse
import datetime
import imutils
import time
import cv2
import numpy as np

# TODO: Add a counter and only return moving objects if they have been seen a few times


VIDEO_SIZE = 800
STALE_THRESHOLD = 6

class TrackObjectsUsingContours:
    def __init__(self):
        self.objects = []
        self.ball = None

    def add(self, contour):
        if self.ball:
            if self.ball.is_this_me(contour):
                self.ball.recalibrate_position(contour)
            return
        # if ball not found, then search for ball
        for moving_object in self.objects:
            if moving_object.is_this_me(contour):
                moving_object.recalibrate_position(contour)
                return  # is this right? I've found myself, but could also be 1) another object

        # didn't find a match, add as a new object
        self.objects.append(MovingObject(contour=contour))

    def how_many_objects(self):
        return len(self.objects)

    def fastest_objects(self):
        fastest = sorted(self.objects, key=lambda x: x.speed, reverse=True)
        return fastest[:2]

    def most_traveled_objects(self):
        longest_traveled = sorted(self.objects, key=lambda x: x.distance, reverse=True)
        return longest_traveled[:2]

    def all_moving_objects(self):
        return self.objects

    def find_ball(self):
        if self.ball:
            return
        potential_balls = self.most_traveled_objects()
        if potential_balls:
            if potential_balls[0].distance > 300:
                self.ball = potential_balls[0]
            else:
                self.ball = None

    def cleanup_stale_objects(self):
        self.objects = [obj for obj in self.objects if obj.stale < STALE_THRESHOLD]



class MovingObject:

    def __init__(self, contour):
        self.contour = contour
        self.speed = 0
        self.distance = 0

        self.stale = 0

    def is_this_me(self, contour):
        found = self.contourIntersect(self.contour, contour)
        if not found:
            self.stale += 1
        else:
            self.stale = 0
        return found

    def recalibrate_position(self, contour):
        prev_contour = self.contour
        (x, y, _, _) = cv2.boundingRect(prev_contour)
        (x2, y2, _, _) = cv2.boundingRect(contour)

        self.contour = contour

        new_speed = math.sqrt(abs(x2 - x) ** 2 + abs(y2 - y) ** 2)
        self.speed = (new_speed + self.speed) / 2
        self.distance += new_speed


    def contourIntersect(self,contour1, contour2):
        # Two separate contours trying to check intersection on
        contours = [contour1, contour2]

        # Create image filled with zeros the same size of original image
        blank = np.zeros((450, VIDEO_SIZE))

        # Copy each contour into its own image and fill it with '1'
        image1 = cv2.drawContours(blank.copy(), contours, 0, 1)
        image2 = cv2.drawContours(blank.copy(), contours, 1, 1)

        # Use the logical AND operation on the two images
        # Since the two images had bitwise and applied to it,
        # there should be a '1' or 'True' where there was intersection
        # and a '0' or 'False' where it didnt intersect
        intersection = np.logical_and(image1, image2)

        # Check if there was a '1' in the intersection
        return intersection.any()



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
firstFrame = None
lastFrame = None


###############track bars ##########

def nothing(x):
    pass


trackbars = np.zeros((300, 512, 3), np.uint8)
cv2.namedWindow('trackbars')

# create trackbars for color change
cv2.createTrackbar('Threshold', 'trackbars', 20, 255, nothing)
cv2.createTrackbar('Iterations', 'trackbars', 20, 255, nothing)
cv2.createTrackbar('min_size', 'trackbars', 50, 1000, nothing)
cv2.createTrackbar('max_size', 'trackbars', 1000, 1000, nothing)

object_tracker = TrackObjectsUsingContours()

# loop over the frames of the video
while True:
    # grab the current frame and initialize the occupied/unoccupied
    # text

    threshold = cv2.getTrackbarPos('Threshold', 'trackbars')
    iterations = cv2.getTrackbarPos('Iterations', 'trackbars')
    min_size = cv2.getTrackbarPos('min_size', 'trackbars')
    max_size = cv2.getTrackbarPos('max_size', 'trackbars')

    frame = vs.read()
    frame = frame if args.get("video", None) is None else frame[1]
    text = "Unoccupied"

    # if the frame could not be grabbed, then we have reached the end
    # of the video
    if frame is None:
        break

    # resize the frame, convert it to grayscale, and blur it
    frame = imutils.resize(frame, width=VIDEO_SIZE)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gaussian_blur = 3
    gray = cv2.GaussianBlur(gray, (gaussian_blur, gaussian_blur), 0)

    if lastFrame is None:
        lastFrame = gray

    # compute the absolute difference between the current frame and
    # first frame
    frameDelta = cv2.absdiff(lastFrame, gray)
    thresh = cv2.threshold(frameDelta,
                           threshold,  # how big the squares are
                           255,
                           cv2.THRESH_BINARY)[1]

    # dilate the thresholded image to fill in holes, then find contours
    # on thresholded image
    thresh = cv2.dilate(thresh, None, iterations=iterations)
    cnts = cv2.findContours(thresh.copy(),
                            cv2.RETR_EXTERNAL,
                            cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)

    # TODO: ADD UI drag elements to tweak all these settings on the fly

    # loop over the contours
    for c in cnts:
        # if the contour is too small, ignore it
        # if cv2.contourArea(c) > max_size or cv2.contourArea(c) < min_size:
        #     continue

        object_tracker.add(c)

        # # compute the bounding box for the contour, draw it on the frame,
        # # and update the text
        # (x, y, w, h) = cv2.boundingRect(c)
        # crop_img = frame[y:y + h, x: x + w]
        # greenLower = (29, 86, 6)
        # greenUpper = (64, 255, 255)
        # thresh2 = cv2.inRange(crop_img, greenLower, greenUpper)
        # average = cv2.mean(thresh2)[0]
        # if not average == 0:  # meaning its green, doesn't work so well
        #     cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        #     text = "Ball Found"
        # else:
        #     cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
        #     text = "Ball not found"


        # (x, y, w, h) = cv2.boundingRect(c)
        # cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

    # for obj in object_tracker.fastest_objects():
    #     (x, y, w, h) = cv2.boundingRect(obj.contour)
    #     speed = obj.speed
    #     cv2.putText(frame, str(speed),(x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 125, 125), 1)
    #     cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 125, 125), 2)
    #
    # for obj in object_tracker.most_traveled_objects():
    #     (x, y, w, h) = cv2.boundingRect(obj.contour)
    #     distance = obj.distance
    #     cv2.putText(frame, str(distance),(x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 255), 1)
    #     cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 1)

    ball = object_tracker.ball
    if ball is not None:
        (x, y, w, h) = cv2.boundingRect(ball.contour)
        cv2.putText(frame, "ball", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 125, 125), 1)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 125, 125), 2)

    # draw the text and timestamp on the frame
    cv2.putText(frame, "Room Status: {}".format(text), (10, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    cv2.putText(frame, datetime.datetime.now().strftime("%A %d %B %Y %I:%M:%S%p"),
                (10, frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 255), 1)

    # show the frame and record if the user presses a key

    cv2.imshow("Gray", gray)
    cv2.imshow("Security Feed", frame)
    cv2.imshow("Thresh", thresh)
    cv2.imshow("Frame Delta", frameDelta)
    cv2.imshow("trackbars", trackbars)

    cv2.moveWindow("Security Feed", 1000, 0)
    cv2.moveWindow("Gray", 0, 0)
    cv2.moveWindow("Thresh", 1000, 1000)
    cv2.moveWindow("Frame Delta", 2000, 0)

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


