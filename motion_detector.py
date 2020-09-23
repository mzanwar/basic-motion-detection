# USAGE
# python motion_detector.py
# python motion_detector.py --video videos/example_01.mp4

# import the necessary packages
import math

from imutils.video import VideoStream
from imutils import contours
import argparse
import datetime
import imutils
import time
import cv2
import numpy as np
from skimage import measure

# TODO: Add a counter and only return moving objects if they have been seen a few times


VIDEO_SIZE = 800
STALE_THRESHOLD = 20

def region_of_interest(image):
    height = image.shape[0]
    width = image.shape[1]

    polygons = np.array([[(0, 207), (width, 20), (width, height), (0, height)]])
    mask = np.zeros_like(image)
    cv2.fillPoly(mask, polygons, 255)
    masked_image = cv2.bitwise_and(image,mask)
    return masked_image


def scale_contour(cnt, scale):
    M = cv2.moments(cnt)
    cx = int(M['m10']/M['m00'])
    cy = int(M['m01']/M['m00'])

    cnt_norm = cnt - [cx, cy]
    cnt_scaled = cnt_norm * scale
    cnt_scaled = cnt_scaled + [cx, cy]
    cnt_scaled = cnt_scaled.astype(np.int32)

    return cnt_scaled

class TrackObjectsUsingContours:
    def __init__(self):
        self.objects = []
        self.count = 0

    def add(self, contour):
        # if ball not found, then search for ball
        for moving_object in self.objects:
            if moving_object.is_this_me(contour):
                moving_object.recalibrate_position(contour)
                return  # is this right? I've found myself, but could also be 1) another object

        # didn't find a match, add as a new object
        self.count += 1
        self.objects.append(MovingObject(contour=contour, id = self.count))

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
        potential_balls = self.most_traveled_objects()
        if potential_balls:
            if potential_balls[0].distance > 300:
                self.ball = potential_balls[0]
            else:
                self.ball = None

    def cleanup_stale_objects(self):
        # remove object after N number of frames
        self.objects = [obj for obj in self.objects if obj.stale < STALE_THRESHOLD]



class MovingObject:

    def __init__(self, contour, id):
        self.id = id
        self.contour = contour
        self.speed = 0
        self.distance = 0

        self.age = 0
        self.stale = 0

    @property
    def area(self):
        return cv2.contourArea(self.contour)

    @property
    def perimeter(self):
        return cv2.arcLength(self.contour, True)

    def is_this_me(self, new_contour):
        old_contour = self.contour
        overlap = self.contourIntersect(old_contour, new_contour)
        matching_shape = cv2.matchShapes(old_contour, new_contour, 1, 0.0) < 0.5
        similar_area_perimeter = self.is_similar_enough_in_area_perimeter(new_contour)
        found = overlap and matching_shape and similar_area_perimeter
        if not found:
            self.stale += 1
            print("ID {}: overlap: {}, shape: {}, similar_area: {}".format(self.id, overlap, matching_shape, similar_area_perimeter))
        else:
            # if found, increment age
            self.age += 1
            self.stale = 0
        return found

    def is_similar_enough_in_area_perimeter(self, new_contour):
        BUFFER = 1 # area or perimeter can be +/-
        new_area = cv2.contourArea(new_contour)
        new_perimeter = cv2.arcLength(self.contour, True)

        area_similar = MovingObject.within_bounds(new_area, self.area, BUFFER)
        perimeter_similar = MovingObject.within_bounds(new_perimeter, self.perimeter, BUFFER)

        return area_similar and perimeter_similar


    @staticmethod
    def calculate_lower_upper_bounds(number, buffer):
        lower = number * ( 1 - buffer)
        upper = number * ( 1 + buffer)
        return lower, upper

    @staticmethod
    def within_bounds(new_number, old_number, buffer):
        lower_bound, upper_bound = MovingObject.calculate_lower_upper_bounds(old_number, buffer)
        return (new_number >= lower_bound) and (new_number <= upper_bound)

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
        SCALE = 3 # increase size of intersection
        contour1 = scale_contour(contour1, SCALE)
        contour2 = scale_contour(contour2, SCALE)
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
cv2.createTrackbar('Threshold', 'trackbars', 25, 255, nothing)
cv2.createTrackbar('Iterations', 'trackbars', 16, 255, nothing)
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
    gray = region_of_interest(gray)
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
    dilatation_type = cv2.MORPH_ELLIPSE
    element = cv2.getStructuringElement(dilatation_type, (2 * iterations + 1, 2 * iterations + 1),
                                       (iterations, iterations))
    thresh = cv2.dilate(thresh, element)
    thresh = cv2.erode(thresh, None, iterations=3)
    cnts = cv2.findContours(thresh.copy(),
                            cv2.RETR_EXTERNAL,
                            cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    # cnts = contours.sort_contours(cnts)[0]


    # TODO: ADD UI drag elements to tweak all these settings on the fly

    # perform a connected component analysis on the thresholded
    # image, then initialize a mask to store only the "large"
    # components
    # labels = measure.label(thresh, neighbors=8, background=0)
    # mask = np.zeros(thresh.shape, dtype="uint8")
    # # loop over the unique components
    # for label in np.unique(labels):
    #     # if this is the background label, ignore it
    #     if label == 0:
    #         continue
    #     # otherwise, construct the label mask and count the
    #     # number of pixels
    #     labelMask = np.zeros(thresh.shape, dtype="uint8")
    #     labelMask[labels == label] = 255
    #     numPixels = cv2.countNonZero(labelMask)
    #     # if the number of pixels in the component is sufficiently
    #     # large, then add it to our mask of "large blobs"
    #     if numPixels > 300:
    #         mask = cv2.add(mask, labelMask)
    #
    # cv2.imshow("mask", mask)


    # loop over the contours
    for idx, c in enumerate(cnts):
        # if the contour is too small, ignore it
        # if cv2.contourArea(c) > max_size or cv2.contourArea(c) < min_size:
        #     continue

        object_tracker.add(c)
        contours.label_contour(frameDelta, c, idx)

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

        # TODO: use different properties of contours to isolate ball

        (x, y, w, h) = cv2.boundingRect(c)
        moment = cv2.moments(c)
        area = cv2.contourArea(c)
        perimeter = cv2.arcLength(c, True)
        hull = str(cv2.convexHull(c, returnPoints=True))
        convex = cv2.isContourConvex(c)
        ratio = cv2.contourArea(c) / cv2.arcLength(c, True)
        # print(idx, area, perimeter, ratio, len(hull))
        # if (ratio < 5.5 or ratio > 8.7) or (area > 1100):
        #     continue
        # cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        # cv2.putText(frame, str(idx) + " " + str(ratio) + " " + str(area), (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 255), 1)

        (x, y), radius = cv2.minEnclosingCircle(c)
        center = (int(x), int(y))
        radius = int(radius)
        img = cv2.circle(frame, center, radius, (0, 255, 0), 1)

    for obj in object_tracker.objects:
        if obj.age < 100:
            continue
        (x, y, w, h) = cv2.boundingRect(obj.contour)
        cv2.putText(frame, str(obj.id) + "  " + str(obj.distance),(x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 125, 125), 1)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 125, 125), 2)
    #
    # for obj in object_tracker.most_traveled_objects():
    #     (x, y, w, h) = cv2.boundingRect(obj.contour)
    #     distance = obj.distance
    #     cv2.putText(frame, str(distance),(x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 255), 1)
    #     cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 1)

    # ball = object_tracker.ball
    # if ball is not None:
    #     (x, y, w, h) = cv2.boundingRect(ball.contour)
    #     cv2.putText(frame, "ball", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 125, 125), 1)
    #     cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 125, 125), 2)

    # draw the text and timestamp on the frame
    cv2.putText(frame, "Room Status: {}".format(text), (10, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    cv2.putText(frame, datetime.datetime.now().strftime("%A %d %B %Y %I:%M:%S%p"),
                (10, frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 255), 1)

    # # Setup SimpleBlobDetector parameters.
    # params = cv2.SimpleBlobDetector_Params()
    #
    # # Change thresholds
    # params.minThreshold = 10
    # params.maxThreshold = 20
    #
    # # Filter by Area.
    # params.filterByArea = True
    # params.minArea = 0
    # params.maxArea = 100
    #
    # # Filter by Circularity
    # params.filterByCircularity = True
    # params.minCircularity = 0.87
    #
    # # Filter by Convexity
    # params.filterByConvexity = True
    # params.minConvexity = 0.2
    #
    # # Filter by Inertia
    # params.filterByInertia = True
    # params.minInertiaRatio = 0.5
    #
    # detector = cv2.SimpleBlobDetector_create(params)
    # keypoints = detector.detect(gray)
    # print(keypoints)

    # img_key_points = cv2.drawKeypoints(gray, keypoints, np.array([]), (0,0,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    #
    # cv2.imshow("KEYPOINTS", img_key_points)
    # cv2.moveWindow("KEYPOINTS", 1500, 800)

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

