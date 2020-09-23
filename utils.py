import cv2
import imutils
import numpy as np
import math

VIDEO_SIZE = 800

CONTOUR_SEARCH_TYPE = cv2.RETR_TREE
CONTOUR_SEARCH_POINTS = cv2.CHAIN_APPROX_SIMPLE

STALE_THRESHOLD = 20 # how many frames to wait before evicting objects
SHAPE_THRESHOLD = 0.5 # matchShape 0 is perfect match, Hu-Moments are seven moments invariant to translation, rotation and scale
SCALE_INTERSECT = 3 # increase size of intersection by scaling BOTH contours
SIMILAR_ENOUGH_BUFFER_PERCENTAGE = 1 # area or perimeter can be +/-  1 => 100%


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


def compute_difference_with_dilation_erosion(lastFrame, currentFrame, threshold, iterations):
    frameDelta = cv2.absdiff(lastFrame, currentFrame)
    thresh = cv2.threshold(frameDelta,
                           threshold,  # how granular
                           255,
                           cv2.THRESH_BINARY)[1]

    # dilate the thresholded image to fill in holes, then find contours
    dilatation_type = cv2.MORPH_ELLIPSE
    element = cv2.getStructuringElement(dilatation_type, (2 * iterations + 1, 2 * iterations + 1),
                                        (iterations, iterations))
    thresh = cv2.dilate(thresh, element)
    thresh = cv2.erode(thresh, None, iterations=3)

    return thresh


def find_all_contours(image):
    # External points, or nested structures, play around with tree if it improves accuracy
    cnts = cv2.findContours(image.copy(),
                            CONTOUR_SEARCH_TYPE,
                            CONTOUR_SEARCH_POINTS)
    cnts = imutils.grab_contours(cnts)

    return cnts


class ObjectTracker:
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
        matching_shape = cv2.matchShapes(old_contour, new_contour, 1, 0.0) < SHAPE_THRESHOLD
        similar_area_perimeter = self.is_similar_enough_in_area_perimeter(new_contour)
        found = overlap and matching_shape and similar_area_perimeter
        if not found:
            self.stale += 1
        else:
            # if found, increment age
            self.age += 1
            self.stale = 0
        return found

    def is_similar_enough_in_area_perimeter(self, new_contour):
        new_area = cv2.contourArea(new_contour)
        new_perimeter = cv2.arcLength(self.contour, True)

        area_similar = MovingObject.within_bounds(new_area, self.area, SIMILAR_ENOUGH_BUFFER_PERCENTAGE)
        perimeter_similar = MovingObject.within_bounds(new_perimeter, self.perimeter, SIMILAR_ENOUGH_BUFFER_PERCENTAGE)

        old_ratio = self.area / self.perimeter
        new_ratio = new_area / new_perimeter
        ratio_similar = MovingObject.within_bounds(new_perimeter, self.perimeter, SIMILAR_ENOUGH_BUFFER_PERCENTAGE)

        return area_similar and perimeter_similar and ratio_similar

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
        contour1 = scale_contour(contour1, SCALE_INTERSECT)
        contour2 = scale_contour(contour2, SCALE_INTERSECT)
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