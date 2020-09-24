import cv2
import imutils
import numpy as np
import math

from imutils import contours

VIDEO_SIZE = 800

CONTOUR_SEARCH_TYPE = cv2.RETR_TREE
CONTOUR_SEARCH_POINTS = cv2.CHAIN_APPROX_SIMPLE

STALE_THRESHOLD = 15 # how many frames to wait before evicting objects
SHAPE_THRESHOLD = 0.1 # matchShape 0 is perfect match, Hu-Moments are seven moments invariant to translation, rotation and scale
SCALE_INTERSECT = 3 # increase size of intersection by scaling BOTH contours
SIMILAR_ENOUGH_BUFFER_PERCENTAGE = 2 # area or perimeter can be +/-  1 => 100%

def distance_between_two_contours(c1, c2):
    #between their center of masses
    M = cv2.moments(c1)
    if int(M['m00']) == 0:
        return 0
    c1x = int(M['m10']/M['m00'])
    c1y = int(M['m01']/M['m00'])

    M = cv2.moments(c2)
    if int(M['m00']) == 0:
        return 0
    c2x = int(M['m10']/M['m00'])
    c2y = int(M['m01']/M['m00'])

    return math.sqrt((c1x - c2x)**2 + (c1y - c2y)**2)


def region_of_interest(image):
    height = image.shape[0]
    width = image.shape[1]

    polygons = np.array([[(0, 207), (width, 20), (width, height), (0, height)]])
    mask = np.zeros_like(image)
    cv2.fillPoly(mask, polygons, 255)
    masked_image = cv2.bitwise_and(image, mask)
    return masked_image


def contourIntersect(contour1, contour2, scale=SCALE_INTERSECT):
    # Two separate contours trying to check intersection on
    contour1 = scale_contour(contour1, scale)
    contour2 = scale_contour(contour2, scale)
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

def region_of_interest_contour(contour, image):
    (x, y), radius = cv2.minEnclosingCircle(contour) # increase the size of the window to search in
    mask = np.zeros_like(image)
    # maybe ellipse in direction of trajectory
    cv2.circle(mask, (int(x), int(y)), int(radius * 3), (255, 255, 255), -1)
    # cv2.fillPoly(mask, pts=[cross_hairs], color=(255, 255, 255))
    cv2.imshow("Mask", mask)
    masked_image = cv2.bitwise_and(image, mask)
    cv2.imshow("Masked", masked_image)
    return masked_image


def scale_contour(cnt, scale):
    M = cv2.moments(cnt)
    if int(M['m00']) == 0:
        return cnt
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

def find_contours_around_ball(ball, image):
    # isolate image
    ball_window_cropped = region_of_interest_contour(contour=ball.contour, image=image)
    cnts = cv2.findContours(ball_window_cropped.copy(),
                            CONTOUR_SEARCH_TYPE,
                            CONTOUR_SEARCH_POINTS)
    cnts = imutils.grab_contours(cnts)

    return cnts


class ObjectTracker:
    def __init__(self):
        self.objects = []
        self.count = 0

        self.ball_was_found = False
        self.ball = None
        self.ball_found = False

        self.contours_to_be_objects = []

    def try_match_with_existing_objects(self, cnts, frame):
        # loop over the contours
        no_matches = []
        for idx, contour in enumerate(cnts):
            M = cv2.moments(contour)
            if int(M['m00']) == 0:
                continue
            contours.label_contour(frame, contour, idx)
            found_match = self.match_contour_to_objects(contour)
            if not found_match:
                no_matches.append(contour)

        self.resolve_potential_matches()
        self.cleanup_stale_objects()

        return no_matches

    def match_contour_to_objects(self, contour):
        found_matches = False
        for moving_object in self.objects:
            if moving_object.is_this_me(contour):
                moving_object.add_potential_match(contour)
                found_matches = True

        return found_matches

    def search_for_ball(self, where, frame):
        cnts = find_all_contours(where)
        no_matches_found = self.try_match_with_existing_objects(cnts, frame)
        self.contours_to_be_objects.extend(no_matches_found)

    def track_found_ball_per_frame(self, cnts):
        ball_tracked = False
        for contour in cnts:
            if self.ball.is_this_me(contour):
                self.ball.recalibrate_position(contour)
                ball_tracked = True
                break # on first hit

        if not ball_tracked:
            # Change state to Search for ball
            self.ball_found = False

    def search_for_missing_ball(self, thresh, frame):
        missing_ball_found = False
        cnts = find_contours_around_ball(self.ball, thresh)
        # loop over the contours
        for idx, c in enumerate(cnts):
            M = cv2.moments(c)
            if int(M['m00']) == 0:
                # bad contour, skip it
                continue
            if self.ball.is_this_me(c):
                self.ball.recalibrate_position(c)
                missing_ball_found = True
            contours.label_contour(frame, c, idx)  # so much better than the circles

    def make_unmatched_contours_objects(self):
        for cnt in self.contours_to_be_objects:
            self.add_object(cnt)
        self.contours_to_be_objects = []

    def add_object(self, contour):
        self.count += 1
        self.objects.append(MovingObject(contour=contour, id=self.count))

    def how_many_objects(self):
        return len(self.objects)

    def fastest_objects(self, limit=10):
        fastest = sorted(self.objects, key=lambda x: x.speed, reverse=True)
        return fastest[:limit]

    def most_traveled_objects(self, limit=10):
        longest_traveled = sorted(self.objects, key=lambda x: x.distance, reverse=True)
        return longest_traveled[:limit]

    def oldest_objects(self, limit=10):
        oldest = sorted(self.objects, key=lambda x: x.age, reverse=True)
        return oldest[:limit]

    def all_moving_objects(self):
        return self.objects

    def try_find_ball_from_objects(self, ball_state_machine):
        MAX_AREA_OF_BALL = 1000
        MIN_DISTANCE_TRAVELLED = 100
        MIN_AGE = 10

        most_travelled = self.most_traveled_objects(5)
        # oldest = self.oldest_objects(3)

        potential_balls = most_travelled
        potential_balls = list(filter(lambda x: x.area <= MAX_AREA_OF_BALL, potential_balls))
        potential_balls = list(filter(lambda x: x.distance >= MIN_DISTANCE_TRAVELLED, potential_balls))
        potential_balls = list(filter(lambda x: x.age >= MIN_AGE, potential_balls))

        potential_balls = sorted(potential_balls, key=lambda x: x.speed, reverse=True)

        if len(potential_balls) == 0:
            # No ball found
            self.ball_found = False
        elif len(potential_balls) > 0:
            ball_state_machine.found()
            if len(potential_balls) > 1:
                print("Multiple Balls Found, using [0] => ", [potential_ball.id for potential_ball in potential_balls])
            self.objects = [potential_balls[0]]
            self.ball = potential_balls[0]
            self.ball_found = True
            self.ball_was_found = True

    def cleanup_stale_objects(self):
        # remove object after N number of frames
        self.objects = [obj for obj in self.objects if obj.stale < STALE_THRESHOLD]

    def resolve_potential_matches(self):
        for obj in self.objects:
            coeffs = [(idx, obj.calculate_matching_coefficient(match)) for idx, match in enumerate(obj.potential_matches)]
            coeffs = sorted(coeffs, key=lambda x: x[1], reverse=False)
            print(coeffs)
            if not coeffs:
                return
            id = coeffs[0][0]
            obj.recalibrate_position(obj.potential_matches[id])
            obj.potential_matches = []

class MovingObject:
    def __init__(self, contour, id):
        self.id = id
        self.contour = contour
        self.speed = 0
        self.distance = 0

        self.age = 0
        self.stale = 0

        self.potential_matches = []

    @property
    def area(self):
        return cv2.contourArea(self.contour)

    @property
    def perimeter(self):
        return cv2.arcLength(self.contour, True)

    def is_this_me(self, new_contour):
        old_contour = self.contour
        overlap = contourIntersect(old_contour, new_contour, SCALE_INTERSECT)
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

    def calculate_matching_coefficient(self, new_contour):
        old_contour = self.contour
        proximity_coeff = distance_between_two_contours(new_contour, old_contour) / VIDEO_SIZE
        matching_shape_coeff = cv2.matchShapes(old_contour, new_contour, 1, 0.0)
        deltaArea, deltaPerimeter = self.get_delta_area_perimieter(new_contour)
        deltaArea /= 100
        deltaPerimeter /= 100
        coefficient = proximity_coeff + matching_shape_coeff + deltaArea + deltaPerimeter
        return coefficient

    def is_similar_enough_in_area_perimeter(self, new_contour):
        new_area = cv2.contourArea(new_contour)
        new_perimeter = cv2.arcLength(self.contour, True)

        area_similar = MovingObject.within_bounds(new_area, self.area, SIMILAR_ENOUGH_BUFFER_PERCENTAGE)
        perimeter_similar = MovingObject.within_bounds(new_perimeter, self.perimeter, SIMILAR_ENOUGH_BUFFER_PERCENTAGE)

        old_ratio = self.area / self.perimeter
        new_ratio = new_area / new_perimeter
        ratio_similar = MovingObject.within_bounds(new_ratio, old_ratio, SIMILAR_ENOUGH_BUFFER_PERCENTAGE)

        return area_similar and perimeter_similar and ratio_similar

    def get_delta_area_perimieter(self, new_contour):
        delta_area = abs(cv2.contourArea(new_contour) - cv2.contourArea(self.contour))
        delta_perimeter = abs(cv2.arcLength(new_contour, True) - cv2.arcLength(self.contour, True))
        return delta_area, delta_perimeter

    @staticmethod
    def calculate_lower_upper_bounds(number, buffer):
        lower = number * ( 1 - buffer)
        upper = number * ( 1 + buffer)
        return lower, upper

    @staticmethod
    def within_bounds(new_number, old_number, buffer):
        lower_bound, upper_bound = MovingObject.calculate_lower_upper_bounds(old_number, buffer)
        return (new_number >= lower_bound) and (new_number <= upper_bound)

    def add_potential_match(self, contour):
        self.potential_matches.append(contour)

    def recalibrate_position(self, contour):
        # add to list of potential contours
        prev_contour = self.contour
        (x, y, _, _) = cv2.boundingRect(prev_contour)
        (x2, y2, _, _) = cv2.boundingRect(contour)

        self.contour = contour

        new_speed = math.sqrt(abs(x2 - x) ** 2 + abs(y2 - y) ** 2)
        self.speed = (new_speed + self.speed) / 2
        self.distance += new_speed
