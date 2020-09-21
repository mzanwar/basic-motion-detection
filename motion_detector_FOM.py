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
from matplotlib import pyplot as plt
import os

BASE = os.path.abspath('.')
VIDEO = BASE + '/videos/tennis_short.mp4'
IMG_TEMPLATE = './frames/tennis-frame-{}.png'



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


def extract_img(video, img_name):
	cap = cv2.VideoCapture(video)

	idx = 1

	while cap.isOpened():
		ret, frame = cap.read()
		cv2.imshow('tennis', frame)
		key = cv2.waitKey(0) & 0xFF
		if key == ord('q'):
			break
		elif key == ord('s'):
			img = img_name.format(idx)
			if not cv2.imwrite(img, frame):
				raise Exception("Can't save image")
			idx += 1

	cap.release()
	cv2.destroyAllWindows()


extract_img(VIDEO, IMG_TEMPLATE)

###############track bars ##########



# loop over the frames of the video

# resize the frame, convert it to grayscale, and blur it
images = [cv2.imread(IMG_TEMPLATE.format(idx), cv2.IMREAD_GRAYSCALE)
		  for idx in range(1, 412)]
fig = plt.figure(figsize=(18, 16), edgecolor='k')

for t in range(1, 409):
	plt.subplot(1, 411, t)
	plt.imshow(images[t], 'gray')
	plt.title("Frame t = {}".format(t))
	im_tm1 = images[t - 1]
	im_t = images[t]
	im_tp1 = images[t + 1]

	delta_plus = cv2.absdiff(im_t, im_tm1)
	delta_0 = cv2.absdiff(im_tp1, im_tm1)
	delta_minus = cv2.absdiff(im_t, im_tp1)

sp = cv2.meanStdDev(delta_plus)
sm = cv2.meanStdDev(delta_minus)
s0 = cv2.meanStdDev(delta_0)
print("E(d+):", sp, "\nE(d-):", sm, "\nE(d0):", s0)

th = [
	sp[0][0, 0] + 3 * math.sqrt(sp[1][0, 0]),
	sm[0][0, 0] + 3 * math.sqrt(sm[1][0, 0]),
	s0[0][0, 0] + 3 * math.sqrt(s0[1][0, 0]),
]


def combine(dbp, dbm, db0r):
	"""Combines the three binary images.

	If the corresponding pixel in all three images is non-zero, a "black" value is emitted
	in the result image, otherwise a "white" value.

	The resultant image should be a "negative" image of moving objects in the original frames.
	"""
	rows, columns = dbp.shape
	res = np.zeros((rows, columns), dtype="uint8")

	for row in range(rows):
		for col in range(columns):
			res[row, col] = 0 if dbp[row, col] > 0 and \
								 dbm[row, col] > 0 and \
								 db0r[row, col] > 0 else 255
	return res


start = time.time()
ret, dbp = cv2.threshold(delta_plus, th[0], 255, cv2.THRESH_BINARY)
ret, dbm = cv2.threshold(delta_minus, th[1], 255, cv2.THRESH_BINARY)
ret, db0 = cv2.threshold(delta_0, th[2], 255, cv2.THRESH_BINARY)

detect = cv2.bitwise_not(
	cv2.bitwise_and(cv2.bitwise_and(dbp, dbm),
					cv2.bitwise_not(db0)))

ocv_time = (time.time() - start) * 1000

fig = plt.figure(figsize=(24, 20), edgecolor='k')

titles = ["OpenCV ({:6.2f} msec)".format(ocv_time)]
for t, img in enumerate([detect,]):
	plt.subplot(1, 3, t + 1)
	plt.imshow(img, 'gray')
	plt.title(titles[t])
	cv2.imwrite(IMG_TEMPLATE.format(titles[t].split()[0]), img)

start = time.time()

# The original `detect` image was suitable for display, but it is "inverted" and not suitable
# for component detection; we need to invert it first.
nd = cv2.bitwise_not(detect)
num, labels, stats, centroids = cv2.connectedComponentsWithStats(nd, ltype=cv2.CV_16U)

plt.figure(figsize=(20, 30))

font = cv2.FONT_HERSHEY_SIMPLEX
fontScale = 0.6
fontColor = 127
lineType = 1

# We set an arbitrary threshold to screen out smaller "components"
# which may result simply from noise, or moving leaves, and other
# elements not of interest.
min_area = 500

d = detect.copy()
candidates = list()
for stat in stats:
	area = stat[cv2.CC_STAT_AREA]
	if area < min_area:
		continue  # Skip small objects (noise)

	lt = (stat[cv2.CC_STAT_LEFT], stat[cv2.CC_STAT_TOP])
	rb = (lt[0] + stat[cv2.CC_STAT_WIDTH], lt[1] + stat[cv2.CC_STAT_HEIGHT])
	bottomLeftCornerOfText = (lt[0], lt[1] - 15)

	candidates.append((lt, rb, area))
	cv2.rectangle(d, lt, rb, fontColor, lineType)

	cv2.putText(d, "{}: {:.0f}".format(len(candidates), stat[cv2.CC_STAT_AREA]),
				bottomLeftCornerOfText,
				font, fontScale, fontColor, lineType)

plt.title("Candidates: {} ({:.2f} msec)".format(len(candidates),
												(time.time() - start) * 1000))
plt.imshow(d, 'gray')
print("Found {} components".format(num - 1))
plt.show()
key = cv2.waitKey(0) & 0xFF