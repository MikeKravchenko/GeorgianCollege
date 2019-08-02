import numpy as np
import cv2

# some constants
from imutils import face_utils
import dlib
import wget

from os import listdir
from os.path import isfile, join, sep

samples_dir = "cropped"
output_dir = "marked"
shape_predictor = "shape_predictor_68_face_landmarks.dat"

# ok, let's try make it simpler to use
link = f"https://github.com/AKSHAYUBHAT/TensorFace/blob/master/openface/models/dlib/{shape_predictor}?raw=True"
if not isfile(shape_predictor):
	wget.download(link)

def detect_and_show_parts(image_name, output_dir, parts):
	image = cv2.imread(image_name)
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	rects = detector(gray, 1)

	for (i, rect) in enumerate(rects):
		shape = predictor(gray, rect)
		shape = face_utils.shape_to_np(shape)
		for (name, (i, j)) in face_utils.FACIAL_LANDMARKS_IDXS.items():
			if name in parts:
				if name.endswith("_eye"):
					color = (0,255,0)
				else:
					color = (255,0,0)
				(x, y, w, h) = cv2.boundingRect(np.array([shape[i:j]]))
				cv2.rectangle(image,(x-10,y-10),(x+w+10,y+h+10),color,2)

		cv2.imwrite(join(output_dir, image_name.split(sep)[1]), image)

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(shape_predictor)

need_to_show = [
	'right_eye',
	'left_eye',
	'jaw',
]
onlyfiles = [f for f in listdir(samples_dir) if isfile(join(samples_dir, f))]
for f in onlyfiles:
	detect_and_show_parts(join(samples_dir, f), output_dir, need_to_show)
