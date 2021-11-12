import cv2 
import dlib
import numpy as np
from swapFaceTranformation import swap, resize

cap = cv2.VideoCapture(0)

pathShapePredictor = r"E:\Models\Dlib\shape_predictor_68_face_landmarks.dat"
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(pathShapePredictor)

referenceImage = cv2.imread("images/rock.jpg")
referenceImage = resize(referenceImage, width=500)
grayReferenceImage = cv2.cvtColor(referenceImage, cv2.COLOR_BGR2GRAY)

while True:
	ret, frame = cap.read()
	frame = resize(frame, width=500)
	
	try:
		# output = swap(frame, referenceImage, detector, predictor)

		# Reverse swaping
		output = swap(referenceImage, frame, detector, predictor)
	except Exception as e:
		output = frame.copy()

	cv2.imshow("output", output)

	if cv2.waitKey(1) == ord('q'):
		break

cap.release()
cv2.destroyAllWindows()
