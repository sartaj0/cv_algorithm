import cv2
import dlib
import numpy as np 
from PIL import Image

import skimage
from skimage import transform
from skimage.color import rgb2gray, gray2rgb

def resize(image, width=None, height=None):
	if (width is None) & (height is None):
		raise Exception("Height and Width npth are None")
	elif (width is not None) & (height is not None):
		raise Exception("You haved passed npth Height and Width both value")
	elif (width is not None) & (height is None):
		h, w, c = image.shape
		height = int((h / w) * width)
		return cv2.resize(image, (width, height))
	elif (width is None) & (height is not None):
		h, w, c = image.shape
		width = int((w / h) * height)
		return cv2.resize(image, (width, height))


def extractKeypoint(gray, detector, predictor):
	faces = detector(gray)
	for face in faces:
		landmarks = predictor(gray, face)
		landmarks_points = []
		for n in range(0, 68):
			x = landmarks.part(n).x
			y = landmarks.part(n).y
			landmarks_points.append((x, y))

	return landmarks_points


def drawKeypoints(image, keypoints, label="drawedKeypoints"):
	drawedKeypoints = image.copy()
	for point in keypoints:
		drawedKeypoints = cv2.circle(drawedKeypoints, point, 3, (0, 0, 0), -1)
	cv2.imshow(label, drawedKeypoints)
	# cv2.imwrite(label+".jpg", drawedKeypoints)



def tranform(image1, image2, kps1, kps2):
	model_robust = transform.estimate_transform("projective", 
		np.array(kps1), np.array(kps2))

	# image1_warped = transform.warp(image1, model_robust.inverse, output_shape=output_shape)
	image1_warped = transform.warp(image1, model_robust.inverse, output_shape=image2.shape[:2])
	image1_warped = np.round(image1_warped * 255).astype(np.uint8)

	return image1_warped
	

def swap(image1, image2, detector, predictor):
	gray1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
	gray2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)

	keypoints1 = extractKeypoint(gray1, detector, predictor)
	keypoints2 = extractKeypoint(gray2, detector, predictor)

	image1_warped = tranform(image1, image2, keypoints1, keypoints2)

	mask = np.zeros(image2.shape[:2], dtype=np.uint8)
	pts = keypoints2[:17] + keypoints2[17:27][::-1]
	mask = cv2.fillPoly(mask.copy(), pts=np.int32([pts]), color=(255, 255, 255), lineType=cv2.LINE_AA)

	mask1_bit = cv2.bitwise_and(image1_warped.copy(), image1_warped.copy(), mask=mask.copy())

	# mask2_inv = cv2.fillPoly(image2.copy(), pts=np.int32([pts]), color=(0, 0, 0), lineType=cv2.LINE_AA)
	# final1 = mask1_bit + mask2_inv

	mask_inv = cv2.bitwise_not(mask)
	face1 = cv2.merge([mask_inv, mask_inv, mask_inv])
	face1 = cv2.bitwise_or(face1, mask1_bit)
	body = cv2.fillPoly(image2.copy(), pts=np.int32([pts]), color=(255, 255, 255), lineType=cv2.LINE_AA)

	final2 = cv2.bitwise_and(body, face1)

	(x, y, w, h) = cv2.boundingRect(np.int32([pts]))
	center = (int((x + x + w) / 2), int((y + y + h) / 2))
	final = cv2.seamlessClone(final2, image2, mask, center, cv2.NORMAL_CLONE)

	# drawKeypoints(image1, keypoints1, label="kps1")
	# drawKeypoints(image2, keypoints2, label="kps2")

	# cv2.imwrite("transformed_face.jpg", image1_warped)

	# cv2.imwrite("croped_face.jpg", face1)
	# cv2.imwrite("cropped_body.jpg", body)

	# cv2.imwrite("pasted_image.jpg", final2)

	return final

if __name__ == '__main__':

	width = 500
	pathShapePredictor = r"E:\Models\Dlib\shape_predictor_68_face_landmarks.dat"
	
	image1 = cv2.imread("images/rock.jpg")
	image2 = cv2.imread("images/will_smith.jpg")

	image1 = resize(image1, width=width)
	image2 = resize(image2, width=width)

	detector = dlib.get_frontal_face_detector()
	predictor = dlib.shape_predictor(pathShapePredictor)

	final = swap(image1, image2, detector, predictor)

	cv2.imshow("final", final)
	cv2.imwrite("final.jpg", final)
	cv2.waitKey(0)