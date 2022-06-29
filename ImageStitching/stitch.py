import os, cv2
import numpy as np
from PIL import Image

from skimage.feature import ORB
from skimage.feature import SIFT
# from skimage.feature import SURF
from skimage.transform import warp
from skimage.measure import ransac
from skimage.color import rgb2gray
from skimage.feature import match_descriptors
from skimage.transform import EuclideanTransform
from skimage.transform import ProjectiveTransform
from skimage.transform import SimilarityTransform 


def cropImage(image):
	gray = cv2.cvtColor(image.copy(), cv2.COLOR_BGR2GRAY)
	_,thresh = cv2.threshold(gray,5,255,cv2.THRESH_BINARY)
	kernel = np.ones((5,5), np.uint8)
	thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
	thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

	contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
	c = max(contours, key=cv2.contourArea)
	(x, y, w, h) = cv2.boundingRect(c)
	return image[y: y+h, x: x+w]


def resize(image, width=None, height=None):
	h, w = image.size
	if (width is None) & (height is None):
		raise Exception("Height and Width npth are None")
	elif (width is not None) & (height is not None):
		raise Exception("You haved passed npth Height and Width both value")
	elif (width is not None) & (height is None):
		height = int((h / w) * width)
	elif (width is None) & (height is not None):
		width = int((w / h) * height)
	return image.resize((height, width))


def featureDetectKeypointsDescriptors(featureDescriptors, image):
	featureDescriptors.detect_and_extract(image)
	keypoints = featureDescriptors.keypoints
	descriptors = featureDescriptors.descriptors

	return keypoints, descriptors


def matchDescriptor(descriptors1, descriptors2):
	matches = match_descriptors(descriptors1, descriptors2, cross_check=True)
	return matches


def ransacRemoveOutliers(keypoints1, keypoints2, matches):
	src = keypoints1[matches[:, 0]][:, ::-1]
	dst = keypoints2[matches[:, 1]][:, ::-1]

	model_robust, inliers = ransac((src, dst), ProjectiveTransform, min_samples=4, residual_threshold=1, max_trials=300)

	return model_robust, inliers

def wraping(image1, image2, model_robust):
	r, c = image1.shape[:2]
	corners = np.array([[0, 0], 
		[0, r], 
		[c, 0], 
		[c, r]])
	warped_corners = model_robust(corners)
	all_corners = np.vstack((warped_corners, corners))

	corner_min = np.min(all_corners, axis=0)
	corner_max = np.max(all_corners, axis=0)
	output_shape = np.round((corner_max - corner_min))[::-1]

	print(output_shape, corner_min, corner_max)

	offset = SimilarityTransform(translation= -corner_min)
	# offset = EuclideanTransform(translation= -corner_min)

	transform = (model_robust + offset)
	image1_warped = warp(image1, transform.inverse, order=3, output_shape=output_shape, cval=-1)
	image2_wraped = warp(image2, offset.inverse, order=3, output_shape=output_shape, cval=-1)

	# print(corner_min, offset.inverse, transform)
	# print(warped_corners, transform(corners))
	# print(corner_min, np.min(np.vstack((transform(corners), corners)), axis=0))
	# print(corner_max, np.max(np.vstack((transform(corners), corners)), axis=0))

	image1_mask = (image1_warped != -1)
	image1_warped[~image1_mask] = 0

	image2_mask = (image2_wraped != -1)
	image2_wraped[~image2_mask] = 0

	# image2_mask2 = image2_mask.copy()
	image2_mask = np.where(image2_mask == True, 1.0, image2_mask)
	image2_mask = np.where(image2_mask == False, 0.0, image2_mask)

	merged = (image1_warped + image2_wraped)

	overlap = (image1_mask * 1.0 + image2_mask)
	# cv2.imshow("overlap", overlap / 2)
	# cv2.imshow("output", cv2.cvtColor(output, cv2.COLOR_BGR2RGB))

	normalized = merged / np.maximum(overlap, 1)

	normalized = np.round(normalized * 255).astype(np.uint8)
	normalized = cropImage(normalized)

	return normalized


def process(image1, image2):

	gray1 = rgb2gray(np.array(image1))
	gray2 = rgb2gray(np.array(image2))

	# featureDescriptors = ORB(n_keypoints=400, fast_threshold=0.05)
	featureDescriptors = SIFT()

	keypoints1, descriptors1 = featureDetectKeypointsDescriptors(featureDescriptors, gray1)
	keypoints2, descriptors2 = featureDetectKeypointsDescriptors(featureDescriptors, gray2)

	matches = matchDescriptor(descriptors1, descriptors2)

	model_robust, inliers = ransacRemoveOutliers(keypoints1, keypoints2, matches)

	return wraping(image1, image2, model_robust)

if __name__ == "__main__":

	width = 500

	image1 = Image.open(r"images\data\image_0001.jpg").convert('RGB')
	image1 = resize(image1, width=width)
	image1 = np.array(image1)	
	
	image2 = Image.open(r"images\data\image_0021.jpg").convert('RGB')
	image2 = resize(image2, width=width)
	image2 = np.array(image2)
	
	image3 = Image.open(r"images\data\image_0041.jpg").convert('RGB')
	image3 = resize(image3, width=width)
	image3 = np.array(image3)

	image4 = Image.open(r"images\data\image_0051.jpg").convert('RGB')
	image4 = resize(image4, width=width)
	image4 = np.array(image4)

	image5 = Image.open(r"images\data\image_0061.jpg").convert('RGB')
	image5 = resize(image5, width=width)
	image5 = np.array(image5)

	image6 = Image.open(r"images\data\image_0071.jpg").convert('RGB')
	image6 = resize(image6, width=width)
	image6 = np.array(image6)


	image7 = Image.open(r"images\data\image_0081.jpg").convert('RGB')
	image7 = resize(image7, width=width)
	image7 = np.array(image7)


	image8 = Image.open(r"images\data\image_0091.jpg").convert('RGB')
	image8 = resize(image8, width=width)
	image8 = np.array(image8)


	image9 = Image.open(r"images\data\image_0101.jpg").convert('RGB')
	image9 = resize(image9, width=width)
	image9 = np.array(image9)


	image = process(image1, image2)
	image = process(image, image3)
	image = process(image, image4)
	image = process(image, image5)
	image = process(image, image6)
	image = process(image, image7)
	image = process(image, image8)
	image = process(image, image9)

	cv2.imshow("image", cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
	cv2.waitKey(0)
	image = Image.fromarray(image)

	image.save("images/stitched.png")

