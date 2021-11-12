# FaceSwap

## Output Example

| Body | Face | Output |
| --- | --- | --- |
|<img src="./images/ironman.jpg" width="200" title="failure cases"> | <img src="./images/rock.jpg"  width="200" title="failure cases"> | <img src="./images/ironman_rock.jpg"  width="200" title="failure cases"> |
|<img src="./images/will_smith.jpg" width="200" title="failure cases"> | <img src="./images/ryan_reynolds.webp"  width="200" title="failure cases"> | <img src="./images/will_ryan.jpg"  width="200" title="failure cases"> |
|<img src="./images/will_smith.jpg" width="200" title="failure cases"> | <img src="./images/rock.jpg"  width="200" title="failure cases"> | <img src="./images/will_rock.jpg"  width="200" title="failure cases"> |

## Dependency 
- [Download]("http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2") the dlib shape predictor which [68 landmark points](https://ibug.doc.ic.ac.uk/media/uploads/images/300-w/figure_1_68.jpg) on face
- Requirements: skimage, opencv-python, Pillow, dlib <br> `pip3 install Pillow opencv-python dlib scikit-image`


## Steps:
- Extract Landmark
- Compute and Apply Projective Transformation of face image
- Crop and Paste face image on the body image
- Apply Blending

<p float="left">
	<img src="./images/rock.jpg" width="200" />
	<img src="./images/will_smith.jpg" width="200" /> 
</p>

### Extract Landmark
Extract 68 facial landmark using dlib shape predictor
<p float="left">
	<img src="./images/kps1.jpg" width="200" />
	<img src="./images/kps2.jpg" width="200" /> 
</p>

### Compute and Apply Geometric Transformation of face image
By using 68 keypoint calculate the projective transformation and apply it on the face image
<p float="left"><img src="./images/transformed_face.jpg" width="200" /></p>

### Crop and Paste face image on the body image

#### Crop the image
<p float="left">
	<img src="./images/croped_face.jpg" width="200" />
	<img src="./images/croped_body.jpg" width="200" /> 
</p>

#### Paste face image on the body image
<p float="left"><img src="./images/pasted_image.jpg" width="200" /></p>

### Apply Blending
To match the body tone with the face we are using `cv2.seamlessClone`
<p float="left"><img src="./images/will_rock.jpg" width="200" /></p>


## References:
- [Satya Malik / learnopencv](https://learnopencv.com/face-swap-using-opencv-c-python/)