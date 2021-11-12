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
- Compute and Apply Projective Tranformation on face image
- Crop and Paste face image on the second image
- Apply Blending


### Extract Landmark

![alt-text-1](./images/will_smith.jpg) ![alt-text-2](./images/rock.jpg)



## References:
- [Satya Malik / learnopencv](https://learnopencv.com/face-swap-using-opencv-c-python/)