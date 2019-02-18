import numpy as np
import cv2
import json


def sift_keypoints(image):
	'''Get list of detected keypoint objects from an image'''
	sift_obj = cv2.xfeatures2d.SIFT_create()
	keypoints = sift_obj.detect(image, None)
	return keypoints

def sift_keypoints_descriptors(image):
	'''
		Get sift descriptors from image, we not only return the keypoint
		objects but also we compute its corresponding descriptors
	'''
	sift_obj = cv2.xfeatures2d.SIFT_create()
	keypoints, descriptors = sift_obj.detectAndCompute(image,None)
	return (keypoints, descriptors)

def draw_keypoints(image, keypoints):
	'''
		draw keypoints on image
	'''
	return (cv2.drawKeypoints(image,keypoints, cv2.DRAW_MATCHES_FLAGS_DRAW_OVER_OUTIMG))

def save_keypoints(keypoints, descriptors, out_path):
	'''
		save on a json file the keyponts and descriptors of an image or patch
		it is needed to specify the full path and resulting filename for the json file
		(e.g. out_path -> /path/to/patch_00001_from_1144.json)
	'''
	data = {
        'keypoints':keypoints,
        'descriptors':descriptors.tolist()
    }
	with open(out_path, 'w') as fp:
    		json.dump(data, fp, indent=4)
	print('Keypoints saved!')


if __name__ == "__main__":
	from helpers import image
	img = image.read_img('./0001.jpg')
	(kp, desc) = sift_keypoints(img)
	image.save_img('./kp_drawn.jpg', draw_keypoints(img, kp))

	for i, x in enumerate(kp):
		kp[i] = x.pt

	save_keypoints(kp,desc,'.lala.json')
