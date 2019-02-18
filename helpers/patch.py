import numpy as np


def sliding_window(img, window_size, step_size):
	'''
		This method returns a list of coordinates, this coordinates are (x,y) tuples
		that correspond to top-left corners of the patches that will be applied to the 
		input image
	'''
	xy_coord_list = []
	for x in range(0, img.shape[0], step_size):
		for y in range(0, img.shape[1], step_size):
			if x + window_size > img.shape[0]:
				x = img.shape[0] - window_size
			if y + window_size > img.shape[1]:
				y = img.shape[1] - window_size
			xy_coord_list.append(np.asarray([x, y]))

	return np.asarray(xy_coord_list)

def get_patch(image ,top_left, size):
	'''
		Get a patch from the input image, this method returns a slice of the input array 
		(the image).
	'''
	return image[top_left[0]:(top_left[0] + size), top_left[1]:(top_left[1] + size), :]

def search_for_ground_truth(gt_csv, img_name, patch_x, patch_y, window_size, min_overlap):
	'''
		Get the ground truth for the generated patch

		Given that we have a ground truth csv that provides labels for each patch generated on the
		original slding window run, we calculate the overlap of the newly generated patch and the 
		bud ground truth patches. If the overlap with the closest ground truth patch is greater than
		min_overlap it is marked as positive.
		This way we generate the ground truth for our new patches based on the original ground truth
	'''
    # get patches belonging to the same image
	gt_image_patches = gt_csv.loc[gt_csv['imageOrigin'] == img_name, :]
	(patch_x_center, patch_y_center) = (patch_x + (window_size/2), patch_y + (window_size/2))
	#gt_image_patches = gt_image_patches.loc[gt_image_patches['class'] == 'TRUE', :]

	for idx, row in gt_image_patches.iterrows():
		gt_left_corner = ((row['xBudCenter'] - (row['radio']/2)), (row['yBudCenter'] - (row['radio']/2)))
		if (patch_x < gt_left_corner[0]):
			match_pixels_x = (patch_x + window_size) - gt_left_corner[0]
		else:
			match_pixels_x = (gt_left_corner[0]+ row['radio']) - patch_x
		if (patch_y < gt_left_corner[1]):
			match_pixels_y = (patch_y + window_size) - gt_left_corner[1]
		else:
			match_pixels_y = (gt_left_corner[1]+ row['radio']) - patch_y
		if match_pixels_x > 0 and match_pixels_y > 0:
			if row['radio']**2 > window_size**2:
				overlap = match_pixels_x*match_pixels_y / row['radio']**2
			else:
				overlap = match_pixels_x*match_pixels_y / window_size**2
		else:
			return False
		print(overlap)
		if overlap >= min_overlap:
			return True
		else:
			return False




if __name__ == "__main__":
	
	import image
	img = image.read_img('../0001.jpg')
	patch = get_patch(img, (1500,1500), 1000)
	print(patch.shape)
	image.save_img('../patch.jpg', patch)



