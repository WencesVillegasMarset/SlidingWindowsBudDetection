import cv2
import numpy as np
import matplotlib.pyplot as plt

def connected_components_with_threshold(image, threshold):
    '''
        Function that takes a mask and filters its component given a provided threshold
        this returns the number of resulting components and a new filtered mask (tuple) 
    '''
    num_components, mask = cv2.connectedComponents(image)
    filtered_mask = np.zeros_like(image, dtype=np.uint8)
    component_list = []
    for component in np.arange(1, num_components):
        isolated_component = (mask == component)
        if np.sum(isolated_component) >= threshold:
            filtered_mask += isolated_component.astype(np.uint8)
            component_list.append(component)
    return len(component_list), filtered_mask 

def plot_roc(rec, prec):
    plt.scatter(rec, prec, c=[1,2,3])
    plt.xlabel('Precision')
    plt.ylabel('Recall')
    plt.xlim(0,1)
    plt.ylim(0,1)
    plt.show()
if __name__ == "__main__":
    
    image = cv2.imread('../output/result300_150step/binary_masks/bin_mask_sw_0062.jpg', 0)
    plt.imshow(image)
    plt.show()
    num_components , mask = connected_components_with_threshold(image, 150*150+150)
    plt.imshow(mask)
    plt.show()
    print(num_components)
