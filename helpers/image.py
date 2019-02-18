import cv2


def read_img(path):
    '''
        Read an image in RGB mode, full path required
    '''
    return cv2.cvtColor(cv2.imread(path,1), cv2.COLOR_BGR2RGB)

def save_img(path, image):
    '''
        Save an image in the specified path with the specified imagename e.g (/path/to/, 'mypic.jpg')
    '''
    cv2.imwrite(path, image)

