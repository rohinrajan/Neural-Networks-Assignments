import numpy as np
import scipy.misc

def read_one_image_and_convert_to_vector(file_name):
    img = scipy.misc.imread(file_name).astype(np.float32) # read image and convert to float
    val= img.reshape(-1,1) # reshape to column vector and return it
    #val = img
    val /=255 # normalizing the values in the code
    return val
