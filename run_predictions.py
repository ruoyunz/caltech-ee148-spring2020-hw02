import os
import numpy as np
import json
from PIL import Image
import matplotlib.pyplot as plt


def single_template(I, T):
    (n_rows,n_cols,n_channels) = np.shape(I)
    heatmap = np.zeros((n_rows, n_cols, 3))
    (box_height, box_width, _) = np.shape(T)

    T = np.asarray(T)
    T = T.flatten()
    T = T / np.linalg.norm(T)

    (n_rows, n_cols, n_channels) = np.shape(I)
    all_boxes = []

    for r in range((n_rows - box_height)):
        for c in range(n_cols - box_width):
            x = (I[r:(r + box_height), c:(c + box_width), :]).flatten()
            x = x / np.linalg.norm(x)
            heatmap[r, c] = [np.dot(T, x), box_height, box_width]

    return heatmap

def compute_convolution(I, T1, T2, T3, stride=None):
    '''
    This function takes an image <I> and a template <T> (both numpy arrays) 
    and returns a heatmap where each grid represents the output produced by 
    convolution at each location. You can add optional parameters (e.g. stride, 
    window_size, padding) to create additional functionality. 
    '''
    (n_rows,n_cols,n_channels) = np.shape(I)

    '''
    BEGIN YOUR CODE
    '''
    heatmap = np.zeros((n_rows, n_cols, 3))
    show_map = np.zeros((n_rows, n_cols))

    hm1 = single_template(I, T1)
    hm2 = single_template(I, T2)
    hm3 = single_template(I, T3) * .9

    for r in range(n_rows):
        for c in range(n_cols):
            heatmap[r, c] = max([hm1[r, c], hm2[r, c], hm3[r, c]], key =lambda l: l[0])
            show_map[r, c] = heatmap[r, c][0]
    '''
    END YOUR CODE
    '''
    # Code for generating heat maps 
    # plt.imshow(show_map[:, :], cmap='hsv')
    # plt.savefig("./heatmap.jpg")

    return heatmap


def predict_boxes(I, heatmap):
    '''
    This function takes heatmap and returns the bounding boxes and associated
    confidence scores.
    '''

    output = []
    (n_rows,n_cols,n_channels) = np.shape(I)

    '''
    BEGIN YOUR CODE
    '''
    for r in range(int(0.5 * n_rows)):
        for c in range(n_cols):
            R, box_height, box_width = heatmap[r, c]

            # Ensure that the red light (most important feature) is taken care of
            # by only checking for pixels where the top light area has a high R 
            # channel value
            if R > 0.9 and I[int(r + box_height / 2), int(c + box_width / 2), 0] > 200:
                output.append([r, c, r + box_height, c + box_width, R])

    return output


# Helper function to calculate the amount of overlap between two 
# boxes of the same size given the top left coordinates
def areaOverlap(tl_r1, tl_c1, br_r1, br_c1, tl_r2, tl_c2, br_r2, br_c2):
    height = max(min(br_r1, br_r2) - max(tl_r1, tl_r2) + 1, 0)
    width = max(min(br_c1, br_c2) - max(tl_c1, tl_c2) + 1, 0)
    intersection = height * width

    a1 = max(br_r1 - tl_r1 + 1, 0) * max(br_c1 - tl_c1 + 1, 0)
    a2 = max(br_r2 - tl_r2 + 1, 0) * max(br_c2 - tl_c2 + 1, 0)

    if intersection == 0:
        return 0
    return intersection / min(a1, a2)

# Consolidate bounding boxes--get overlapping bounding boxes and 
# only save the one with the highest threshold. This is done by using
# non maximum suppression--all boxes that overlap by more than 50% will
# be consolidated.
# Adapted from assignment 1
def consolidation(all_boxes):
    temp_boxes = [all_boxes[0]]

    for i in range(1, len(all_boxes)):
        (tl_r, tl_c, br_r, br_c, R) = all_boxes[i]
        new_max = False
        overlap = False
        j = 0
        while j < len(temp_boxes):
            (tl_row, tl_col, br_row, br_col, temp_R) = temp_boxes[j]
            a = areaOverlap(tl_r, tl_c, br_r, br_c, tl_row, tl_col, br_row, br_col)
            if a > 0.5:
                overlap = True
                if R > temp_R:
                    new_max = True
                    temp_boxes.pop(j)
                    j -= 1
            j += 1
        if (overlap and new_max) or (not overlap):
            temp_boxes.append(all_boxes[i])

    return temp_boxes

def detect_red_light_mf(I):
    '''
    This function takes a numpy array <I> and returns a list <output>.
    The length of <output> is the number of bounding boxes predicted for <I>. 
    Each entry of <output> is a list <[row_TL,col_TL,row_BR,col_BR,score]>. 
    The first four entries are four integers specifying a bounding box 
    (the row and column index of the top left corner and the row and column 
    index of the bottom right corner).
    <score> is a confidence score ranging from 0 to 1. 

    Note that PIL loads images in RGB order, so:
    I[:,:,0] is the red channel
    I[:,:,1] is the green channel
    I[:,:,2] is the blue channel
    '''

    '''
    BEGIN YOUR CODE
    '''
    # Get the traffic light image (kernel)
    print("Kernel 1")
    T1 = Image.open(os.path.join(data_path, 
        "../../caltech-ee148-spring2020-hw02/truth1.jpg"))

    print("Kernel 2")
    T2 = Image.open(os.path.join(data_path, 
        "../../caltech-ee148-spring2020-hw02/truth2.jpg"))

    print("Kernel 3")
    T3 = Image.open(os.path.join(data_path, 
        "../../caltech-ee148-spring2020-hw02/truth3.jpg"))
    
    heatmap = single_template(I, T1)
    all_boxes = predict_boxes(I, heatmap)

    print("Consolidating")
    if len(all_boxes) == 0:
        return []
    output = consolidation(all_boxes)

    '''
    END YOUR CODE
    '''

    for i in range(len(output)):
        assert len(output[i]) == 5
        assert (output[i][4] >= 0.0) and (output[i][4] <= 1.0)

    return output

# Note that you are not allowed to use test data for training.
# set the path to the downloaded data:
data_path = '../data/RedLights2011_Medium'

# load splits: 
split_path = '../data/hw02_splits'
file_names_train = np.load(os.path.join(split_path,'file_names_train.npy'))
file_names_test = np.load(os.path.join(split_path,'file_names_test.npy'))

# set a path for saving predictions:
preds_path = '../data/hw02_preds'
os.makedirs(preds_path, exist_ok=True) # create directory if needed

# Set this parameter to True when you're done with algorithm development:
done_tweaking = True

'''
Make predictions on the training set.
'''
preds_train = {}
for i in range(len(file_names_train)):
    # read image using PIL:
    I = Image.open(os.path.join(data_path, file_names_train[i]))

    # convert to numpy array:
    I = np.asarray(I)
    print(file_names_train[i])
    preds_train[file_names_train[i]] = detect_red_light_mf(I)

# save preds (overwrites any previous predictions!)
with open(os.path.join(preds_path,'preds_train_weak.json'),'w') as f:
    json.dump(preds_train,f)

if done_tweaking:
    '''
    Make predictions on the test set. 
    '''
    preds_test = {}
    for i in range(len(file_names_test)):

        # read image using PIL:
        I = Image.open(os.path.join(data_path, file_names_test[i]))

        # convert to numpy array:
        I = np.asarray(I)

        preds_test[file_names_test[i]] = detect_red_light_mf(I)

    # save preds (overwrites any previous predictions!)
    with open(os.path.join(preds_path,'preds_test_weak.json'),'w') as f:
        json.dump(preds_test,f)
