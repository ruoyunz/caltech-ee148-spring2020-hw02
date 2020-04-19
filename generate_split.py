import numpy as np
import os
import json

import cv2
import sys

np.random.seed(2020) # to ensure you always get the same train/test split

data_path = '../data/RedLights2011_Medium'
gts_path = '../data/hw02_annotations'
split_path = '../data/hw02_splits'
os.makedirs(split_path, exist_ok=True) # create directory if needed

split_test = True # set to True and run when annotations are available

train_frac = 0.85

# get sorted list of files:
file_names = sorted(os.listdir(data_path))

# remove any non-JPEG files:
file_names = [f for f in file_names if '.jpg' in f]

# split file names into train and test
file_names_train = []
file_names_test = []

np.random.shuffle(file_names)
file_names_train = file_names[0:(int(train_frac * len(file_names)))]
file_names_test = file_names[(int(train_frac * len(file_names))):]

assert (len(file_names_train) + len(file_names_test)) == len(file_names)
assert len(np.intersect1d(file_names_train,file_names_test)) == 0

np.save(os.path.join(split_path,'file_names_train.npy'),file_names_train)
np.save(os.path.join(split_path,'file_names_test.npy'),file_names_test)

# Function for viewing annotations one by one if needed
def view_annotation(fs):
    for i in range(len(fs)):
        img = cv2.imread(os.path.join(data_path,fs[i]),cv2.IMREAD_COLOR)
        preds = gts_stud[fs[i]]
        
        for p in preds2:
            (r, c, r2, c2) = p
            cv2.rectangle(img, (int(c), int(r)), (int(c2), int(r2)), (0, 255, 0), 1)
            cv2.imshow('image',img)
            ch = cv2.waitKey(0)
            if ch == 27:
                exit()

if split_test:
    with open(os.path.join(gts_path, 'formatted_annotations_students.json'),'r') as f:
        gts = json.load(f)
    
    # Use file_names_train and file_names_test to apply the split to the
    # annotations
    gts_train = {}
    gts_test = {}

    for i in range(len(file_names_train)):
        gts_train[file_names_train[i]] = gts[file_names_train[i]]
    for i in range(len(file_names_test)):
        gts_test[file_names_test[i]] = gts[file_names_test[i]]

    with open(os.path.join(gts_path, 'annotations_train.json'),'w') as f:
        json.dump(gts_train,f)
    
    with open(os.path.join(gts_path, 'annotations_test.json'),'w') as f:
        json.dump(gts_test,f)
    
    
