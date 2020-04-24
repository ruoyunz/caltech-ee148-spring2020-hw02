import os
import json
import numpy as np

import matplotlib.pyplot as plt

def compute_iou(box_1, box_2):
    '''
    This function takes a pair of bounding boxes and returns intersection-over-
    union (IoU) of two bounding boxes.
    '''
    tl_r1, tl_c1, br_r1, br_c1 = box_1
    tl_r2, tl_c2, br_r2, br_c2 = box_2

    height = max(min(br_r1, br_r2) - max(tl_r1, tl_r2) + 1, 0)
    width = max(min(br_c1, br_c2) - max(tl_c1, tl_c2) + 1, 0)
    intersection = height * width

    a1 = max(br_r1 - tl_r1 + 1, 0) * max(br_c1 - tl_c1 + 1, 0)
    a2 = max(br_r2 - tl_r2 + 1, 0) * max(br_c2 - tl_c2 + 1, 0)

    iou = intersection / (a1 + a2 - intersection)
    
    assert (iou >= 0) and (iou <= 1.0)

    return iou


def compute_counts(preds, gts, iou_thr=0.5, conf_thr=0.5):
    '''
    This function takes a pair of dictionaries (with our JSON format; see ex.) 
    corresponding to predicted and ground truth bounding boxes for a collection
    of images and returns the number of true positives, false positives, and
    false negatives. 
    <preds> is a dictionary containing predicted bounding boxes and confidence
    scores for a collection of images.
    <gts> is a dictionary containing ground truth bounding boxes for a
    collection of images.
    '''
    TP = 0
    FP = 0
    FN = 0

    '''
    BEGIN YOUR CODE
    '''
    for pred_file, pred in preds.items():
        gt = gts[pred_file]
        detected = len(pred) * [False]
        for i in range(len(gt)):
            done = False
            for j in range(len(pred)):
                iou = compute_iou(pred[j][:4], gt[i])
                if pred[j][4] >= conf_thr and iou >= iou_thr and not detected[j]:
                    detected[j] = True 
                    TP += 1
                    done = True
                    break
            if not done:
                FN += 1
        for k in range(len(detected)):
            if detected[k] == 0 and pred[k][4] >= conf_thr:
                FP += 1
    '''
    END YOUR CODE
    '''

    return TP, FP, FN

def pr_plots(name, preds_train, gts_train):
    # idk why the but the given code for sorting wasn't working for me
    confidence_thrs = []
    for fname in preds_train:
        for i in preds_train[fname]:
            confidence_thrs.append(i[4])
    confidence_thrs = np.sort(confidence_thrs)

    for iou in [0.25, 0.5, 0.75]:
        tp_train = np.zeros(len(confidence_thrs))
        fp_train = np.zeros(len(confidence_thrs))
        fn_train = np.zeros(len(confidence_thrs))
        for i, conf_thr in enumerate(confidence_thrs):
            tp_train[i], fp_train[i], fn_train[i]  = compute_counts(preds_train, gts_train, iou_thr=iou, conf_thr=conf_thr)

        prec = tp_train / (tp_train + fp_train)
        recall = tp_train / (tp_train + fn_train)

        plt.step(recall, prec, label=str(iou))

    plt.xlabel("recall")
    plt.ylabel("precision")
    plt.legend(title='IOU Thresholds')
    plt.savefig("./pr_curve" + name + ".jpg")
    plt.clf()

# set a path for predictions and annotations:
preds_path = '../data/hw02_preds'
gts_path = '../data/hw02_annotations'

# load splits:
split_path = '../data/hw02_splits'
file_names_train = np.load(os.path.join(split_path,'file_names_train.npy'))
file_names_test = np.load(os.path.join(split_path,'file_names_test.npy'))

# Set this parameter to True when you're done with algorithm development:
done_tweaking = True

'''
Load training data. 
'''
with open(os.path.join(preds_path,'preds_train_final.json'),'r') as f:
    preds_train = json.load(f)
    
with open(os.path.join(gts_path, 'annotations_train.json'),'r') as f:
    gts_train = json.load(f)

if done_tweaking:
    
    '''
    Load test data.
    '''
    
    with open(os.path.join(preds_path,'preds_test_final.json'),'r') as f:
        preds_test = json.load(f)
        
    with open(os.path.join(gts_path, 'annotations_test.json'),'r') as f:
        gts_test = json.load(f)



# For a fixed IoU threshold, vary the confidence thresholds.
# The code below gives an example on the training set for one IoU threshold. 
pr_plots("_train", preds_train, gts_train)


if done_tweaking:
    print('Code for plotting test set PR curves.')
    pr_plots("_test", preds_test, gts_test)
