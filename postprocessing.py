# -*- coding: utf-8 -*-
"""
Created on Fri Feb  8 18:59:38 2019

@author: jxu
"""

import numpy as np

def logistic(x):
    return 1 / (1 + np.exp(-x))
    
def softmax(x):
    exp_out = np.exp(x - np.max(x, axis = -1)[...,None])
    return exp_out / np.sum(exp_out, axis=-1)[..., None]
    
   
# https://www.pyimagesearch.com/2015/02/16/faster-non-maximum-suppression-python/
def non_max_suppression(boxes, labels, overlapThresh):
    
    if len(boxes) == 0:
        return []
        
    pick = []
    
    x1 = boxes[:, 0]
    y1 = boxes[:,1]
    x2 = boxes[:,2]
    y2 = boxes[:,3]    
    
    area = (x2 - x1 + 1)*(y2 - y1 + 1)
    idxs = np.argsort(y2)
    
    while len(idxs) > 0:
        i = idxs[-1]
        pick.append(i)
        
        xx1 = np.maximum(x1[i], x1[idxs[:-1]])
        yy1 = np.maximum(y1[i], y1[idxs[:-1]])
        xx2 = np.minimum(x2[i], x2[idxs[:-1]])
        yy2 = np.minimum(y2[i], y2[idxs[:-1]])
        
        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)
 
        # compute the ratio of overlap
        overlap = (w * h) / area[idxs[:-1]]
 
        # delete all indexes from the index list that have
        idxs = (idxs[:-1])[overlap < overlapThresh]
 
    # return only the bounding boxes that were picked using the
    # integer data type
        return boxes[pick], labels[pick]
        
        
def getBoundingBoxesFromNetOutput(clf, anchors, confidence_threshold, cell_h, cell_w):
    
    pw, ph = anchors[:, 0], anchors[:, 1]
    cell_inds = np.arange(clf.shape[1])
    
    tx = clf[..., 0]
    ty = clf[..., 1]
    tw = clf[..., 2]
    th = clf[..., 3]
    to = clf[..., 4]
    
    sftmx = softmax(clf[..., 5:])
    predicted_labels = np.argmax(sftmx, axis=-1)
    class_confidences = np.max(sftmx, axis=-1)
    
    bx = logistic(tx) + cell_inds[None, :, None]
    by = logistic(ty) + cell_inds[None, :, None]
    bw = pw * np.exp(tw) / 2
    bh = ph * np.exp(th) / 2
    object_confidences = logistic(to)

    left = bx - bw
    right = bx + bw
    top = by - bh
    bottom = by + bh

    boxes = np.stack((left, top, right, bottom), axis = -1)*cell_h   ##  reshape ??
    
    final_confidence = class_confidences * object_confidences
    boxes = boxes[final_confidence > confidence_threshold].reshape(-1, 4).astype(np.int32)
    labels = predicted_labels[final_confidence > confidence_threshold]
    return boxes, labels
    

def yoloPostProcess(yolo_output, priors, maxsuppression=True, maxsuppressionthresh=0.5, classthresh=0.6, cell_size=32):
    allboxes = []
    for o in yolo_output:
        boxes, labels = getBoundingBoxesFromNetOutput(o, priors, confidence_threshold=classthresh, cell_size=cell_size)
        if maxsuppression and len(boxes) > 0:
            boxes, labels = non_max_suppression(boxes, labels, maxsuppressionthresh)
        allboxes.append((boxes, labels))

    return allboxes
    