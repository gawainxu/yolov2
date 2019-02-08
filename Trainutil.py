# -*- coding: utf-8 -*-
"""
Created on Fri Feb  8 12:36:21 2019

@author: jxu
"""

import tensorflow as tf
import numpy as np

def bbToYoloFormat(bb):
    # convert bounding box format from x1, y1, x2, y2 to w, h, c_x, cy
    x1, y1, x2, y2 = np.split(bb, 4, axis = 1)
    
    w = x2 - x1
    h = y2 - y1
    c_x = x1 + w / 2
    c_y = y1 + h / 2
    
    return np.concatenate([c_x, c_y, w, h], axis = -1)
    
    
def findBestPrior(bb, priors):
    
    w1, h1 = bb[:, 2], bb[:, 3]
    w2, h2 = priors[:, 0], priors[:, 1]
    
    horizontal_overlap = np.minimum(w1[:, None], w2)
    veritical_overlap = np.minimum(h1[:, None], h2)
    
    intersection = horizontal_overlap * veritical_overlap
    union = (w1 * h1)[:, None] + (w2 * h2) - intersection
    iou = intersection / union
    
    return np.argmax(iou, axis = 1)
    

def processGroundTruth(bb, labels, priors, network_output_shape):
    """
    Given bounding boxes in normal x1,y1,x2,y2 format, the relevant labels in one-hot form,
    the anchor priors and the yolo model's output shape
    build the y_true vector to be used in yolov2 loss calculation
    """
    
    bb = bbToYoloFormat(bb) / 32
    best_anchor_indices = findBestPrior(bb, priors)
    
    responsible_grid_coords = np.floor(bb).astype(np.uint32)[:, :2]
    
    values = np.concatenate((bb, np.ones((len(bb), 1)), labels), axis = 1)
    
    x, y = np.split(responsible_grid_coords, 2, axis = 1)
    y = y.ravel()
    x = x.ravel()
    
    y_true = np.zeros(network_output_shape)
    y_true[y, x, best_anchor_indices] = values
    
    return y_true
    

class YoloLossKeras:
    
    def __init__(self, priors):
        self.priors = priors
        
    def loss(self, y_true, y_pred):
        
        n_cells = y_pred.get_shape().as_list()[1]
        y_true = tf.reshape(y_true, tf.shape(y_pred), name = 'y_true')
        y_pred = tf.identity(y_pred, name = 'y_pred')
        
        predicted_xy = tf.nn.sigmoid(y_pred[...,:2])
        
        cell_inds = tf.range(n_cells, dtype = tf.float32)
        predicted_xy = tf.stack((
            predicted_xy[..., 0] + tf.reshape(cell_inds, [1, -1, 1]), 
            predicted_xy[..., 1] + tf.reshape(cell_inds, [-1, 1, 1])), axis=-1)
            
        predicted_wh = self.priors * tf.exp(y_pred[..., 2:4])
        
        predicted_min = predicted_xy - predicted_wh / 2
        predicted_max = predicted_xy + predicted_wh / 2
        
        predicted_objectedness = tf.nn.sigmoid(y_pred[..., 4])
        predicted_logits = tf.nn.softmax(y_pred[..., 5:])
        
        true_xy = y_true[..., :2]
        true_wh = y_true[..., 2:4]
        true_logits = y_true[..., 5:]
        
        true_min = true_xy - true_wh / 2
        true_max = true_xy + true_wh / 2
        
        # compute iou between true and predicted
        intersect_mins = tf.maximum(predicted_min, true_min)
        intersect_maxes = tf.minimum(predicted_max, true_max)
        intersect_wh = tf.maximum(intersect_maxes - intersect_mins, 0.)
        intersect_areas = intersect_wh[..., 0] * intersect_wh[..., 1]
        
        true_areas = true_wh[..., 0] * true_wh[..., 1]
        pred_areas = predicted_wh[..., 0] * predicted_wh[..., 1]
        
        union_areas = pred_areas + true_areas - intersect_areas
        iou_scores  = intersect_areas / union_areas
        
        responsibility_selector = y_true[..., 4]
        
        xy_diff = tf.square(true_xy - predicted_xy) * responsibility_selector[..., None]
        xy_loss = tf.reduce_sum(xy_diff, axis=[1, 2, 3, 4])

        wh_diff = tf.square(tf.sqrt(true_wh) - tf.sqrt(predicted_wh)) * responsibility_selector[..., None]
        wh_loss = tf.reduce_sum(wh_diff, axis=[1, 2, 3, 4])
        
        obj_diff = tf.square(iou_scores - predicted_objectedness) * responsibility_selector
        obj_loss = tf.reduce_sum(obj_diff, axis=[1, 2, 3])
        
        best_iou = tf.reduce_max(iou_scores, axis=-1)
        no_obj_diff = tf.square(0 - predicted_objectedness) * tf.to_float(best_iou < 0.6)[..., None] * (1 - responsibility_selector)
        no_obj_loss = tf.reduce_sum(no_obj_diff, axis=[1, 2, 3])

        clf_diff = tf.square(true_logits - predicted_logits) * responsibility_selector[..., None]
        clf_loss = tf.reduce_sum(clf_diff, axis=[1, 2, 3, 4])

        object_coord_scale = 5        ######################
        object_conf_scale = 1
        noobject_conf_scale = 1
        object_class_scale = 1

        loss = object_coord_scale * (xy_loss + wh_loss) + \
                object_conf_scale * obj_loss + noobject_conf_scale * no_obj_loss + \
             object_class_scale * clf_loss
             
        return loss

