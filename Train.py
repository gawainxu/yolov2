# -*- coding: utf-8 -*-
"""
Created on Fri Feb  8 20:22:22 2019

@author: jxu
"""

from Net import Net
from Net import TINY_YOLOV2_ANCHOR_PRIORS as priors
from keras.optimizers import Adam
from Trainutils import YoloLossKeras, processGroundTruth

net = Net(13 * 32, 5, 20, is_learning_phase=True)     ########## change in config

loss = YoloLossKeras(priors).loss
net.model.compile(optimizer=Adam(lr=1e-4), loss=loss, metrics=None)

y_true = processGroundTruth(boxes, labels, priors, (13, 13, 5, 25))
net.m.fit(image[None], y_true[None], steps_per_epoch=30, epochs=10)