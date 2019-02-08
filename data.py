# -*- coding: utf-8 -*-

import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.image as mpimg
import numpy as np
import os
import cv2

class Data:
    
    def __init__(self, dataset_path):
        
        self.dataset_path = dataset_path
        self.dataset_folders = os.listdir(self.dataset_path)
        self.dataset = []                                          # it includes all the training images and labels
        
    def sort_images(self):
       
        for folder in self.dataset_folders:
            class_name = ''.join([i for i in folder if not i.isdigit()])
            
            folder_path = os.path.join(self.dataset_path, folder)
            images_and_labels = os.listdir(folder_path)
            images_and_labels.sort()                                  # all the files in one dataset folder 
            
            for datas in images_and_labels:
                if '.xml' not in datas:      # it means it is an image file
                    image_num = datas.split('.')[0]
                    image = dict()           # dictionary to store all the info of the image 
                    new_image_name = folder + '_' + datas
                    #os.rename(datas, new_image_name)    # if full path required here?
                    
                    image['name'] = new_image_name
                    image['path'] = os.path.join(folder_path, datas)
                    image['class_name'] = class_name                    
                    
                    label_tree = ET.parse(folder_path + '/' + image_num + ".xml")
                    label_root = label_tree.getroot()
                    
                    s = label_root[1]                  #
                    width = int(s.find('width').text)
                    height = int(s.find('height').text)
                    
                    bbox = label_root[2][1]            #
                    xmax = int(bbox.find('xmax').text)
                    xmin = int(bbox.find('xmin').text)
                    ymax = int(bbox.find('ymax').text)
                    ymin = int(bbox.find('ymin').text)
                    
                    image['width'] = width
                    image['height'] = height
                    image['xmax'] = xmax
                    image['xmin'] = xmin
                    image['ymax'] = ymax
                    image['ymin'] = ymin
                    
                    self.dataset.append(image)
        

    def visualize(self):
        
        for d in self.dataset:
            im = cv2.imread(d["path"], 1)
            cv2.rectangle(im, (d["xmin"], d["ymin"]), (d["xmax"], d["ymax"]), (0, 255, 0), 2)
            cv2.imwrite('./data/' + d['name'], im)
     
def main():
    dataset_path = "/home/jxu/papers/SDC/data_training"
    data_parser = Data(dataset_path)
    data_parser.sort_images()
    
    data_parser.visualize()
    
    return data_parser.dataset

if __name__ == '__main__':
    dataset = main()
                    
                    