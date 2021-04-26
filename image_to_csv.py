#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 16 10:50:16 2021

@author: rg
"""

import numpy as np
import cv2
import os
import csv
import pandas as pd  

def create_file_list(mydir, format='.jpg'):
    file_list = []
    
    for(root, dirs, files) in os.walk(mydir, topdown = False):
        for name in files:
            full_name = os.path.join(root,name)
            file_list.append(full_name)
            
    return file_list

dataset = 'dataset'

for i in ['1', '2', '3', '4', '5', '6', '7', '8', '9']:
    path = os.path.join(dataset, i)
    for file in os.listdir(path):
        img = cv2.imread(os.path.join(path,file))
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        thresh = cv2.threshold(gray, 127,255,
                               cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
        thresh = cv2.resize(thresh, (28, 28))
        
        value = np.asarray(thresh, dtype=np.int)
        value = value.flatten()
        value = np.insert(value, 0, int(i), axis=0)
        
        with open("train2.csv", 'a') as f:
            writer = csv.writer(f)
            writer.writerow(value)
            
ds = pd.read_csv("train2.csv")
ds = ds.sample(frac=1)
ds.to_csv("train.csv")
os.remove("train2.csv")
        