# -*- coding: utf-8 -*-
"""
Created on Sun Nov 24 22:30:46 2019

@author: Alex
"""

import bson
import numpy as np

class ModelFile:
    def __init__(self, filename):
        self.valid = False
        with open(filename,'rb') as f:
            data = bson.decode(f.read())
            if 'version' in data:
                self.version = data['version']
                if self.version == 1:
                    loadModel_1(data)    
    def loadModel_1(self, data):
        imageObj = data['image']
        cols = imageObj['cols']
        rows = imageObj['rows']
        elemSize = imageObj['elemSize']
        elemType = imageObj['elemType']
        self.image = np.asarray(base64.decodestring(imageObj['data']), dtype=np.uint8)      
        self.image = np.reshape(self.image, (cols, rows, elemType))
        self.leds = []
        ledsArr = data['leds']
        for ledObj in ledsArr:
            led = dict();
            
        
        
        