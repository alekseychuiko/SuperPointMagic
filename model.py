# -*- coding: utf-8 -*-
"""
Created on Sun Nov 24 22:30:46 2019

@author: Alex
"""

import json
import base64
import numpy as np

class ModelFile:
    def __init__(self, filename):
        self.valid = False
        with open(filename,'rb') as f:
            content = f.read()
            data = json.loads(content)  
            if 'version' in data:
                self.version = data['version']
                if self.version == 1:
                    self.loadModel_1(data)    
    def loadModel_1(self, data):
        imageObj = data['image']
        cols = imageObj['cols']
        rows = imageObj['rows']
        elemSize = imageObj['elemSize']
        imageData = imageObj['data'].encode('ascii')
        decodeData = base64.decodebytes(imageData)
        self.image = np.frombuffer(decodeData, dtype=np.uint8)
        self.image = np.reshape(self.image, (rows, cols,  elemSize))
        self.leds = data['leds']
        self.valid = True
            
        
        
        