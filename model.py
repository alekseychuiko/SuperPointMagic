# -*- coding: utf-8 -*-
"""
Created on Sun Nov 24 22:30:46 2019

@author: Alex
"""

import bson

class ModelFile:
    def __init__(self, filename):
        self.valid = False
        with open(filename,'rb') as f:
            data = bson.decode(f.read())
            if 'version' in data:
                self.version = data['version']
                if self.version == 1:
                    loadModel_1(data)    
    def loadModel_1(data):
        