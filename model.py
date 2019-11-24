# -*- coding: utf-8 -*-
"""
Created on Sun Nov 24 22:30:46 2019

@author: Alex
"""

import bson

class ModelFile:
    def __init__(self, filename):
        with open(filename,'rb') as f:
            data = bson.decode(f.read())
            if 'version' in data:
                version = data['version']
                if version == 1:
                    loadModel_1(data)    
    def