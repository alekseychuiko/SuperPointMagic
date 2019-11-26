# -*- coding: utf-8 -*-
"""
Created on Sun Nov 24 22:30:46 2019

@author: Alex
"""
import model
import cv2

if __name__ == '__main__':
    m = model.ModelFile('imx287.lmd')
    cv2.imwrite('imx287.jpg', m.image)