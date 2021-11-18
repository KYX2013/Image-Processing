from PyQt5 import QtWidgets
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *

import GUI as ui
import numpy as np
import matplotlib.pyplot as plt
import cv2

class Main(QDialog, ui.Ui_dialog):
    def __init__(self):
        super().__init__()
        self.setupUi(self)

        self.toolButton.clicked.connect(self.hw1_1)
        self.toolButton_2.clicked.connect(self.hw1_2)
        self.toolButton_3.clicked.connect(self.hw1_3)
        self.toolButton_4.clicked.connect(self.hw1_4)
        self.toolButton_5.clicked.connect(self.hw2_1)
        self.toolButton_6.clicked.connect(self.hw2_2)
        self.toolButton_7.clicked.connect(self.hw2_3)
        #self.toolButton_9.clicked.connect(self.hw3_1)
        #self.toolButton_10.clicked.connect(self.hw3_2)
        #self.toolButton_11.clicked.connect(self.hw3_3)
        #self.toolButton_12.clicked.connect(self.hw3_4)
        #self.toolButton_13.clicked.connect(self.hw4_1)
        #self.toolButton_14.clicked.connect(self.hw4_2)
        #self.toolButton_15.clicked.connect(self.hw4_3)
        #self.toolButton_16.clicked.connect(self.hw4_4)
    def hw1_1(self):
        img = cv2.imread('Dataset_OpenCvDl_Hw1\Q1_Image\Sun.jpg')
        cv2.imshow('LoadImage',img)
        print("Height: ",img.shape[0])
        print(" Width: ",img.shape[1])
    def hw1_2(self):
        img = cv2.imread('Dataset_OpenCvDl_Hw1\Q1_Image\Sun.jpg')
        (imgB,imgG,imgR) = cv2.split(img)
        zero = np.zeros(img.shape[:2],dtype = "uint8")
        cv2.imshow("B channel",cv2.merge([imgB,zero,zero]))
        cv2.imshow("G channel",cv2.merge([zero,imgG,zero]))
        cv2.imshow("R channel",cv2.merge([zero,zero,imgR]))
    def hw1_3(self):
        img = cv2.imread('Dataset_OpenCvDl_Hw1\Q1_Image\Sun.jpg')
        (imgB,imgG,imgR) = cv2.split(img)
        imgI1 = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        imgI2 = imgB[:]//3+imgG[:]//3+imgR[:]//3
        cv2.imshow("Perceptually weighted formula",imgI1)
        cv2.imshow("Average weighted formula",imgI2)
    def hw1_4(self):
        imgS = cv2.imread('Dataset_OpenCvDl_Hw1\Q1_Image\Dog_Strong.jpg')
        imgW = cv2.imread('Dataset_OpenCvDl_Hw1\Q1_Image\Dog_Weak.jpg')
        def bar(x):
            global barPos
            barPos = cv2.getTrackbarPos('Blend','Blend')
            imgDog = cv2.addWeighted(imgS,(255-barPos)/255,imgW,barPos/255,0)
            cv2.imshow('Blend',imgDog)
        cv2.namedWindow('Blend')
        cv2.createTrackbar('Blend','Blend',0,255,bar)
        cv2.setTrackbarPos('Blend','Blend',127)
    def hw2_1(self):
        img = cv2.imread('Dataset_OpenCvDl_Hw1\Q2_Image\Lenna_whiteNoise.jpg')
        blur = cv2.GaussianBlur(img,(5,5),0)
        cv2.imshow('whiteNoise',img)
        cv2.imshow('Gaussian',blur)
    def hw2_2(self):
        img = cv2.imread('Dataset_OpenCvDl_Hw1\Q2_Image\Lenna_whiteNoise.jpg')
        blur = cv2.bilateralFilter(img,9,90,90)
        cv2.imshow('whiteNoise',img)
        cv2.imshow('Bilateral',blur)
    def hw2_3(self):
        img = cv2.imread('Dataset_OpenCvDl_Hw1\Q2_Image\Lenna_pepperSalt.jpg')
        blur3 = cv2.medianBlur(img,3)
        blur5 = cv2.medianBlur(img,5)
        cv2.imshow('pepperSalt',img)
        cv2.imshow('Median(3x3)',blur3)
        cv2.imshow('Median(5x5)',blur5)
    def hw3_1(self):
        img = cv2.imread('Dataset_OpenCvDl_Hw1\Q3_Image\House.jpg')
    #def hw3_2(self):
    #def hw3_3(self):
    #def hw3_4(self):
    #def hw4_1(self):
    #def hw4_2(self):
    #def hw4_3(self):
    #def hw4_4(self):

if __name__ == '__main__':
  import sys
  app = QtWidgets.QApplication(sys.argv)
  window = Main()
  window.show()
  sys.exit(app.exec_())



