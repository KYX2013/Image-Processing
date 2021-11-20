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
        self.toolButton_9.clicked.connect(self.hw3_1)
        self.toolButton_10.clicked.connect(self.hw3_2)
        self.toolButton_11.clicked.connect(self.hw3_3)
        self.toolButton_12.clicked.connect(self.hw3_4)
        self.toolButton_13.clicked.connect(self.hw4_1)
        self.toolButton_14.clicked.connect(self.hw4_2)
        self.toolButton_15.clicked.connect(self.hw4_3)
        self.toolButton_16.clicked.connect(self.hw4_4)
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
        imgGray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        Ginit = [(-1,1),(0,1),(1,1),
                 (-1,0),(0,0),(1,0),
                 (-1,-1),(0,-1),(1,-1)]
        Gfilter = Ginit
        SUM = 0
        for i in range(9):
            (x,y) = Ginit[i]
            Gfilter[i] = np.exp(-(x**2+y**2)/(2*0.5))/(2*np.pi*0.5) #取σ^2=0.5
            SUM += Gfilter[i]
        Gnorm = np.array(Gfilter)/SUM
        global Gblur
        Gblur = np.array(imgGray).copy()
        for i in range(1,Gblur.shape[0]-1):
            for j in range(1,Gblur.shape[1]-1):
                temp = 0
                for a in range(i-1,i+2):
                    for b in range(j-1,j+2):
                        temp += imgGray[a][b]*Gnorm[(a-i+1)*3+(b-j+1)]
                Gblur[i][j] = temp
        cv2.imshow('Gaussian Blur',Gblur)
    def hw3_2(self):
        Xfilter = [-1,0,1,
                   -2,0,2,
                   -1,0,1]
        Xnorm = np.array(Xfilter)
        global sobX
        sobX = np.array(Gblur).copy()/255
        for i in range(1,Gblur.shape[0]-1):
            for j in range(1,Gblur.shape[1]-1):
                temp = 0
                for a in range(i-1,i+2):
                    for b in range(j-1,j+2):
                        temp += Gblur[a][b]*Xnorm[(a-i+1)*3+(b-j+1)]
                sobX[i][j] = temp/255
        cv2.imshow('Sobel X',sobX)
    def hw3_3(self):
        Yfilter = [1,2,1,
                   0,0,0,
                   -1,-2,-1]
        Ynorm = np.array(Yfilter)
        global sobY
        sobY = np.array(Gblur).copy()/255
        for i in range(1,Gblur.shape[0]-1):
            for j in range(1,Gblur.shape[1]-1):
                temp = 0
                for a in range(i-1,i+2):
                    for b in range(j-1,j+2):
                        temp += Gblur[a][b]*Ynorm[(a-i+1)*3+(b-j+1)]
                sobY[i][j] = temp/255
        cv2.imshow('Sobel Y',sobY)
    def hw3_4(self):
        sobXnp = np.array(sobX)
        sobYnp = np.array(sobY)
        Mag = ((sobXnp**2+sobYnp**2))**(0.5)
        cv2.imshow('Magnitude',Mag)
    def hw4_1(self):
        img = cv2.imread('Dataset_OpenCvDl_Hw1\Q4_Image\SQUARE-01.png')
        reHeight = 256
        reWidth = 256
        global imgRe
        imgRe = cv2.resize(img,(reHeight,reWidth),interpolation=cv2.INTER_AREA)
        cv2.imshow('img_Resized',imgRe)
    def hw4_2(self):
        global winX,winY
        winX,winY=400,300
        global Xmov,Ymov
        Xmov,Ymov = 0,60
        m = np.float32([[1,0,Xmov],[0,1,Ymov]])
        global imgTr
        imgTr = cv2.warpAffine(imgRe,m,(winX,winY))
        cv2.imshow('img_Resized',imgRe)
        cv2.imshow('img_Translation',imgTr)
    def hw4_3(self):
        angle = 10
        scale = 0.5
        m = cv2.getRotationMatrix2D((imgTr.shape[0]/2,imgTr.shape[1]/2),10,0.5)
        global imgRo
        imgRo = cv2.warpAffine(imgTr,m,(winX,winY))
        cv2.imshow('img_Rotation,Scaling',imgRo)
    def hw4_4(self):
        oldL = [[50,50],
                [200,50],
                [50,200]]
        newL = [[10,100],
                [200,50],
                [100,250]]
        m = cv2.getAffineTransform(np.float32(oldL),np.float32(newL))
        imgSh = cv2.warpAffine(imgRo,m,(winX,winY))
        cv2.imshow('img_Shearing',imgSh)

if __name__ == '__main__':
  import sys
  app = QtWidgets.QApplication(sys.argv)
  window = Main()
  window.show()
  sys.exit(app.exec_())



