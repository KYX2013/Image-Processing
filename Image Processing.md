# Image Processing
---
## 題外話
### GUI的生成(PyQt)
先用 Qt Designer 製作想要的模板    [Reference](https://clay-atlas.com/blog/2019/08/26/python-chinese-pyqt5-tutorial-install/)
![](https://i.imgur.com/A57GK6f.png)
* Main Window
* Dialog
* Widget

完成之後在cmd下指令(切到目標dir)
```
pyuic5 -x filename.ui -o filename.py
```
```
python filename.py
```
成功的話這裡會跳出視窗
![](https://i.imgur.com/5XxG6B4.jpg)

寫一個控制的主程式 **( main.py )**
```python=
from PyQt5 import QtWidgets
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *

import GUI as ui  #GUI.py為剛剛上面用GUI.ui產生出來的

class Main(QDialog, ui.Ui_Dialog):
  def __init__(self):
    super().__init__()
    self.setupUi(self)

if __name__ == '__main__':
  import sys
  app = QtWidgets.QApplication(sys.argv)
  window = Main()
  window.show()
  sys.exit(app.exec_())
```
之後要控制UI上的物件，例如:按鈕觸發的程式，也是加在這個主程式

---
## HW1
### 1.1 Load Image
利用OpenCV讀取圖片
[Reference](https://blog.gtwang.org/programming/opencv-basic-image-read-and-write-tutorial/)
```python=
import numpy as np
import cv2

img = cv2.imread('image.jpg')

#check:
type(img)
img.shape

# 以灰階的方式讀取圖檔
img_gray = cv2.imread('image.jpg', cv2.IMREAD_GRAYSCALE)

# 顯示圖片
cv2.imshow('視窗名稱', img)

# 關閉特定視窗
cv2.destroyWindow('視窗名稱')

# 按下任意鍵則關閉所有視窗
cv2.waitKey(0)
cv2.destroyAllWindows()

# 讓視窗可以自由縮放大小
cv2.namedWindow('視窗名稱', cv2.WINDOW_NORMAL)
```
遇到問題: 已經設定環境參數，但是還是無法import cv2
```
解決辦法:
改成直接用cmd下載opencv

pip install opencv-python
```
### 1.2 Color Separation
利用openCV內建function(cv2.split(img))
```python=
img = cv2.imread("Path")

(imgB,imgG,imgR) = cv2.split(img)

#補零是為了讓圖片不變灰階
zero = np.zeros(img.shape[:2],dtype = "uint8")
cv2.imshow("B channel",cv2.merge([imgB,zero,zero]))
cv2.imshow("G channel",cv2.merge([zero,imgG,zero]))
cv2.imshow("R channel",cv2.merge([zero,zero,imgR]))

cv2.imshow()
```

RBG圖片的shape = (height, width, 3)
灰階圖片的shape = (height, width)
### 1.3 Color Transformation
* Method 1
用openCV內建function(cv2.cvtColor(img,cv2.COLOR_BGR2GRAY))
formula: 0.07*B+0.72*G+0.21*R
```python=
img = cv2.imread('Dataset_OpenCvDl_Hw1\Q1_Image\Sun.jpg')
imgI1 = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
cv2.imshow("Perceptually weighted formula",imgI1)
```
* Method 2
自訂比例(ex. (B+G+R)/3)
```python=
img = cv2.imread('Dataset_OpenCvDl_Hw1\Q1_Image\Sun.jpg')
(imgB,imgG,imgR) = cv2.split(img)
imgI2 = imgB[:]//3+imgG[:]//3+imgR[:]//3
cv2.imshow("Average weighted formula",imgI2)
```
### 1.4 Blending
用openCV內建function
* cv2.createTrackbar
cv2.createTrackbar('參數名稱','視窗名稱',最小值,最大值,回傳函數)
用回傳函數來更新
* cv2.addWeighted
cv2.addWeighted(image1,alpha1,image2,alpha2,gamma)
gamma是附加值(效果類似亮度)
```python=
imgS = cv2.imread('Dataset_OpenCvDl_Hw1\Q1_Image\Dog_Strong.jpg')
        imgW = cv2.imread('Dataset_OpenCvDl_Hw1\Q1_Image\Dog_Weak.jpg')
        def bar(x):
            global barPos
            barPos = cv2.getTrackbarPos('Blend','DOG')
            imgDog = cv2.addWeighted(imgS,(255-barPos)/255,imgW,barPos/255,0)
            cv2.imshow('DOG',imgDog)
        cv2.namedWindow('DOG')
        cv2.createTrackbar('Blend','DOG',0,255,bar)
        cv2.setTrackbarPos('Blend','DOG',127)
```
### 2.1 Gaussian Blur
Smoothing([reference](https://chtseng.wordpress.com/2016/11/17/python-%E8%88%87-opencv-%E6%A8%A1%E7%B3%8A%E8%99%95%E7%90%86/)) using Gaussian Blur
![](https://i.imgur.com/PCDKPPz.png)
直接call內建function(cv2.GaussianBlur())
```python=
img = cv2.imread('Dataset_OpenCvDl_Hw1\Q2_Image\Lenna_whiteNoise.jpg')
blur = cv2.GaussianBlur(img,(5,5),0)
cv2.imshow('whiteNoise',img)
cv2.imshow('Gaussian',blur)
```
### 2.2 Bilateral filter
Smoothing using Bilateral filter
![](https://i.imgur.com/APkLwKv.png)
直接call內建function(cv2.bilateralFilter())
```python=
img = cv2.imread('Dataset_OpenCvDl_Hw1\Q2_Image\Lenna_whiteNoise.jpg')
blur = cv2.bilateralFilter(img,9,90,90)
cv2.imshow('whiteNoise',img)
cv2.imshow('Bilateral',blur)
```
### 2.3 Median filter
Smoothing using Median filter
![](https://i.imgur.com/WS1fkt7.png)
直接call內建function(cv2.medianBlur())
```python=
img = cv2.imread('Dataset_OpenCvDl_Hw1\Q2_Image\Lenna_pepperSalt.jpg')
blur3 = cv2.medianBlur(img,3)
blur5 = cv2.medianBlur(img,5)
cv2.imshow('pepperSalt',img)
cv2.imshow('Median(3x3)',blur3)
cv2.imshow('Median(5x5)',blur5)
```
### 3.1 Gaussian Blur
### 3.2 Sobel X
### 3.3 Sobel Y
### 3.4 Magnitude
### 4.1 Resize
### 4.2 Translation
### 4.3 Rotation, Scaling
### 4.4 Shearing
---
## HW2