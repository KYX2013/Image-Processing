from PyQt5 import QtWidgets
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *

import cv2
import GUI_5 as ui
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D, MaxPooling2D
from tensorflow.keras.utils import to_categorical

import ssl

class Main(QDialog, ui.Ui_Dialog):
    def __init__(self):
        super().__init__()
        self.setupUi(self)
        
        #ssl._create_default_https_context = ssl._create_unverified_context
        global x_train, y_train, x_test, y_test
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
        print('data loaded')
        
        self.b1.clicked.connect(self.no1)
        self.b2.clicked.connect(self.no2)
        self.b3.clicked.connect(self.no3)
        self.b4.clicked.connect(self.no4)
        self.b5.clicked.connect(self.no5)

    def no1(self):
        global label
        label = {"[0]":'airplane',"[1]":'automobile',"[2]":'bird',"[3]":'cat',"[4]":'deer',
                 "[5]":'dog',"[6]":'frog',"[7]":'horse',"[8]":'ship',"[9]":'truck'}
        idx = np.random.randint(0,49991)
        for i in range(idx,idx+9):
            img = plt.subplot(3,3,i-idx+1)
            img.imshow(x_train[i],cmap='binary')
            title = label.get(str(y_train[i]))
            img.set_title(title,fontsize=10)
            img.set_xticks([]);img.set_yticks([])
        plt.show()
            
    def no2(self):
        global BatchSize,Opt
        BatchSize = 32;LR = 0.01;Opt = keras.optimizers.SGD(learning_rate=LR)
        print('hyperparameters:')
        print('batch size: ',format(BatchSize))
        print('learning rate: ',format(LR))
        print('optimizer: ',format('SGD'))
    def no3(self):
        global model
        model = Sequential()

        model.add(Conv2D(filters=64, kernel_size=(3,3), padding='same', input_shape=(32,32,3), activation='relu'))
        model.add(Conv2D(filters=64, kernel_size=(3,3), padding='same', activation='relu'))
        model.add(MaxPool2D(pool_size = (2,2)))

        model.add(Conv2D(filters=128, kernel_size=(3,3), padding='same', activation='relu'))
        model.add(Conv2D(filters=128, kernel_size=(3,3), padding='same', activation='relu'))
        #model.add(MaxPool2D(pool_size = (2,2)))

        model.add(Conv2D(filters=256, kernel_size=(3,3), padding='same', activation='relu'))
        model.add(Conv2D(filters=256, kernel_size=(3,3), padding='same', activation='relu'))
        model.add(Conv2D(filters=256, kernel_size=(3,3), padding='same', activation='relu'))
        model.add(MaxPool2D(pool_size = (2,2)))

        model.add(Conv2D(filters=512, kernel_size=(3,3), padding='same', activation='relu'))
        model.add(Conv2D(filters=512, kernel_size=(3,3), padding='same', activation='relu'))
        model.add(Conv2D(filters=512, kernel_size=(3,3), padding='same', activation='relu'))
        #model.add(MaxPool2D(pool_size = (2,2)))

        model.add(Conv2D(filters=512, kernel_size=(3,3), padding='same', activation='relu'))
        model.add(Conv2D(filters=512, kernel_size=(3,3), padding='same', activation='relu'))
        model.add(Conv2D(filters=512, kernel_size=(3,3), padding='same', activation='relu'))
        model.add(MaxPool2D(pool_size = (2,2)))

        model.add(Flatten())
        model.add(Dense(512, activation='relu'))
        model.add(Dropout(0.25))
        model.add(Dense(512, activation='relu'))
        model.add(Dropout(0.25))
        model.add(Dense(10, activation='softmax'))
        
        model.summary()
    def no4(self):
        '''
        trainNor = x_train/255
        trainOnehot = to_categorical(y_train)

        model.compile(loss='categorical_crossentropy', optimizer=Optimizer, metrics=['accuracy'])
        train_history = model.fit(trainNor,trainOnehot, validation_split=0.2, epochs = 20, batch_size = BatchSize, verbose=1)
        
        plt.subplot(2,1,1)
        plt.plot(train_history.history['accuracy'])
        plt.plot(train_history.history['val_accuracy'])
        plt.ylabel('%')
        plt.xlabel('epoch')
        plt.legend(['Training','Testing'],loc='lower right')
        plt.subplot(2,1,2)
        plt.plot(train_history.history['loss'])
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.show()
        '''
        hw5_4 = cv2.imread('hw1_5_4/20211124.jpg')
        cv2.imshow('hw5_4',hw5_4)
        
    def no5(self):
        testNor = x_test/255
        testOnehot = to_categorical(y_test)
        modelSaved = keras.models.load_model('model\saved_model')
        print('model loaded')
        prediction = np.argmax(modelSaved.predict(testNor),axis=-1)
        
        testNo = int(self.lineEdit.text())
        count = [0 for i in range(10)]
        for i in range(len(y_test)):
          if y_test[i] == y_test[testNo]:
            count[prediction[i]] += 1
        plt.figure(1)
        plt.title('label='+label.get(str(y_test[testNo])))
        plt.imshow(x_test[testNo],cmap='binary')
        plt.figure(2)
        x = [i for i in range(10)]
        xTicks = ['plane','car','bird','cat','deer','dog','frog','horse','ship','truck']
        plt.bar(x, count, width=0.85, bottom=None, align='center', data=None, color='black')
        plt.xticks(x,xTicks)
        plt.xlabel('prediction')
        plt.show()

if __name__ == '__main__':
  import sys
  app = QtWidgets.QApplication(sys.argv)
  window = Main()
  window.show()
  sys.exit(app.exec_())
