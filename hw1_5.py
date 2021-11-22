from PyQt5 import QtWidgets
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *

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
        print('file loaded')
        
        self.b1.clicked.connect(self.no1)
        #self.b2.clicked.connect(self.no2)
        self.b3.clicked.connect(self.no3)
        self.b4.clicked.connect(self.no4)
        #self.b5.clicked.connect(self.no5)

    def no1(self):
        #plt.gcf().set_size_inches(10,10)
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
            
    #def no2(self):
        #print('Batch Size{}',format(Batchsize variable name))
    def no3(self):
        global model
        model = Sequential()

        model.add(Conv2D(filters=64, kernel_size=(3,3), padding='same', input_shape=(32,32,3), activation='relu'))
        model.add(Conv2D(filters=64, kernel_size=(3,3), padding='same', activation='relu'))
        model.add(MaxPool2D(pool_size = (2,2)))

        model.add(Conv2D(filters=128, kernel_size=(3,3), padding='same', activation='relu'))
        model.add(Conv2D(filters=128, kernel_size=(3,3), padding='same', activation='relu'))
        model.add(MaxPool2D(pool_size = (2,2)))

        model.add(Conv2D(filters=256, kernel_size=(3,3), padding='same', activation='relu'))
        model.add(Conv2D(filters=256, kernel_size=(3,3), padding='same', activation='relu'))
        model.add(Conv2D(filters=256, kernel_size=(3,3), padding='same', activation='relu'))
        model.add(MaxPool2D(pool_size = (2,2)))

        model.add(Conv2D(filters=512, kernel_size=(3,3), padding='same', activation='relu'))
        model.add(Conv2D(filters=512, kernel_size=(3,3), padding='same', activation='relu'))
        model.add(Conv2D(filters=512, kernel_size=(3,3), padding='same', activation='relu'))
        model.add(MaxPool2D(pool_size = (2,2)))

        model.add(Conv2D(filters=512, kernel_size=(3,3), padding='same', activation='relu'))
        model.add(Conv2D(filters=512, kernel_size=(3,3), padding='same', activation='relu'))
        model.add(Conv2D(filters=512, kernel_size=(3,3), padding='same', activation='relu'))
        model.add(MaxPool2D(pool_size = (2,2)))

        model.add(Flatten())
        model.add(Dense(512, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(512, activation='relu'))
        model.add(Dropout(0.25))
        model.add(Dense(10, activation='softmax'))
        
        model.summary()
        print("")
    def no4(self):
        trainNor = x_train/255
        testNor = x_test/255
        trainOnehot = to_categorical(y_train)
        testOnehot = to_categorical(y_test)

        model.compile(loss='categorical_crossentropy', optimizer='SGD', metrics=['accuracy'])
        train_history = model.fit(trainNor,trainOnehot, validation_split=0.1, epochs = 20, batch_size = 32, verbose=1)
        def show_train_history(train_history,train,validation):
            plt.plot(train_history.history[train])
            plt.plot(train_history.history[validation])
            plt.title('Train History')
            plt.ylabel(train)
            plt.xlabel('Epoch')
            plt.legend(['train','validation'],loc='upper left')
            plt.show()

        show_train_history(train_history,'accuracy','val_accuracy')
        show_train_history(train_history,'loss','val_loss')
    #def no5(self):

if __name__ == '__main__':
  import sys
  app = QtWidgets.QApplication(sys.argv)
  window = Main()
  window.show()
  sys.exit(app.exec_())
