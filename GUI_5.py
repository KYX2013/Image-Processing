# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'GUI_5.ui'
#
# Created by: PyQt5 UI code generator 5.15.4
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_HW1_5(object):
    def setupUi(self, HW1_5):
        HW1_5.setObjectName("HW1_5")
        HW1_5.resize(278, 292)
        self.groupBox = QtWidgets.QGroupBox(HW1_5)
        self.groupBox.setGeometry(QtCore.QRect(10, 0, 261, 281))
        self.groupBox.setObjectName("groupBox")
        self.b1 = QtWidgets.QToolButton(self.groupBox)
        self.b1.setGeometry(QtCore.QRect(20, 30, 221, 31))
        self.b1.setCheckable(True)
        self.b1.setObjectName("b1")
        self.b2 = QtWidgets.QToolButton(self.groupBox)
        self.b2.setGeometry(QtCore.QRect(20, 70, 221, 31))
        self.b2.setCheckable(True)
        self.b2.setObjectName("b2")
        self.b3 = QtWidgets.QToolButton(self.groupBox)
        self.b3.setGeometry(QtCore.QRect(20, 110, 221, 31))
        self.b3.setCheckable(True)
        self.b3.setObjectName("b3")
        self.b4 = QtWidgets.QToolButton(self.groupBox)
        self.b4.setGeometry(QtCore.QRect(20, 150, 221, 31))
        self.b4.setCheckable(True)
        self.b4.setObjectName("b4")
        self.b5 = QtWidgets.QToolButton(self.groupBox)
        self.b5.setGeometry(QtCore.QRect(20, 230, 221, 31))
        self.b5.setCheckable(True)
        self.b5.setObjectName("b5")
        self.lineEdit = QtWidgets.QLineEdit(self.groupBox)
        self.lineEdit.setGeometry(QtCore.QRect(20, 190, 221, 31))
        font = QtGui.QFont()
        font.setFamily("Microsoft YaHei UI")
        font.setPointSize(12)
        font.setBold(True)
        font.setWeight(75)
        self.lineEdit.setFont(font)
        self.lineEdit.setObjectName("lineEdit")
        self.b1.raise_()
        self.b3.raise_()
        self.b4.raise_()
        self.b2.raise_()
        self.b5.raise_()
        self.lineEdit.raise_()

        self.retranslateUi(HW1_5)
        QtCore.QMetaObject.connectSlotsByName(HW1_5)

    def retranslateUi(self, HW1_5):
        _translate = QtCore.QCoreApplication.translate
        HW1_5.setWindowTitle(_translate("HW1_5", "Dialog"))
        self.groupBox.setTitle(_translate("HW1_5", "VGG16 TEST"))
        self.b1.setText(_translate("HW1_5", "1. Show Train Images"))
        self.b2.setText(_translate("HW1_5", "2. Show HyperParameter"))
        self.b3.setText(_translate("HW1_5", "3. Show Model Shortcut"))
        self.b4.setText(_translate("HW1_5", "4. Show Accuracy"))
        self.b5.setText(_translate("HW1_5", "5. Test"))


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    HW1_5 = QtWidgets.QDialog()
    ui = Ui_HW1_5()
    ui.setupUi(HW1_5)
    HW1_5.show()
    sys.exit(app.exec_())
