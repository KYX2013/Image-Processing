from PyQt5 import QtWidgets
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *

import GUI_5 as ui

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
