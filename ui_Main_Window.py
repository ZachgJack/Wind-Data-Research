# -*- coding: utf-8 -*-

################################################################################
## Form generated from reading UI file 'mainwindow.ui'
##
## Created by: Qt User Interface Compiler version 6.9.1
##
## WARNING! All changes made in this file will be lost when recompiling UI file!
################################################################################

from PySide6.QtCore import (QCoreApplication, QDate, QDateTime, QLocale,
    QMetaObject, QObject, QPoint, QRect,
    QSize, QTime, QUrl, Qt)
from PySide6.QtGui import (QBrush, QColor, QConicalGradient, QCursor,
    QFont, QFontDatabase, QGradient, QIcon,
    QImage, QKeySequence, QLinearGradient, QPainter,
    QPalette, QPixmap, QRadialGradient, QTransform)
from PySide6.QtWidgets import (QApplication, QDateEdit, QFrame, QLabel,
    QMainWindow, QMenuBar, QPushButton, QSizePolicy,
    QStatusBar, QVBoxLayout, QWidget)

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        if not MainWindow.objectName():
            MainWindow.setObjectName(u"MainWindow")
        MainWindow.resize(1111, 871)
        self.centralwidget = QWidget(MainWindow)
        self.centralwidget.setObjectName(u"centralwidget")
        self.pushButton = QPushButton(self.centralwidget)
        self.pushButton.setObjectName(u"pushButton")
        self.pushButton.setGeometry(QRect(120, 390, 101, 30))
        self.terminalPlaceholder = QFrame(self.centralwidget)
        self.terminalPlaceholder.setObjectName(u"terminalPlaceholder")
        self.terminalPlaceholder.setGeometry(QRect(10, 440, 331, 371))
        self.terminalPlaceholder.setFrameShape(QFrame.Shape.StyledPanel)
        self.terminalPlaceholder.setFrameShadow(QFrame.Shadow.Raised)
        self.verticalLayout = QVBoxLayout(self.terminalPlaceholder)
        self.verticalLayout.setObjectName(u"verticalLayout")
        self.canvasPlaceholder = QFrame(self.centralwidget)
        self.canvasPlaceholder.setObjectName(u"canvasPlaceholder")
        self.canvasPlaceholder.setGeometry(QRect(350, 10, 751, 811))
        self.canvasPlaceholder.setFrameShape(QFrame.Shape.StyledPanel)
        self.canvasPlaceholder.setFrameShadow(QFrame.Shadow.Raised)
        self.verticalLayout_2 = QVBoxLayout(self.canvasPlaceholder)
        self.verticalLayout_2.setObjectName(u"verticalLayout_2")
        self.lossPlaceholder = QFrame(self.canvasPlaceholder)
        self.lossPlaceholder.setObjectName(u"lossPlaceholder")
        self.lossPlaceholder.setFrameShape(QFrame.Shape.StyledPanel)
        self.lossPlaceholder.setFrameShadow(QFrame.Shadow.Raised)
        self.verticalLayout_5 = QVBoxLayout(self.lossPlaceholder)
        self.verticalLayout_5.setObjectName(u"verticalLayout_5")

        self.verticalLayout_2.addWidget(self.lossPlaceholder)

        self.predictPlaceholder = QFrame(self.canvasPlaceholder)
        self.predictPlaceholder.setObjectName(u"predictPlaceholder")
        self.predictPlaceholder.setFrameShape(QFrame.Shape.StyledPanel)
        self.predictPlaceholder.setFrameShadow(QFrame.Shadow.Raised)
        self.verticalLayout_6 = QVBoxLayout(self.predictPlaceholder)
        self.verticalLayout_6.setObjectName(u"verticalLayout_6")

        self.verticalLayout_2.addWidget(self.predictPlaceholder)

        self.dropPlaceholder = QFrame(self.centralwidget)
        self.dropPlaceholder.setObjectName(u"dropPlaceholder")
        self.dropPlaceholder.setGeometry(QRect(10, 10, 331, 101))
        self.dropPlaceholder.setFrameShape(QFrame.Shape.StyledPanel)
        self.dropPlaceholder.setFrameShadow(QFrame.Shadow.Raised)
        self.verticalLayout_3 = QVBoxLayout(self.dropPlaceholder)
        self.verticalLayout_3.setObjectName(u"verticalLayout_3")
        self.datePlaceholder = QFrame(self.centralwidget)
        self.datePlaceholder.setObjectName(u"datePlaceholder")
        self.datePlaceholder.setGeometry(QRect(110, 190, 121, 151))
        self.datePlaceholder.setFrameShape(QFrame.Shape.StyledPanel)
        self.datePlaceholder.setFrameShadow(QFrame.Shadow.Raised)
        self.verticalLayout_4 = QVBoxLayout(self.datePlaceholder)
        self.verticalLayout_4.setObjectName(u"verticalLayout_4")
        self.label_3 = QLabel(self.datePlaceholder)
        self.label_3.setObjectName(u"label_3")

        self.verticalLayout_4.addWidget(self.label_3)

        self.label_2 = QLabel(self.datePlaceholder)
        self.label_2.setObjectName(u"label_2")

        self.verticalLayout_4.addWidget(self.label_2)

        self.dateFrom = QDateEdit(self.datePlaceholder)
        self.dateFrom.setObjectName(u"dateFrom")

        self.verticalLayout_4.addWidget(self.dateFrom)

        self.label = QLabel(self.datePlaceholder)
        self.label.setObjectName(u"label")

        self.verticalLayout_4.addWidget(self.label)

        self.dateTo = QDateEdit(self.datePlaceholder)
        self.dateTo.setObjectName(u"dateTo")

        self.verticalLayout_4.addWidget(self.dateTo)

        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QMenuBar(MainWindow)
        self.menubar.setObjectName(u"menubar")
        self.menubar.setGeometry(QRect(0, 0, 1111, 21))
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QStatusBar(MainWindow)
        self.statusbar.setObjectName(u"statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)

        QMetaObject.connectSlotsByName(MainWindow)
    # setupUi

    def retranslateUi(self, MainWindow):
        MainWindow.setWindowTitle(QCoreApplication.translate("MainWindow", u"MainWindow", None))
        self.pushButton.setText(QCoreApplication.translate("MainWindow", u"Run LSTM", None))
        self.label_3.setText(QCoreApplication.translate("MainWindow", u"Pick a date range:", None))
        self.label_2.setText(QCoreApplication.translate("MainWindow", u"From", None))
        self.label.setText(QCoreApplication.translate("MainWindow", u"To", None))
    # retranslateUi
