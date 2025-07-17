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
from PySide6.QtWidgets import (QApplication, QCheckBox, QDateEdit, QFrame,
    QLabel, QMainWindow, QPushButton, QSizePolicy,
    QTabWidget, QVBoxLayout, QWidget)

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        if not MainWindow.objectName():
            MainWindow.setObjectName(u"MainWindow")
        MainWindow.resize(1110, 869)
#if QT_CONFIG(whatsthis)
        MainWindow.setWhatsThis(u"")
#endif // QT_CONFIG(whatsthis)
        MainWindow.setStyleSheet(u"background-color: qlineargradient(spread:pad, x1:0, y1:0, x2:0.702, y2:1, stop:0 rgba(74, 170, 0, 255), stop:0.88764 rgba(255, 255, 255, 255));")
        self.centralwidget = QWidget(MainWindow)
        self.centralwidget.setObjectName(u"centralwidget")
        self.centralwidget.setStyleSheet(u"")
        self.tabWidget = QTabWidget(self.centralwidget)
        self.tabWidget.setObjectName(u"tabWidget")
        self.tabWidget.setGeometry(QRect(0, 0, 1111, 871))
        self.tabWidget.setStyleSheet(u"border-color: rgb(0, 0, 0);")
        self.CSVtab = QWidget()
        self.CSVtab.setObjectName(u"CSVtab")
#if QT_CONFIG(whatsthis)
        self.CSVtab.setWhatsThis(u"")
#endif // QT_CONFIG(whatsthis)
        self.CSVtab.setStyleSheet(u"")
        self.datePlaceholder = QFrame(self.CSVtab)
        self.datePlaceholder.setObjectName(u"datePlaceholder")
        self.datePlaceholder.setGeometry(QRect(100, 170, 121, 151))
        self.datePlaceholder.setStyleSheet(u"background-color: rgb(255, 255, 255);\n"
"border-color: rgb(0, 0, 0);\n"
"color: rgb(0, 0, 0);")
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

        self.pushButton = QPushButton(self.CSVtab)
        self.pushButton.setObjectName(u"pushButton")
        self.pushButton.setGeometry(QRect(110, 390, 101, 30))
        self.pushButton.setStyleSheet(u"color: rgb(0, 0, 0);\n"
"background-color: rgb(255, 255, 255);")
        self.terminalPlaceholder = QFrame(self.CSVtab)
        self.terminalPlaceholder.setObjectName(u"terminalPlaceholder")
        self.terminalPlaceholder.setGeometry(QRect(10, 450, 321, 371))
        self.terminalPlaceholder.setStyleSheet(u"background-color: rgb(2, 2, 2);\n"
"color: rgb(0, 255, 0);")
        self.terminalPlaceholder.setFrameShape(QFrame.Shape.StyledPanel)
        self.terminalPlaceholder.setFrameShadow(QFrame.Shadow.Raised)
        self.verticalLayout = QVBoxLayout(self.terminalPlaceholder)
        self.verticalLayout.setObjectName(u"verticalLayout")
        self.canvasPlaceholder = QFrame(self.CSVtab)
        self.canvasPlaceholder.setObjectName(u"canvasPlaceholder")
        self.canvasPlaceholder.setGeometry(QRect(340, 20, 751, 811))
        self.canvasPlaceholder.setStyleSheet(u"background-color: rgb(255, 255, 255);\n"
"color: rgb(0, 0, 0);")
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

        self.dropPlaceholder = QFrame(self.CSVtab)
        self.dropPlaceholder.setObjectName(u"dropPlaceholder")
        self.dropPlaceholder.setGeometry(QRect(20, 20, 291, 101))
        self.dropPlaceholder.setStyleSheet(u"background-color: rgb(143, 143, 143);")
        self.dropPlaceholder.setFrameShape(QFrame.Shape.StyledPanel)
        self.dropPlaceholder.setFrameShadow(QFrame.Shadow.Raised)
        self.verticalLayout_3 = QVBoxLayout(self.dropPlaceholder)
        self.verticalLayout_3.setObjectName(u"verticalLayout_3")
        self.targetToggle = QCheckBox(self.CSVtab)
        self.targetToggle.setObjectName(u"targetToggle")
        self.targetToggle.setGeometry(QRect(110, 340, 101, 31))
        self.targetToggle.setStyleSheet(u"background-color: rgb(255, 255, 255);\n"
"color: rgb(0, 0, 0);")
        self.tabWidget.addTab(self.CSVtab, "")
        self.APItab = QWidget()
        self.APItab.setObjectName(u"APItab")
        self.tabWidget.addTab(self.APItab, "")
        self.Instructionstab = QWidget()
        self.Instructionstab.setObjectName(u"Instructionstab")
        self.tabWidget.addTab(self.Instructionstab, "")
        MainWindow.setCentralWidget(self.centralwidget)

        self.retranslateUi(MainWindow)

        self.tabWidget.setCurrentIndex(0)


        QMetaObject.connectSlotsByName(MainWindow)
    # setupUi

    def retranslateUi(self, MainWindow):
        MainWindow.setWindowTitle(QCoreApplication.translate("MainWindow", u"MainWindow", None))
        self.label_3.setText(QCoreApplication.translate("MainWindow", u"Pick a date range:", None))
        self.label_2.setText(QCoreApplication.translate("MainWindow", u"From", None))
        self.label.setText(QCoreApplication.translate("MainWindow", u"To", None))
        self.pushButton.setText(QCoreApplication.translate("MainWindow", u"Run LSTM", None))
        self.targetToggle.setText(QCoreApplication.translate("MainWindow", u"Analyze Solar", None))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.CSVtab), QCoreApplication.translate("MainWindow", u"Train With CSV", None))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.APItab), QCoreApplication.translate("MainWindow", u"Train With API", None))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.Instructionstab), QCoreApplication.translate("MainWindow", u"Instructions", None))
    # retranslateUi
