
from PyQt5.QtWidgets import QApplication, QMessageBox
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from PyQt5 import QtCore, QtGui, QtWidgets
import cv2 
import sys
import numpy as np 
from PyQt5.QtWidgets import *
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import Qt
from matplotlib import pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from segcoloration import ColorSegmentationApp



class Ui_ImageProcessingAPP(object):
    def setupUi(self, ImageProcessingAPP):
        ImageProcessingAPP.setObjectName("ImageProcessingAPP")
        ImageProcessingAPP.resize(750, 550)
        ImageProcessingAPP.setStyleSheet("color: rgb(100, 100, 100);\n"
"color: rgb(0, 0, 0);")
        self.centralwidget = QtWidgets.QWidget(ImageProcessingAPP)
        self.centralwidget.setObjectName("centralwidget")
        self.verticalLayout = QtWidgets.QVBoxLayout(self.centralwidget)
        self.verticalLayout.setObjectName("verticalLayout")
        self.stackedWidget = QtWidgets.QStackedWidget(self.centralwidget)
        font = QtGui.QFont()
        font.setPointSize(12)
        font.setBold(True)
        font.setWeight(75)
        self.stackedWidget.setFont(font)
        self.stackedWidget.setStyleSheet("background-color: rgb(226, 226, 226);")
        self.stackedWidget.setObjectName("stackedWidget")
        self.homepage = QtWidgets.QWidget()
        self.homepage.setObjectName("homepage")
        self.firstimg = QtWidgets.QLabel(self.homepage)
        self.firstimg.setGeometry(QtCore.QRect(110, 50, 351, 261))
        self.firstimg.setStyleSheet("background-color: rgb(0, 0, 0);\n"
"color: rgb(255, 255, 255);")
        self.firstimg.setScaledContents(True)
        self.firstimg.setObjectName("firstimg")
        self.uploadbutton = QtWidgets.QPushButton(self.homepage)
        self.uploadbutton.setGeometry(QtCore.QRect(230, 360, 71, 31))
        font = QtGui.QFont()
        font.setBold(True)
        font.setWeight(75)
        self.uploadbutton.setFont(font)
        self.uploadbutton.setStyleSheet("background-color: rgb(0, 170, 255);")
        self.uploadbutton.setObjectName("uploadbutton")
        self.stackedWidget.addWidget(self.homepage)
        self.grayconv = QtWidgets.QWidget()
        self.grayconv.setObjectName("grayconv")
        self.beforegray = QtWidgets.QLabel(self.grayconv)
        self.beforegray.setGeometry(QtCore.QRect(320, 120, 231, 221))
        self.beforegray.setStyleSheet("background-color: rgb(0, 0, 0);")
        self.beforegray.setText("")
        self.beforegray.setScaledContents(True)
        self.beforegray.setObjectName("beforegray")
        self.aftergray = QtWidgets.QLabel(self.grayconv)
        self.aftergray.setGeometry(QtCore.QRect(30, 120, 241, 221))
        self.aftergray.setStyleSheet("background-color: rgb(0, 0, 0);")
        self.aftergray.setText("")
        self.aftergray.setScaledContents(True)
        self.aftergray.setObjectName("aftergray")
        self.upload_gray = QtWidgets.QPushButton(self.grayconv)
        self.upload_gray.setGeometry(QtCore.QRect(60, 390, 71, 31))
        font = QtGui.QFont()
        font.setBold(True)
        font.setWeight(75)
        self.upload_gray.setFont(font)
        self.upload_gray.setStyleSheet("background-color: rgb(0, 170, 255);")
        self.upload_gray.setObjectName("upload_gray")
        self.savegray = QtWidgets.QPushButton(self.grayconv)
        self.savegray.setGeometry(QtCore.QRect(410, 390, 71, 31))
        font = QtGui.QFont()
        font.setBold(True)
        font.setWeight(75)
        self.savegray.setFont(font)
        self.savegray.setStyleSheet("background-color: rgb(170, 0, 255);")
        self.savegray.setObjectName("savegray")
        self.label_6 = QtWidgets.QLabel(self.grayconv)
        self.label_6.setGeometry(QtCore.QRect(90, 60, 101, 41))
        self.label_6.setObjectName("label_6")
        self.label_7 = QtWidgets.QLabel(self.grayconv)
        self.label_7.setGeometry(QtCore.QRect(370, 60, 101, 41))
        self.label_7.setObjectName("label_7")
        self.label_2 = QtWidgets.QLabel(self.grayconv)
        self.label_2.setGeometry(QtCore.QRect(170, 10, 211, 41))
        font = QtGui.QFont()
        font.setPointSize(12)
        font.setBold(True)
        font.setWeight(75)
        self.label_2.setFont(font)
        self.label_2.setStyleSheet("color: rgb(15, 15, 15);")
        self.label_2.setObjectName("label_2")
        self.applygray = QtWidgets.QPushButton(self.grayconv)
        self.applygray.setGeometry(QtCore.QRect(230, 390, 81, 31))
        font = QtGui.QFont()
        font.setBold(True)
        font.setWeight(75)
        self.applygray.setFont(font)
        self.applygray.setStyleSheet("background-color: rgb(85, 255, 255);")
        self.applygray.setObjectName("applygray")
        self.stackedWidget.addWidget(self.grayconv)
        self.histogram = QtWidgets.QWidget()
        self.histogram.setObjectName("histogram")
        self.before_hist = QtWidgets.QLabel(self.histogram)
        self.before_hist.setGeometry(QtCore.QRect(150, 120, 251, 241))
        self.before_hist.setStyleSheet("background-color: rgb(125, 125, 125);\n"
"background-color: rgb(0, 0, 0);")
        self.before_hist.setText("")
        self.before_hist.setScaledContents(True)
        self.before_hist.setObjectName("before_hist")
        self.upload_hist = QtWidgets.QPushButton(self.histogram)
        self.upload_hist.setGeometry(QtCore.QRect(140, 390, 81, 31))
        font = QtGui.QFont()
        font.setBold(True)
        font.setWeight(75)
        self.upload_hist.setFont(font)
        self.upload_hist.setStyleSheet("background-color: rgb(0, 170, 255);")
        self.upload_hist.setObjectName("upload_hist")
        self.label_178 = QtWidgets.QLabel(self.histogram)
        self.label_178.setGeometry(QtCore.QRect(210, 80, 91, 31))
        self.label_178.setStyleSheet("color: rgb(34, 34, 34);")
        self.label_178.setObjectName("label_178")
        self.label_180 = QtWidgets.QLabel(self.histogram)
        self.label_180.setGeometry(QtCore.QRect(220, 10, 111, 61))
        font = QtGui.QFont()
        font.setPointSize(12)
        font.setBold(True)
        font.setWeight(75)
        self.label_180.setFont(font)
        self.label_180.setStyleSheet("color: rgb(6, 6, 6);")
        self.label_180.setObjectName("label_180")
        self.applyhist = QtWidgets.QPushButton(self.histogram)
        self.applyhist.setGeometry(QtCore.QRect(330, 390, 81, 31))
        font = QtGui.QFont()
        font.setBold(True)
        font.setWeight(75)
        self.applyhist.setFont(font)
        self.applyhist.setStyleSheet("background-color: rgb(85, 255, 255);")
        self.applyhist.setObjectName("applyhist")
        self.stackedWidget.addWidget(self.histogram)
        self.throu = QtWidgets.QWidget()
        self.throu.setObjectName("throu")
        self.label_177 = QtWidgets.QLabel(self.throu)
        self.label_177.setGeometry(QtCore.QRect(220, 10, 121, 41))
        font = QtGui.QFont()
        font.setPointSize(12)
        font.setBold(True)
        font.setWeight(75)
        self.label_177.setFont(font)
        self.label_177.setObjectName("label_177")
        self.th = QtWidgets.QLabel(self.throu)
        self.th.setGeometry(QtCore.QRect(180, 50, 91, 41))
        self.th.setObjectName("th")
        self.threshhold = QtWidgets.QSpinBox(self.throu)
        self.threshhold.setGeometry(QtCore.QRect(260, 60, 71, 31))
        self.threshhold.setStyleSheet("background-color: rgb(162, 162, 162);")
        self.threshhold.setMaximum(255)
        self.threshhold.setObjectName("threshhold")
        self.label_5 = QtWidgets.QLabel(self.throu)
        self.label_5.setGeometry(QtCore.QRect(370, 75, 161, 31))
        self.label_5.setObjectName("label_5")
        self.label_4 = QtWidgets.QLabel(self.throu)
        self.label_4.setGeometry(QtCore.QRect(40, 80, 101, 31))
        self.label_4.setObjectName("label_4")
        self.before_b = QtWidgets.QLabel(self.throu)
        self.before_b.setGeometry(QtCore.QRect(9, 121, 281, 271))
        self.before_b.setStyleSheet("background-color: rgb(0, 0, 0);")
        self.before_b.setText("")
        self.before_b.setScaledContents(True)
        self.before_b.setObjectName("before_b")
        self.after_b = QtWidgets.QLabel(self.throu)
        self.after_b.setGeometry(QtCore.QRect(307, 116, 341, 271))
        self.after_b.setStyleSheet("background-color: rgb(0, 0, 0);")
        self.after_b.setText("")
        self.after_b.setScaledContents(True)
        self.after_b.setObjectName("after_b")
        self.Upload_b = QtWidgets.QPushButton(self.throu)
        self.Upload_b.setGeometry(QtCore.QRect(50, 400, 71, 31))
        font = QtGui.QFont()
        font.setBold(True)
        font.setWeight(75)
        self.Upload_b.setFont(font)
        self.Upload_b.setStyleSheet("background-color: rgb(0, 170, 255);")
        self.Upload_b.setObjectName("Upload_b")
        self.Binarize = QtWidgets.QPushButton(self.throu)
        self.Binarize.setGeometry(QtCore.QRect(260, 400, 71, 31))
        font = QtGui.QFont()
        font.setBold(True)
        font.setWeight(75)
        self.Binarize.setFont(font)
        self.Binarize.setStyleSheet("background-color: rgb(85, 255, 255);")
        self.Binarize.setObjectName("Binarize")
        self.save_b = QtWidgets.QPushButton(self.throu)
        self.save_b.setGeometry(QtCore.QRect(460, 400, 71, 31))
        font = QtGui.QFont()
        font.setBold(True)
        font.setWeight(75)
        self.save_b.setFont(font)
        self.save_b.setStyleSheet("background-color: rgb(162, 162, 162);\n"
"background-color: rgb(170, 0, 255);")
        self.save_b.setObjectName("save_b")
        self.stackedWidget.addWidget(self.throu)
        self.otsu = QtWidgets.QWidget()
        self.otsu.setObjectName("otsu")
        self.label_175 = QtWidgets.QLabel(self.otsu)
        self.label_175.setGeometry(QtCore.QRect(350, 50, 131, 31))
        self.label_175.setObjectName("label_175")
        self.after_b_6 = QtWidgets.QLabel(self.otsu)
        self.after_b_6.setGeometry(QtCore.QRect(298, 86, 281, 291))
        self.after_b_6.setStyleSheet("background-color: rgb(0, 0, 0);")
        self.after_b_6.setText("")
        self.after_b_6.setScaledContents(True)
        self.after_b_6.setObjectName("after_b_6")
        self.Binarize_6 = QtWidgets.QPushButton(self.otsu)
        self.Binarize_6.setGeometry(QtCore.QRect(240, 390, 91, 31))
        font = QtGui.QFont()
        font.setBold(True)
        font.setWeight(75)
        self.Binarize_6.setFont(font)
        self.Binarize_6.setStyleSheet("background-color: rgb(85, 255, 255);")
        self.Binarize_6.setObjectName("Binarize_6")
        self.save_b_6 = QtWidgets.QPushButton(self.otsu)
        self.save_b_6.setGeometry(QtCore.QRect(440, 390, 81, 31))
        font = QtGui.QFont()
        font.setBold(True)
        font.setWeight(75)
        self.save_b_6.setFont(font)
        self.save_b_6.setStyleSheet("background-color: rgb(170, 0, 255);")
        self.save_b_6.setObjectName("save_b_6")
        self.label_176 = QtWidgets.QLabel(self.otsu)
        self.label_176.setGeometry(QtCore.QRect(30, 50, 111, 31))
        self.label_176.setObjectName("label_176")
        self.Upload_b_6 = QtWidgets.QPushButton(self.otsu)
        self.Upload_b_6.setGeometry(QtCore.QRect(60, 390, 91, 31))
        font = QtGui.QFont()
        font.setBold(True)
        font.setWeight(75)
        self.Upload_b_6.setFont(font)
        self.Upload_b_6.setStyleSheet("background-color: rgb(0, 170, 255);")
        self.Upload_b_6.setObjectName("Upload_b_6")
        self.before_b_6 = QtWidgets.QLabel(self.otsu)
        self.before_b_6.setGeometry(QtCore.QRect(10, 86, 261, 291))
        self.before_b_6.setStyleSheet("background-color: rgb(0, 0, 0);")
        self.before_b_6.setText("")
        self.before_b_6.setScaledContents(True)
        self.before_b_6.setObjectName("before_b_6")
        self.label = QtWidgets.QLabel(self.otsu)
        self.label.setGeometry(QtCore.QRect(240, 0, 101, 31))
        font = QtGui.QFont()
        font.setPointSize(12)
        font.setBold(True)
        font.setWeight(75)
        self.label.setFont(font)
        self.label.setObjectName("label")
        self.stackedWidget.addWidget(self.otsu)
        self.Mean_f = QtWidgets.QWidget()
        self.Mean_f.setObjectName("Mean_f")
        self.beforemeanfil = QtWidgets.QLabel(self.Mean_f)
        self.beforemeanfil.setGeometry(QtCore.QRect(20, 110, 261, 261))
        self.beforemeanfil.setStyleSheet("background-color: rgb(0, 0, 0);")
        self.beforemeanfil.setText("")
        self.beforemeanfil.setScaledContents(True)
        self.beforemeanfil.setObjectName("beforemeanfil")
        self.aftermean_f = QtWidgets.QLabel(self.Mean_f)
        self.aftermean_f.setGeometry(QtCore.QRect(320, 110, 251, 261))
        self.aftermean_f.setStyleSheet("background-color: rgb(0, 0, 0);")
        self.aftermean_f.setText("")
        self.aftermean_f.setScaledContents(True)
        self.aftermean_f.setObjectName("aftermean_f")
        self.uplead_meanf = QtWidgets.QPushButton(self.Mean_f)
        self.uplead_meanf.setGeometry(QtCore.QRect(50, 390, 81, 31))
        font = QtGui.QFont()
        font.setBold(True)
        font.setWeight(75)
        self.uplead_meanf.setFont(font)
        self.uplead_meanf.setStyleSheet("background-color: rgb(0, 170, 255);")
        self.uplead_meanf.setObjectName("uplead_meanf")
        self.save_maenf = QtWidgets.QPushButton(self.Mean_f)
        self.save_maenf.setGeometry(QtCore.QRect(460, 390, 81, 31))
        font = QtGui.QFont()
        font.setBold(True)
        font.setWeight(75)
        self.save_maenf.setFont(font)
        self.save_maenf.setStyleSheet("background-color: rgb(170, 0, 255);")
        self.save_maenf.setObjectName("save_maenf")
        self.applymeanf = QtWidgets.QPushButton(self.Mean_f)
        self.applymeanf.setGeometry(QtCore.QRect(240, 390, 111, 31))
        font = QtGui.QFont()
        font.setBold(True)
        font.setWeight(75)
        self.applymeanf.setFont(font)
        self.applymeanf.setStyleSheet("background-color: rgb(85, 255, 255);")
        self.applymeanf.setObjectName("applymeanf")
        self.label_3 = QtWidgets.QLabel(self.Mean_f)
        self.label_3.setGeometry(QtCore.QRect(180, 70, 81, 20))
        self.label_3.setStyleSheet("color: rgb(0, 0, 0);")
        self.label_3.setScaledContents(True)
        self.label_3.setObjectName("label_3")
        self.k_mean = QtWidgets.QSpinBox(self.Mean_f)
        self.k_mean.setGeometry(QtCore.QRect(260, 70, 81, 22))
        self.k_mean.setStyleSheet("background-color: rgb(162, 162, 162);")
        self.k_mean.setMaximum(100)
        self.k_mean.setSingleStep(1)
        self.k_mean.setObjectName("k_mean")
        self.label_8 = QtWidgets.QLabel(self.Mean_f)
        self.label_8.setGeometry(QtCore.QRect(210, 0, 151, 31))
        font = QtGui.QFont()
        font.setPointSize(12)
        font.setBold(True)
        font.setWeight(75)
        self.label_8.setFont(font)
        self.label_8.setStyleSheet("color: rgb(0, 0, 0);")
        self.label_8.setScaledContents(True)
        self.label_8.setObjectName("label_8")
        self.label_181 = QtWidgets.QLabel(self.Mean_f)
        self.label_181.setGeometry(QtCore.QRect(40, 70, 101, 31))
        self.label_181.setStyleSheet("color: rgb(0, 0, 0);")
        self.label_181.setObjectName("label_181")
        self.label_182 = QtWidgets.QLabel(self.Mean_f)
        self.label_182.setGeometry(QtCore.QRect(380, 70, 101, 31))
        self.label_182.setStyleSheet("color: rgb(0, 0, 0);")
        self.label_182.setObjectName("label_182")
        self.label_261 = QtWidgets.QLabel(self.Mean_f)
        self.label_261.setGeometry(QtCore.QRect(180, 40, 71, 21))
        self.label_261.setScaledContents(True)
        self.label_261.setObjectName("label_261")
        self.mean_f_iteration = QtWidgets.QSpinBox(self.Mean_f)
        self.mean_f_iteration.setGeometry(QtCore.QRect(260, 40, 81, 22))
        self.mean_f_iteration.setStyleSheet("background-color: rgb(162, 162, 162);")
        self.mean_f_iteration.setObjectName("mean_f_iteration")
        self.stackedWidget.addWidget(self.Mean_f)
        self.gamma_F = QtWidgets.QWidget()
        self.gamma_F.setObjectName("gamma_F")
        self.upload_gamma = QtWidgets.QPushButton(self.gamma_F)
        self.upload_gamma.setGeometry(QtCore.QRect(60, 400, 81, 31))
        font = QtGui.QFont()
        font.setBold(True)
        font.setWeight(75)
        self.upload_gamma.setFont(font)
        self.upload_gamma.setStyleSheet("background-color: rgb(0, 170, 255);")
        self.upload_gamma.setObjectName("upload_gamma")
        self.apply_gamma = QtWidgets.QPushButton(self.gamma_F)
        self.apply_gamma.setGeometry(QtCore.QRect(250, 400, 111, 31))
        font = QtGui.QFont()
        font.setBold(True)
        font.setWeight(75)
        self.apply_gamma.setFont(font)
        self.apply_gamma.setStyleSheet("background-color: rgb(85, 255, 255);")
        self.apply_gamma.setObjectName("apply_gamma")
        self.gamma = QtWidgets.QSpinBox(self.gamma_F)
        self.gamma.setGeometry(QtCore.QRect(270, 50, 81, 22))
        self.gamma.setStyleSheet("background-color: rgb(162, 162, 162);")
        self.gamma.setObjectName("gamma")
        self.label_26 = QtWidgets.QLabel(self.gamma_F)
        self.label_26.setGeometry(QtCore.QRect(400, 80, 91, 31))
        self.label_26.setStyleSheet("color: rgb(0, 0, 0);")
        self.label_26.setObjectName("label_26")
        self.save_gamma = QtWidgets.QPushButton(self.gamma_F)
        self.save_gamma.setGeometry(QtCore.QRect(450, 400, 81, 31))
        font = QtGui.QFont()
        font.setBold(True)
        font.setWeight(75)
        self.save_gamma.setFont(font)
        self.save_gamma.setStyleSheet("background-color: rgb(170, 0, 255);")
        self.save_gamma.setObjectName("save_gamma")
        self.label_28 = QtWidgets.QLabel(self.gamma_F)
        self.label_28.setGeometry(QtCore.QRect(70, 80, 101, 31))
        self.label_28.setObjectName("label_28")
        self.after_gamma = QtWidgets.QLabel(self.gamma_F)
        self.after_gamma.setGeometry(QtCore.QRect(300, 120, 281, 261))
        self.after_gamma.setStyleSheet("background-color: rgb(0, 0, 0);")
        self.after_gamma.setText("")
        self.after_gamma.setScaledContents(True)
        self.after_gamma.setObjectName("after_gamma")
        self.before_gamma = QtWidgets.QLabel(self.gamma_F)
        self.before_gamma.setGeometry(QtCore.QRect(30, 120, 251, 261))
        self.before_gamma.setStyleSheet("background-color: rgb(0, 0, 0);")
        self.before_gamma.setText("")
        self.before_gamma.setScaledContents(True)
        self.before_gamma.setObjectName("before_gamma")
        self.label_30 = QtWidgets.QLabel(self.gamma_F)
        self.label_30.setGeometry(QtCore.QRect(180, 50, 81, 21))
        self.label_30.setObjectName("label_30")
        self.label_31 = QtWidgets.QLabel(self.gamma_F)
        self.label_31.setGeometry(QtCore.QRect(200, 10, 201, 31))
        font = QtGui.QFont()
        font.setPointSize(12)
        font.setBold(True)
        font.setWeight(75)
        self.label_31.setFont(font)
        self.label_31.setObjectName("label_31")
        self.stackedWidget.addWidget(self.gamma_F)
        self.Medain_f = QtWidgets.QWidget()
        self.Medain_f.setObjectName("Medain_f")
        self.before_medf = QtWidgets.QLabel(self.Medain_f)
        self.before_medf.setGeometry(QtCore.QRect(10, 110, 251, 261))
        self.before_medf.setStyleSheet("background-color: rgb(0, 0, 0);")
        self.before_medf.setText("")
        self.before_medf.setScaledContents(True)
        self.before_medf.setObjectName("before_medf")
        self.after_medf = QtWidgets.QLabel(self.Medain_f)
        self.after_medf.setGeometry(QtCore.QRect(280, 110, 281, 261))
        self.after_medf.setStyleSheet("background-color: rgb(0, 0, 0);")
        self.after_medf.setText("")
        self.after_medf.setScaledContents(True)
        self.after_medf.setObjectName("after_medf")
        self.upload_medf = QtWidgets.QPushButton(self.Medain_f)
        self.upload_medf.setGeometry(QtCore.QRect(40, 390, 81, 31))
        font = QtGui.QFont()
        font.setBold(True)
        font.setWeight(75)
        self.upload_medf.setFont(font)
        self.upload_medf.setStyleSheet("background-color: rgb(0, 170, 255);")
        self.upload_medf.setObjectName("upload_medf")
        self.save_medf = QtWidgets.QPushButton(self.Medain_f)
        self.save_medf.setGeometry(QtCore.QRect(430, 390, 81, 31))
        font = QtGui.QFont()
        font.setBold(True)
        font.setWeight(75)
        self.save_medf.setFont(font)
        self.save_medf.setStyleSheet("background-color: rgb(170, 0, 255);")
        self.save_medf.setObjectName("save_medf")
        self.apply_medfil = QtWidgets.QPushButton(self.Medain_f)
        self.apply_medfil.setGeometry(QtCore.QRect(230, 390, 111, 31))
        font = QtGui.QFont()
        font.setBold(True)
        font.setWeight(75)
        self.apply_medfil.setFont(font)
        self.apply_medfil.setStyleSheet("background-color: rgb(85, 255, 255);")
        self.apply_medfil.setObjectName("apply_medfil")
        self.label_19 = QtWidgets.QLabel(self.Medain_f)
        self.label_19.setGeometry(QtCore.QRect(160, 60, 81, 21))
        self.label_19.setObjectName("label_19")
        self.k_medfil = QtWidgets.QSpinBox(self.Medain_f)
        self.k_medfil.setGeometry(QtCore.QRect(251, 60, 81, 22))
        self.k_medfil.setStyleSheet("background-color: rgb(162, 162, 162);")
        self.k_medfil.setObjectName("k_medfil")
        self.label_20 = QtWidgets.QLabel(self.Medain_f)
        self.label_20.setGeometry(QtCore.QRect(50, 70, 101, 31))
        self.label_20.setObjectName("label_20")
        self.label_21 = QtWidgets.QLabel(self.Medain_f)
        self.label_21.setGeometry(QtCore.QRect(380, 70, 91, 31))
        self.label_21.setStyleSheet("color: rgb(0, 0, 0);")
        self.label_21.setObjectName("label_21")
        self.label_22 = QtWidgets.QLabel(self.Medain_f)
        self.label_22.setGeometry(QtCore.QRect(180, 0, 201, 31))
        font = QtGui.QFont()
        font.setPointSize(12)
        font.setBold(True)
        font.setWeight(75)
        self.label_22.setFont(font)
        self.label_22.setObjectName("label_22")
        self.label_262 = QtWidgets.QLabel(self.Medain_f)
        self.label_262.setGeometry(QtCore.QRect(150, 40, 91, 21))
        self.label_262.setObjectName("label_262")
        self.medain_f_iteration = QtWidgets.QSpinBox(self.Medain_f)
        self.medain_f_iteration.setGeometry(QtCore.QRect(251, 40, 81, 22))
        self.medain_f_iteration.setStyleSheet("background-color: rgb(162, 162, 162);")
        self.medain_f_iteration.setObjectName("medain_f_iteration")
        self.stackedWidget.addWidget(self.Medain_f)
        self.Roberts = QtWidgets.QWidget()
        self.Roberts.setObjectName("Roberts")
        self.label_101 = QtWidgets.QLabel(self.Roberts)
        self.label_101.setGeometry(QtCore.QRect(210, 0, 111, 31))
        font = QtGui.QFont()
        font.setPointSize(12)
        font.setBold(True)
        font.setWeight(75)
        self.label_101.setFont(font)
        self.label_101.setObjectName("label_101")
        self.after_robert = QtWidgets.QLabel(self.Roberts)
        self.after_robert.setGeometry(QtCore.QRect(290, 110, 281, 271))
        self.after_robert.setStyleSheet("background-color: rgb(0, 0, 0);")
        self.after_robert.setText("")
        self.after_robert.setScaledContents(True)
        self.after_robert.setObjectName("after_robert")
        self.save_robert = QtWidgets.QPushButton(self.Roberts)
        self.save_robert.setGeometry(QtCore.QRect(500, 400, 81, 31))
        font = QtGui.QFont()
        font.setBold(True)
        font.setWeight(75)
        self.save_robert.setFont(font)
        self.save_robert.setStyleSheet("background-color: rgb(170, 0, 255);")
        self.save_robert.setObjectName("save_robert")
        self.brfore_robert = QtWidgets.QLabel(self.Roberts)
        self.brfore_robert.setGeometry(QtCore.QRect(10, 110, 271, 271))
        self.brfore_robert.setStyleSheet("background-color: rgb(0, 0, 0);")
        self.brfore_robert.setText("")
        self.brfore_robert.setScaledContents(True)
        self.brfore_robert.setObjectName("brfore_robert")
        self.Upload_robert = QtWidgets.QPushButton(self.Roberts)
        self.Upload_robert.setGeometry(QtCore.QRect(40, 400, 75, 31))
        font = QtGui.QFont()
        font.setBold(True)
        font.setWeight(75)
        self.Upload_robert.setFont(font)
        self.Upload_robert.setStyleSheet("background-color: rgb(0, 170, 255);")
        self.Upload_robert.setObjectName("Upload_robert")
        self.label_104 = QtWidgets.QLabel(self.Roberts)
        self.label_104.setGeometry(QtCore.QRect(380, 70, 111, 31))
        self.label_104.setObjectName("label_104")
        self.dis_roberts = QtWidgets.QPushButton(self.Roberts)
        self.dis_roberts.setGeometry(QtCore.QRect(180, 400, 131, 31))
        font = QtGui.QFont()
        font.setBold(True)
        font.setWeight(75)
        self.dis_roberts.setFont(font)
        self.dis_roberts.setStyleSheet("background-color: rgb(85, 255, 255);")
        self.dis_roberts.setObjectName("dis_roberts")
        self.label_105 = QtWidgets.QLabel(self.Roberts)
        self.label_105.setGeometry(QtCore.QRect(40, 65, 121, 31))
        self.label_105.setObjectName("label_105")
        self.roberts_th = QtWidgets.QSpinBox(self.Roberts)
        self.roberts_th.setGeometry(QtCore.QRect(280, 50, 101, 21))
        self.roberts_th.setStyleSheet("background-color: rgb(162, 162, 162);")
        self.roberts_th.setMaximum(255)
        self.roberts_th.setObjectName("roberts_th")
        self.th_7 = QtWidgets.QLabel(self.Roberts)
        self.th_7.setGeometry(QtCore.QRect(180, 40, 91, 41))
        self.th_7.setObjectName("th_7")
        self.checkBox_robert = QtWidgets.QCheckBox(self.Roberts)
        self.checkBox_robert.setGeometry(QtCore.QRect(220, 80, 121, 18))
        self.checkBox_robert.setObjectName("checkBox_robert")
        self.hist_robert = QtWidgets.QPushButton(self.Roberts)
        self.hist_robert.setGeometry(QtCore.QRect(360, 400, 91, 31))
        font = QtGui.QFont()
        font.setBold(True)
        font.setWeight(75)
        self.hist_robert.setFont(font)
        self.hist_robert.setStyleSheet("background-color: rgb(85, 255, 127);")
        self.hist_robert.setObjectName("hist_robert")
        self.stackedWidget.addWidget(self.Roberts)
        self.Prewit = QtWidgets.QWidget()
        self.Prewit.setObjectName("Prewit")
        self.label_23 = QtWidgets.QLabel(self.Prewit)
        self.label_23.setGeometry(QtCore.QRect(60, 50, 111, 41))
        self.label_23.setObjectName("label_23")
        self.dis_prewitt = QtWidgets.QPushButton(self.Prewit)
        self.dis_prewitt.setGeometry(QtCore.QRect(170, 390, 121, 31))
        font = QtGui.QFont()
        font.setBold(True)
        font.setWeight(75)
        self.dis_prewitt.setFont(font)
        self.dis_prewitt.setStyleSheet("background-color: rgb(85, 255, 255);")
        self.dis_prewitt.setObjectName("dis_prewitt")
        self.save_prewiit = QtWidgets.QPushButton(self.Prewit)
        self.save_prewiit.setGeometry(QtCore.QRect(470, 390, 81, 31))
        font = QtGui.QFont()
        font.setBold(True)
        font.setWeight(75)
        self.save_prewiit.setFont(font)
        self.save_prewiit.setStyleSheet("background-color: rgb(170, 0, 255);")
        self.save_prewiit.setObjectName("save_prewiit")
        self.label_25 = QtWidgets.QLabel(self.Prewit)
        self.label_25.setGeometry(QtCore.QRect(170, 0, 141, 31))
        font = QtGui.QFont()
        font.setPointSize(12)
        font.setBold(True)
        font.setWeight(75)
        self.label_25.setFont(font)
        self.label_25.setObjectName("label_25")
        self.before_prewitt = QtWidgets.QLabel(self.Prewit)
        self.before_prewitt.setGeometry(QtCore.QRect(20, 100, 251, 271))
        self.before_prewitt.setStyleSheet("background-color: rgb(0, 0, 0);")
        self.before_prewitt.setText("")
        self.before_prewitt.setScaledContents(True)
        self.before_prewitt.setObjectName("before_prewitt")
        self.upload_prewiit = QtWidgets.QPushButton(self.Prewit)
        self.upload_prewiit.setGeometry(QtCore.QRect(20, 390, 81, 31))
        font = QtGui.QFont()
        font.setBold(True)
        font.setWeight(75)
        self.upload_prewiit.setFont(font)
        self.upload_prewiit.setStyleSheet("background-color: rgb(0, 170, 255);")
        self.upload_prewiit.setObjectName("upload_prewiit")
        self.label_27 = QtWidgets.QLabel(self.Prewit)
        self.label_27.setGeometry(QtCore.QRect(380, 50, 111, 41))
        self.label_27.setObjectName("label_27")
        self.after_prewiit = QtWidgets.QLabel(self.Prewit)
        self.after_prewiit.setGeometry(QtCore.QRect(290, 100, 271, 271))
        self.after_prewiit.setStyleSheet("background-color: rgb(0, 0, 0);")
        self.after_prewiit.setText("")
        self.after_prewiit.setScaledContents(True)
        self.after_prewiit.setObjectName("after_prewiit")
        self.prewit_th = QtWidgets.QSpinBox(self.Prewit)
        self.prewit_th.setGeometry(QtCore.QRect(270, 40, 91, 21))
        self.prewit_th.setStyleSheet("background-color: rgb(162, 162, 162);")
        self.prewit_th.setMaximum(255)
        self.prewit_th.setObjectName("prewit_th")
        self.th_8 = QtWidgets.QLabel(self.Prewit)
        self.th_8.setGeometry(QtCore.QRect(170, 40, 91, 21))
        self.th_8.setObjectName("th_8")
        self.checkBox_prewitt = QtWidgets.QCheckBox(self.Prewit)
        self.checkBox_prewitt.setGeometry(QtCore.QRect(210, 70, 121, 18))
        self.checkBox_prewitt.setObjectName("checkBox_prewitt")
        self.hist_prewiit = QtWidgets.QPushButton(self.Prewit)
        self.hist_prewiit.setGeometry(QtCore.QRect(340, 390, 91, 31))
        font = QtGui.QFont()
        font.setBold(True)
        font.setWeight(75)
        self.hist_prewiit.setFont(font)
        self.hist_prewiit.setStyleSheet("background-color: rgb(85, 255, 127);")
        self.hist_prewiit.setObjectName("hist_prewiit")
        self.stackedWidget.addWidget(self.Prewit)
        self.Sobel = QtWidgets.QWidget()
        self.Sobel.setObjectName("Sobel")
        self.label_29 = QtWidgets.QLabel(self.Sobel)
        self.label_29.setGeometry(QtCore.QRect(150, 0, 191, 31))
        font = QtGui.QFont()
        font.setPointSize(12)
        font.setBold(True)
        font.setWeight(75)
        self.label_29.setFont(font)
        self.label_29.setObjectName("label_29")
        self.after_sobel = QtWidgets.QLabel(self.Sobel)
        self.after_sobel.setGeometry(QtCore.QRect(300, 90, 271, 281))
        self.after_sobel.setStyleSheet("background-color: rgb(0, 0, 0);")
        self.after_sobel.setText("")
        self.after_sobel.setScaledContents(True)
        self.after_sobel.setObjectName("after_sobel")
        self.save_sobel = QtWidgets.QPushButton(self.Sobel)
        self.save_sobel.setGeometry(QtCore.QRect(430, 390, 81, 31))
        font = QtGui.QFont()
        font.setBold(True)
        font.setWeight(75)
        self.save_sobel.setFont(font)
        self.save_sobel.setStyleSheet("background-color: rgb(162, 162, 162);\n"
"background-color: rgb(170, 0, 255);")
        self.save_sobel.setObjectName("save_sobel")
        self.before_sobel = QtWidgets.QLabel(self.Sobel)
        self.before_sobel.setGeometry(QtCore.QRect(20, 90, 271, 281))
        self.before_sobel.setStyleSheet("background-color: rgb(0, 0, 0);")
        self.before_sobel.setText("")
        self.before_sobel.setScaledContents(True)
        self.before_sobel.setObjectName("before_sobel")
        self.upload_sobel = QtWidgets.QPushButton(self.Sobel)
        self.upload_sobel.setGeometry(QtCore.QRect(30, 390, 81, 31))
        font = QtGui.QFont()
        font.setBold(True)
        font.setWeight(75)
        self.upload_sobel.setFont(font)
        self.upload_sobel.setStyleSheet("background-color: rgb(162, 162, 162);\n"
"background-color: rgb(0, 170, 255);")
        self.upload_sobel.setObjectName("upload_sobel")
        self.label_32 = QtWidgets.QLabel(self.Sobel)
        self.label_32.setGeometry(QtCore.QRect(360, 50, 131, 31))
        self.label_32.setObjectName("label_32")
        self.display_sobel = QtWidgets.QPushButton(self.Sobel)
        self.display_sobel.setGeometry(QtCore.QRect(130, 390, 141, 31))
        font = QtGui.QFont()
        font.setBold(True)
        font.setWeight(75)
        self.display_sobel.setFont(font)
        self.display_sobel.setStyleSheet("background-color: rgb(85, 255, 255);")
        self.display_sobel.setObjectName("display_sobel")
        self.label_24 = QtWidgets.QLabel(self.Sobel)
        self.label_24.setGeometry(QtCore.QRect(40, 50, 101, 31))
        self.label_24.setObjectName("label_24")
        self.sobel_th = QtWidgets.QSpinBox(self.Sobel)
        self.sobel_th.setGeometry(QtCore.QRect(270, 40, 81, 21))
        self.sobel_th.setStyleSheet("background-color: rgb(162, 162, 162);")
        self.sobel_th.setMaximum(255)
        self.sobel_th.setObjectName("sobel_th")
        self.th_9 = QtWidgets.QLabel(self.Sobel)
        self.th_9.setGeometry(QtCore.QRect(170, 30, 91, 41))
        self.th_9.setObjectName("th_9")
        self.checkBox_sobel = QtWidgets.QCheckBox(self.Sobel)
        self.checkBox_sobel.setGeometry(QtCore.QRect(190, 60, 121, 18))
        self.checkBox_sobel.setObjectName("checkBox_sobel")
        self.hist_sobel = QtWidgets.QPushButton(self.Sobel)
        self.hist_sobel.setGeometry(QtCore.QRect(300, 390, 91, 31))
        font = QtGui.QFont()
        font.setBold(True)
        font.setWeight(75)
        self.hist_sobel.setFont(font)
        self.hist_sobel.setStyleSheet("background-color: rgb(85, 255, 127);")
        self.hist_sobel.setObjectName("hist_sobel")
        self.stackedWidget.addWidget(self.Sobel)
        self.Robinson = QtWidgets.QWidget()
        self.Robinson.setObjectName("Robinson")
        self.label_33 = QtWidgets.QLabel(self.Robinson)
        self.label_33.setGeometry(QtCore.QRect(330, 50, 111, 31))
        self.label_33.setObjectName("label_33")
        self.label_34 = QtWidgets.QLabel(self.Robinson)
        self.label_34.setGeometry(QtCore.QRect(190, 10, 151, 31))
        font = QtGui.QFont()
        font.setPointSize(12)
        font.setBold(True)
        font.setWeight(75)
        self.label_34.setFont(font)
        self.label_34.setObjectName("label_34")
        self.Save_robinson = QtWidgets.QPushButton(self.Robinson)
        self.Save_robinson.setGeometry(QtCore.QRect(480, 390, 81, 31))
        font = QtGui.QFont()
        font.setBold(True)
        font.setWeight(75)
        self.Save_robinson.setFont(font)
        self.Save_robinson.setStyleSheet("background-color: rgb(162, 162, 162);\n"
"background-color: rgb(170, 0, 255);")
        self.Save_robinson.setObjectName("Save_robinson")
        self.label_35 = QtWidgets.QLabel(self.Robinson)
        self.label_35.setGeometry(QtCore.QRect(20, 50, 111, 31))
        self.label_35.setObjectName("label_35")
        self.upload_robinson = QtWidgets.QPushButton(self.Robinson)
        self.upload_robinson.setGeometry(QtCore.QRect(30, 390, 81, 31))
        font = QtGui.QFont()
        font.setBold(True)
        font.setWeight(75)
        self.upload_robinson.setFont(font)
        self.upload_robinson.setStyleSheet("background-color: rgb(0, 170, 255);")
        self.upload_robinson.setObjectName("upload_robinson")
        self.disp_robinson = QtWidgets.QPushButton(self.Robinson)
        self.disp_robinson.setGeometry(QtCore.QRect(150, 390, 141, 31))
        font = QtGui.QFont()
        font.setBold(True)
        font.setWeight(75)
        self.disp_robinson.setFont(font)
        self.disp_robinson.setStyleSheet("background-color: rgb(162, 162, 162);\n"
"background-color: rgb(85, 255, 255);")
        self.disp_robinson.setObjectName("disp_robinson")
        self.after_robinson = QtWidgets.QLabel(self.Robinson)
        self.after_robinson.setGeometry(QtCore.QRect(300, 100, 271, 271))
        self.after_robinson.setStyleSheet("background-color: rgb(0, 0, 0);")
        self.after_robinson.setText("")
        self.after_robinson.setScaledContents(True)
        self.after_robinson.setObjectName("after_robinson")
        self.before_robinson = QtWidgets.QLabel(self.Robinson)
        self.before_robinson.setGeometry(QtCore.QRect(10, 100, 271, 271))
        self.before_robinson.setStyleSheet("background-color: rgb(0, 0, 0);")
        self.before_robinson.setText("")
        self.before_robinson.setScaledContents(True)
        self.before_robinson.setObjectName("before_robinson")
        self.robinson_th = QtWidgets.QSpinBox(self.Robinson)
        self.robinson_th.setGeometry(QtCore.QRect(250, 50, 71, 21))
        self.robinson_th.setStyleSheet("background-color: rgb(162, 162, 162);")
        self.robinson_th.setMaximum(255)
        self.robinson_th.setObjectName("robinson_th")
        self.th_10 = QtWidgets.QLabel(self.Robinson)
        self.th_10.setGeometry(QtCore.QRect(140, 40, 101, 41))
        self.th_10.setObjectName("th_10")
        self.checkBox_robinson = QtWidgets.QCheckBox(self.Robinson)
        self.checkBox_robinson.setGeometry(QtCore.QRect(180, 70, 121, 18))
        self.checkBox_robinson.setObjectName("checkBox_robinson")
        self.hist_robinson = QtWidgets.QPushButton(self.Robinson)
        self.hist_robinson.setGeometry(QtCore.QRect(330, 390, 91, 31))
        font = QtGui.QFont()
        font.setBold(True)
        font.setWeight(75)
        self.hist_robinson.setFont(font)
        self.hist_robinson.setStyleSheet("background-color: rgb(85, 255, 127);")
        self.hist_robinson.setObjectName("hist_robinson")
        self.stackedWidget.addWidget(self.Robinson)
        self.Laplacien = QtWidgets.QWidget()
        self.Laplacien.setObjectName("Laplacien")
        self.label_96 = QtWidgets.QLabel(self.Laplacien)
        self.label_96.setGeometry(QtCore.QRect(380, 50, 91, 41))
        self.label_96.setObjectName("label_96")
        self.label_97 = QtWidgets.QLabel(self.Laplacien)
        self.label_97.setGeometry(QtCore.QRect(180, 10, 191, 31))
        font = QtGui.QFont()
        font.setPointSize(12)
        font.setBold(True)
        font.setWeight(75)
        self.label_97.setFont(font)
        self.label_97.setObjectName("label_97")
        self.Save_laplacien = QtWidgets.QPushButton(self.Laplacien)
        self.Save_laplacien.setGeometry(QtCore.QRect(460, 390, 81, 31))
        font = QtGui.QFont()
        font.setBold(True)
        font.setWeight(75)
        self.Save_laplacien.setFont(font)
        self.Save_laplacien.setStyleSheet("background-color: rgb(162, 162, 162);\n"
"background-color: rgb(170, 0, 255);")
        self.Save_laplacien.setObjectName("Save_laplacien")
        self.label_98 = QtWidgets.QLabel(self.Laplacien)
        self.label_98.setGeometry(QtCore.QRect(60, 55, 101, 31))
        self.label_98.setObjectName("label_98")
        self.upload_laplacien = QtWidgets.QPushButton(self.Laplacien)
        self.upload_laplacien.setGeometry(QtCore.QRect(20, 390, 81, 31))
        font = QtGui.QFont()
        font.setBold(True)
        font.setWeight(75)
        self.upload_laplacien.setFont(font)
        self.upload_laplacien.setStyleSheet("background-color: rgb(0, 170, 255);")
        self.upload_laplacien.setObjectName("upload_laplacien")
        self.dis_laplacien = QtWidgets.QPushButton(self.Laplacien)
        self.dis_laplacien.setGeometry(QtCore.QRect(150, 390, 151, 31))
        font = QtGui.QFont()
        font.setBold(True)
        font.setWeight(75)
        self.dis_laplacien.setFont(font)
        self.dis_laplacien.setStyleSheet("background-color: rgb(162, 162, 162);\n"
"background-color: rgb(85, 255, 255);")
        self.dis_laplacien.setObjectName("dis_laplacien")
        self.after_laplacien = QtWidgets.QLabel(self.Laplacien)
        self.after_laplacien.setGeometry(QtCore.QRect(310, 100, 271, 271))
        self.after_laplacien.setStyleSheet("background-color: rgb(0, 0, 0);")
        self.after_laplacien.setText("")
        self.after_laplacien.setScaledContents(True)
        self.after_laplacien.setObjectName("after_laplacien")
        self.before_laplacien = QtWidgets.QLabel(self.Laplacien)
        self.before_laplacien.setGeometry(QtCore.QRect(20, 100, 281, 271))
        self.before_laplacien.setStyleSheet("background-color: rgb(0, 0, 0);")
        self.before_laplacien.setText("")
        self.before_laplacien.setScaledContents(True)
        self.before_laplacien.setObjectName("before_laplacien")
        self.laplacien_th = QtWidgets.QSpinBox(self.Laplacien)
        self.laplacien_th.setGeometry(QtCore.QRect(280, 50, 81, 21))
        self.laplacien_th.setStyleSheet("background-color: rgb(162, 162, 162);")
        self.laplacien_th.setMaximum(255)
        self.laplacien_th.setObjectName("laplacien_th")
        self.th_11 = QtWidgets.QLabel(self.Laplacien)
        self.th_11.setGeometry(QtCore.QRect(180, 50, 91, 20))
        self.th_11.setObjectName("th_11")
        self.checkBox_laplacien = QtWidgets.QCheckBox(self.Laplacien)
        self.checkBox_laplacien.setGeometry(QtCore.QRect(220, 70, 121, 18))
        self.checkBox_laplacien.setObjectName("checkBox_laplacien")
        self.hist_robert_2 = QtWidgets.QPushButton(self.Laplacien)
        self.hist_robert_2.setGeometry(QtCore.QRect(330, 390, 91, 31))
        font = QtGui.QFont()
        font.setBold(True)
        font.setWeight(75)
        self.hist_robert_2.setFont(font)
        self.hist_robert_2.setStyleSheet("background-color: rgb(85, 255, 127);")
        self.hist_robert_2.setObjectName("hist_robert_2")
        self.stackedWidget.addWidget(self.Laplacien)
        self.Erosion = QtWidgets.QWidget()
        self.Erosion.setObjectName("Erosion")
        self.k_erosion = QtWidgets.QSpinBox(self.Erosion)
        self.k_erosion.setGeometry(QtCore.QRect(250, 40, 111, 22))
        self.k_erosion.setStyleSheet("background-color: rgb(162, 162, 162);")
        self.k_erosion.setObjectName("k_erosion")
        self.label_106 = QtWidgets.QLabel(self.Erosion)
        self.label_106.setGeometry(QtCore.QRect(40, 55, 111, 31))
        self.label_106.setObjectName("label_106")
        self.apply_erosion = QtWidgets.QPushButton(self.Erosion)
        self.apply_erosion.setGeometry(QtCore.QRect(220, 390, 131, 31))
        font = QtGui.QFont()
        font.setBold(True)
        font.setWeight(75)
        self.apply_erosion.setFont(font)
        self.apply_erosion.setStyleSheet("background-color: rgb(85, 255, 255);")
        self.apply_erosion.setObjectName("apply_erosion")
        self.save_erosion = QtWidgets.QPushButton(self.Erosion)
        self.save_erosion.setGeometry(QtCore.QRect(460, 390, 81, 31))
        font = QtGui.QFont()
        font.setBold(True)
        font.setWeight(75)
        self.save_erosion.setFont(font)
        self.save_erosion.setStyleSheet("background-color: rgb(170, 0, 255);")
        self.save_erosion.setObjectName("save_erosion")
        self.label_107 = QtWidgets.QLabel(self.Erosion)
        self.label_107.setGeometry(QtCore.QRect(150, 40, 91, 21))
        self.label_107.setObjectName("label_107")
        self.label_108 = QtWidgets.QLabel(self.Erosion)
        self.label_108.setGeometry(QtCore.QRect(200, 10, 111, 21))
        font = QtGui.QFont()
        font.setPointSize(12)
        font.setBold(True)
        font.setWeight(75)
        self.label_108.setFont(font)
        self.label_108.setObjectName("label_108")
        self.before_erosion = QtWidgets.QLabel(self.Erosion)
        self.before_erosion.setGeometry(QtCore.QRect(20, 100, 261, 271))
        self.before_erosion.setStyleSheet("background-color: rgb(0, 0, 0);")
        self.before_erosion.setText("")
        self.before_erosion.setScaledContents(True)
        self.before_erosion.setObjectName("before_erosion")
        self.Upload_erosion = QtWidgets.QPushButton(self.Erosion)
        self.Upload_erosion.setGeometry(QtCore.QRect(30, 390, 91, 31))
        font = QtGui.QFont()
        font.setBold(True)
        font.setWeight(75)
        self.Upload_erosion.setFont(font)
        self.Upload_erosion.setStyleSheet("background-color: rgb(0, 170, 255);")
        self.Upload_erosion.setObjectName("Upload_erosion")
        self.label_110 = QtWidgets.QLabel(self.Erosion)
        self.label_110.setGeometry(QtCore.QRect(380, 50, 121, 41))
        self.label_110.setObjectName("label_110")
        self.after_erosion = QtWidgets.QLabel(self.Erosion)
        self.after_erosion.setGeometry(QtCore.QRect(290, 100, 281, 271))
        self.after_erosion.setStyleSheet("background-color: rgb(0, 0, 0);")
        self.after_erosion.setText("")
        self.after_erosion.setScaledContents(True)
        self.after_erosion.setObjectName("after_erosion")
        self.label_263 = QtWidgets.QLabel(self.Erosion)
        self.label_263.setGeometry(QtCore.QRect(156, 70, 71, 21))
        self.label_263.setObjectName("label_263")
        self.erosion_iteration = QtWidgets.QSpinBox(self.Erosion)
        self.erosion_iteration.setGeometry(QtCore.QRect(250, 70, 111, 22))
        self.erosion_iteration.setStyleSheet("background-color: rgb(162, 162, 162);")
        self.erosion_iteration.setObjectName("erosion_iteration")
        self.stackedWidget.addWidget(self.Erosion)
        self.dellation = QtWidgets.QWidget()
        self.dellation.setObjectName("dellation")
        self.k_delarion = QtWidgets.QSpinBox(self.dellation)
        self.k_delarion.setGeometry(QtCore.QRect(270, 40, 121, 22))
        self.k_delarion.setStyleSheet("background-color: rgb(162, 162, 162);")
        self.k_delarion.setObjectName("k_delarion")
        self.label_112 = QtWidgets.QLabel(self.dellation)
        self.label_112.setGeometry(QtCore.QRect(40, 55, 111, 31))
        self.label_112.setObjectName("label_112")
        self.apply_delation = QtWidgets.QPushButton(self.dellation)
        self.apply_delation.setGeometry(QtCore.QRect(240, 390, 131, 31))
        font = QtGui.QFont()
        font.setBold(True)
        font.setWeight(75)
        self.apply_delation.setFont(font)
        self.apply_delation.setStyleSheet("background-color: rgb(85, 255, 255);")
        self.apply_delation.setObjectName("apply_delation")
        self.save_delation = QtWidgets.QPushButton(self.dellation)
        self.save_delation.setGeometry(QtCore.QRect(450, 390, 91, 31))
        font = QtGui.QFont()
        font.setBold(True)
        font.setWeight(75)
        self.save_delation.setFont(font)
        self.save_delation.setStyleSheet("background-color: rgb(170, 0, 255);")
        self.save_delation.setObjectName("save_delation")
        self.label_113 = QtWidgets.QLabel(self.dellation)
        self.label_113.setGeometry(QtCore.QRect(180, 40, 91, 21))
        self.label_113.setObjectName("label_113")
        self.label_114 = QtWidgets.QLabel(self.dellation)
        self.label_114.setGeometry(QtCore.QRect(210, 5, 181, 31))
        font = QtGui.QFont()
        font.setPointSize(12)
        font.setBold(True)
        font.setWeight(75)
        self.label_114.setFont(font)
        self.label_114.setObjectName("label_114")
        self.before_delation = QtWidgets.QLabel(self.dellation)
        self.before_delation.setGeometry(QtCore.QRect(10, 100, 271, 281))
        self.before_delation.setStyleSheet("background-color: rgb(0, 0, 0);")
        self.before_delation.setText("")
        self.before_delation.setScaledContents(True)
        self.before_delation.setObjectName("before_delation")
        self.pushButton_60 = QtWidgets.QPushButton(self.dellation)
        self.pushButton_60.setGeometry(QtCore.QRect(30, 390, 81, 31))
        font = QtGui.QFont()
        font.setBold(True)
        font.setWeight(75)
        self.pushButton_60.setFont(font)
        self.pushButton_60.setStyleSheet("background-color: rgb(0, 170, 255);")
        self.pushButton_60.setObjectName("pushButton_60")
        self.label_116 = QtWidgets.QLabel(self.dellation)
        self.label_116.setGeometry(QtCore.QRect(400, 59, 101, 31))
        self.label_116.setObjectName("label_116")
        self.after_delation = QtWidgets.QLabel(self.dellation)
        self.after_delation.setGeometry(QtCore.QRect(290, 100, 291, 281))
        self.after_delation.setStyleSheet("background-color: rgb(0, 0, 0);")
        self.after_delation.setText("")
        self.after_delation.setScaledContents(True)
        self.after_delation.setObjectName("after_delation")
        self.label_264 = QtWidgets.QLabel(self.dellation)
        self.label_264.setGeometry(QtCore.QRect(180, 70, 81, 21))
        self.label_264.setObjectName("label_264")
        self.delatation_iteration = QtWidgets.QSpinBox(self.dellation)
        self.delatation_iteration.setGeometry(QtCore.QRect(270, 70, 121, 22))
        self.delatation_iteration.setStyleSheet("background-color: rgb(162, 162, 162);")
        self.delatation_iteration.setObjectName("delatation_iteration")
        self.stackedWidget.addWidget(self.dellation)
        self.opening = QtWidgets.QWidget()
        self.opening.setObjectName("opening")
        self.k_opning = QtWidgets.QSpinBox(self.opening)
        self.k_opning.setGeometry(QtCore.QRect(250, 40, 121, 22))
        self.k_opning.setStyleSheet("background-color: rgb(162, 162, 162);")
        self.k_opning.setObjectName("k_opning")
        self.label_118 = QtWidgets.QLabel(self.opening)
        self.label_118.setGeometry(QtCore.QRect(80, 80, 81, 16))
        self.label_118.setObjectName("label_118")
        self.apply_opening = QtWidgets.QPushButton(self.opening)
        self.apply_opening.setGeometry(QtCore.QRect(220, 390, 141, 31))
        font = QtGui.QFont()
        font.setBold(True)
        font.setWeight(75)
        self.apply_opening.setFont(font)
        self.apply_opening.setStyleSheet("background-color: rgb(85, 255, 255);")
        self.apply_opening.setObjectName("apply_opening")
        self.save_opning = QtWidgets.QPushButton(self.opening)
        self.save_opning.setGeometry(QtCore.QRect(450, 390, 81, 31))
        font = QtGui.QFont()
        font.setBold(True)
        font.setWeight(75)
        self.save_opning.setFont(font)
        self.save_opning.setStyleSheet("background-color: rgb(170, 0, 255);")
        self.save_opning.setObjectName("save_opning")
        self.label_119 = QtWidgets.QLabel(self.opening)
        self.label_119.setGeometry(QtCore.QRect(110, 40, 141, 21))
        self.label_119.setObjectName("label_119")
        self.label_120 = QtWidgets.QLabel(self.opening)
        self.label_120.setGeometry(QtCore.QRect(200, 0, 131, 31))
        font = QtGui.QFont()
        font.setPointSize(12)
        font.setBold(True)
        font.setWeight(75)
        self.label_120.setFont(font)
        self.label_120.setObjectName("label_120")
        self.before_opening = QtWidgets.QLabel(self.opening)
        self.before_opening.setGeometry(QtCore.QRect(20, 110, 281, 261))
        self.before_opening.setStyleSheet("background-color: rgb(0, 0, 0);")
        self.before_opening.setText("")
        self.before_opening.setScaledContents(True)
        self.before_opening.setObjectName("before_opening")
        self.upload_opening = QtWidgets.QPushButton(self.opening)
        self.upload_opening.setGeometry(QtCore.QRect(50, 390, 81, 31))
        font = QtGui.QFont()
        font.setBold(True)
        font.setWeight(75)
        self.upload_opening.setFont(font)
        self.upload_opening.setStyleSheet("background-color: rgb(0, 170, 255);")
        self.upload_opening.setObjectName("upload_opening")
        self.label_122 = QtWidgets.QLabel(self.opening)
        self.label_122.setGeometry(QtCore.QRect(420, 60, 101, 41))
        self.label_122.setObjectName("label_122")
        self.after_opening = QtWidgets.QLabel(self.opening)
        self.after_opening.setGeometry(QtCore.QRect(310, 110, 261, 261))
        self.after_opening.setStyleSheet("background-color: rgb(0, 0, 0);")
        self.after_opening.setText("")
        self.after_opening.setScaledContents(True)
        self.after_opening.setObjectName("after_opening")
        self.stackedWidget.addWidget(self.opening)
        self.closing = QtWidgets.QWidget()
        self.closing.setObjectName("closing")
        self.k_closing = QtWidgets.QSpinBox(self.closing)
        self.k_closing.setGeometry(QtCore.QRect(280, 50, 91, 22))
        self.k_closing.setStyleSheet("background-color: rgb(162, 162, 162);")
        self.k_closing.setObjectName("k_closing")
        self.label_124 = QtWidgets.QLabel(self.closing)
        self.label_124.setGeometry(QtCore.QRect(70, 55, 81, 31))
        self.label_124.setObjectName("label_124")
        self.apply_closing = QtWidgets.QPushButton(self.closing)
        self.apply_closing.setGeometry(QtCore.QRect(220, 390, 131, 31))
        font = QtGui.QFont()
        font.setBold(True)
        font.setWeight(75)
        self.apply_closing.setFont(font)
        self.apply_closing.setStyleSheet("background-color: rgb(85, 255, 255);")
        self.apply_closing.setObjectName("apply_closing")
        self.save_closing = QtWidgets.QPushButton(self.closing)
        self.save_closing.setGeometry(QtCore.QRect(440, 390, 71, 31))
        font = QtGui.QFont()
        font.setBold(True)
        font.setWeight(75)
        self.save_closing.setFont(font)
        self.save_closing.setStyleSheet("background-color: rgb(170, 0, 255);")
        self.save_closing.setObjectName("save_closing")
        self.label_125 = QtWidgets.QLabel(self.closing)
        self.label_125.setGeometry(QtCore.QRect(170, 50, 101, 31))
        self.label_125.setObjectName("label_125")
        self.label_126 = QtWidgets.QLabel(self.closing)
        self.label_126.setGeometry(QtCore.QRect(210, 0, 121, 41))
        font = QtGui.QFont()
        font.setPointSize(12)
        font.setBold(True)
        font.setWeight(75)
        self.label_126.setFont(font)
        self.label_126.setObjectName("label_126")
        self.before_closing = QtWidgets.QLabel(self.closing)
        self.before_closing.setGeometry(QtCore.QRect(20, 100, 261, 281))
        self.before_closing.setStyleSheet("background-color: rgb(0, 0, 0);")
        self.before_closing.setText("")
        self.before_closing.setScaledContents(True)
        self.before_closing.setObjectName("before_closing")
        self.Upload_closing = QtWidgets.QPushButton(self.closing)
        self.Upload_closing.setGeometry(QtCore.QRect(40, 390, 81, 31))
        font = QtGui.QFont()
        font.setBold(True)
        font.setWeight(75)
        self.Upload_closing.setFont(font)
        self.Upload_closing.setStyleSheet("background-color: rgb(0, 170, 255);")
        self.Upload_closing.setObjectName("Upload_closing")
        self.label_128 = QtWidgets.QLabel(self.closing)
        self.label_128.setGeometry(QtCore.QRect(390, 60, 71, 31))
        self.label_128.setObjectName("label_128")
        self.after_closing = QtWidgets.QLabel(self.closing)
        self.after_closing.setGeometry(QtCore.QRect(290, 100, 291, 281))
        self.after_closing.setStyleSheet("background-color: rgb(0, 0, 0);")
        self.after_closing.setText("")
        self.after_closing.setScaledContents(True)
        self.after_closing.setObjectName("after_closing")
        self.stackedWidget.addWidget(self.closing)
        self.maencontour = QtWidgets.QWidget()
        self.maencontour.setObjectName("maencontour")
        self.upload_meancountour = QtWidgets.QPushButton(self.maencontour)
        self.upload_meancountour.setGeometry(QtCore.QRect(50, 390, 81, 31))
        font = QtGui.QFont()
        font.setBold(True)
        font.setWeight(75)
        self.upload_meancountour.setFont(font)
        self.upload_meancountour.setStyleSheet("background-color: rgb(0, 170, 255);")
        self.upload_meancountour.setObjectName("upload_meancountour")
        self.label_130 = QtWidgets.QLabel(self.maencontour)
        self.label_130.setGeometry(QtCore.QRect(40, 70, 131, 20))
        self.label_130.setObjectName("label_130")
        self.apply_meancountor = QtWidgets.QPushButton(self.maencontour)
        self.apply_meancountor.setGeometry(QtCore.QRect(270, 390, 71, 31))
        font = QtGui.QFont()
        font.setBold(True)
        font.setWeight(75)
        self.apply_meancountor.setFont(font)
        self.apply_meancountor.setStyleSheet("background-color: rgb(85, 255, 255);")
        self.apply_meancountor.setObjectName("apply_meancountor")
        self.before_contor = QtWidgets.QLabel(self.maencontour)
        self.before_contor.setGeometry(QtCore.QRect(20, 100, 271, 281))
        self.before_contor.setStyleSheet("background-color: rgb(0, 0, 0);")
        self.before_contor.setText("")
        self.before_contor.setScaledContents(True)
        self.before_contor.setObjectName("before_contor")
        self.label_132 = QtWidgets.QLabel(self.maencontour)
        self.label_132.setGeometry(QtCore.QRect(180, 50, 71, 21))
        self.label_132.setObjectName("label_132")
        self.k_meancontour = QtWidgets.QSpinBox(self.maencontour)
        self.k_meancontour.setGeometry(QtCore.QRect(260, 50, 101, 22))
        self.k_meancontour.setStyleSheet("background-color: rgb(162, 162, 162);")
        self.k_meancontour.setObjectName("k_meancontour")
        self.save_meancountor = QtWidgets.QPushButton(self.maencontour)
        self.save_meancountor.setGeometry(QtCore.QRect(450, 390, 81, 31))
        font = QtGui.QFont()
        font.setBold(True)
        font.setWeight(75)
        self.save_meancountor.setFont(font)
        self.save_meancountor.setStyleSheet("background-color: rgb(170, 0, 255);")
        self.save_meancountor.setObjectName("save_meancountor")
        self.label_133 = QtWidgets.QLabel(self.maencontour)
        self.label_133.setGeometry(QtCore.QRect(190, 10, 231, 31))
        font = QtGui.QFont()
        font.setPointSize(12)
        font.setBold(True)
        font.setWeight(75)
        self.label_133.setFont(font)
        self.label_133.setObjectName("label_133")
        self.label_134 = QtWidgets.QLabel(self.maencontour)
        self.label_134.setGeometry(QtCore.QRect(360, 70, 121, 20))
        self.label_134.setObjectName("label_134")
        self.after_meancountor = QtWidgets.QLabel(self.maencontour)
        self.after_meancountor.setGeometry(QtCore.QRect(310, 100, 271, 281))
        self.after_meancountor.setStyleSheet("background-color: rgb(0, 0, 0);")
        self.after_meancountor.setText("")
        self.after_meancountor.setScaledContents(True)
        self.after_meancountor.setObjectName("after_meancountor")
        self.stackedWidget.addWidget(self.maencontour)
        self.extarnalcontour = QtWidgets.QWidget()
        self.extarnalcontour.setObjectName("extarnalcontour")
        self.Upload_ex_contors = QtWidgets.QPushButton(self.extarnalcontour)
        self.Upload_ex_contors.setGeometry(QtCore.QRect(50, 390, 81, 31))
        font = QtGui.QFont()
        font.setBold(True)
        font.setWeight(75)
        self.Upload_ex_contors.setFont(font)
        self.Upload_ex_contors.setStyleSheet("background-color: rgb(0, 170, 255);")
        self.Upload_ex_contors.setObjectName("Upload_ex_contors")
        self.label_136 = QtWidgets.QLabel(self.extarnalcontour)
        self.label_136.setGeometry(QtCore.QRect(50, 50, 81, 31))
        self.label_136.setObjectName("label_136")
        self.apply_ex_contour = QtWidgets.QPushButton(self.extarnalcontour)
        self.apply_ex_contour.setGeometry(QtCore.QRect(260, 390, 81, 31))
        font = QtGui.QFont()
        font.setBold(True)
        font.setWeight(75)
        self.apply_ex_contour.setFont(font)
        self.apply_ex_contour.setStyleSheet("background-color: rgb(85, 255, 255);")
        self.apply_ex_contour.setObjectName("apply_ex_contour")
        self.before_ex_contours = QtWidgets.QLabel(self.extarnalcontour)
        self.before_ex_contours.setGeometry(QtCore.QRect(20, 90, 251, 291))
        self.before_ex_contours.setStyleSheet("background-color: rgb(0, 0, 0);")
        self.before_ex_contours.setText("")
        self.before_ex_contours.setScaledContents(True)
        self.before_ex_contours.setObjectName("before_ex_contours")
        self.label_138 = QtWidgets.QLabel(self.extarnalcontour)
        self.label_138.setGeometry(QtCore.QRect(180, 50, 91, 21))
        self.label_138.setObjectName("label_138")
        self.k_exe_contours = QtWidgets.QSpinBox(self.extarnalcontour)
        self.k_exe_contours.setGeometry(QtCore.QRect(280, 50, 81, 22))
        self.k_exe_contours.setStyleSheet("background-color: rgb(162, 162, 162);")
        self.k_exe_contours.setObjectName("k_exe_contours")
        self.save_ex_countor = QtWidgets.QPushButton(self.extarnalcontour)
        self.save_ex_countor.setGeometry(QtCore.QRect(440, 390, 81, 31))
        font = QtGui.QFont()
        font.setBold(True)
        font.setWeight(75)
        self.save_ex_countor.setFont(font)
        self.save_ex_countor.setStyleSheet("background-color: rgb(170, 0, 255);")
        self.save_ex_countor.setObjectName("save_ex_countor")
        self.label_139 = QtWidgets.QLabel(self.extarnalcontour)
        self.label_139.setGeometry(QtCore.QRect(210, 10, 191, 31))
        font = QtGui.QFont()
        font.setPointSize(12)
        font.setBold(True)
        font.setWeight(75)
        self.label_139.setFont(font)
        self.label_139.setObjectName("label_139")
        self.label_140 = QtWidgets.QLabel(self.extarnalcontour)
        self.label_140.setGeometry(QtCore.QRect(400, 50, 91, 31))
        self.label_140.setObjectName("label_140")
        self.after_exer_contour = QtWidgets.QLabel(self.extarnalcontour)
        self.after_exer_contour.setGeometry(QtCore.QRect(280, 90, 291, 291))
        self.after_exer_contour.setStyleSheet("background-color: rgb(0, 0, 0);")
        self.after_exer_contour.setText("")
        self.after_exer_contour.setScaledContents(True)
        self.after_exer_contour.setObjectName("after_exer_contour")
        self.stackedWidget.addWidget(self.extarnalcontour)
        self.inner = QtWidgets.QWidget()
        self.inner.setObjectName("inner")
        self.Upload_inner = QtWidgets.QPushButton(self.inner)
        self.Upload_inner.setGeometry(QtCore.QRect(30, 390, 71, 31))
        font = QtGui.QFont()
        font.setBold(True)
        font.setWeight(75)
        self.Upload_inner.setFont(font)
        self.Upload_inner.setStyleSheet("background-color: rgb(0, 170, 255);")
        self.Upload_inner.setObjectName("Upload_inner")
        self.label_142 = QtWidgets.QLabel(self.inner)
        self.label_142.setGeometry(QtCore.QRect(40, 80, 131, 20))
        self.label_142.setObjectName("label_142")
        self.apply_inner = QtWidgets.QPushButton(self.inner)
        self.apply_inner.setGeometry(QtCore.QRect(260, 390, 81, 31))
        font = QtGui.QFont()
        font.setBold(True)
        font.setWeight(75)
        self.apply_inner.setFont(font)
        self.apply_inner.setStyleSheet("background-color: rgb(85, 255, 255);")
        self.apply_inner.setObjectName("apply_inner")
        self.beffore_inner = QtWidgets.QLabel(self.inner)
        self.beffore_inner.setGeometry(QtCore.QRect(20, 110, 271, 271))
        self.beffore_inner.setStyleSheet("background-color: rgb(0, 0, 0);")
        self.beffore_inner.setText("")
        self.beffore_inner.setScaledContents(True)
        self.beffore_inner.setObjectName("beffore_inner")
        self.label_144 = QtWidgets.QLabel(self.inner)
        self.label_144.setGeometry(QtCore.QRect(180, 60, 81, 21))
        self.label_144.setObjectName("label_144")
        self.k_inner = QtWidgets.QSpinBox(self.inner)
        self.k_inner.setGeometry(QtCore.QRect(260, 60, 81, 22))
        self.k_inner.setStyleSheet("background-color: rgb(162, 162, 162);")
        self.k_inner.setObjectName("k_inner")
        self.save_inner = QtWidgets.QPushButton(self.inner)
        self.save_inner.setGeometry(QtCore.QRect(480, 390, 81, 31))
        font = QtGui.QFont()
        font.setBold(True)
        font.setWeight(75)
        self.save_inner.setFont(font)
        self.save_inner.setStyleSheet("background-color: rgb(170, 0, 255);")
        self.save_inner.setObjectName("save_inner")
        self.label_145 = QtWidgets.QLabel(self.inner)
        self.label_145.setGeometry(QtCore.QRect(200, 10, 171, 31))
        font = QtGui.QFont()
        font.setPointSize(12)
        font.setBold(True)
        font.setWeight(75)
        self.label_145.setFont(font)
        self.label_145.setObjectName("label_145")
        self.label_146 = QtWidgets.QLabel(self.inner)
        self.label_146.setGeometry(QtCore.QRect(346, 79, 141, 21))
        self.label_146.setObjectName("label_146")
        self.after_inner = QtWidgets.QLabel(self.inner)
        self.after_inner.setGeometry(QtCore.QRect(300, 110, 281, 271))
        self.after_inner.setStyleSheet("background-color: rgb(0, 0, 0);")
        self.after_inner.setText("")
        self.after_inner.setScaledContents(True)
        self.after_inner.setObjectName("after_inner")
        self.stackedWidget.addWidget(self.inner)
        self.region_growing = QtWidgets.QWidget()
        self.region_growing.setObjectName("region_growing")
        self.save_region = QtWidgets.QPushButton(self.region_growing)
        self.save_region.setGeometry(QtCore.QRect(440, 390, 81, 31))
        font = QtGui.QFont()
        font.setBold(True)
        font.setWeight(75)
        self.save_region.setFont(font)
        self.save_region.setStyleSheet("background-color: rgb(170, 0, 255);")
        self.save_region.setObjectName("save_region")
        self.label_184 = QtWidgets.QLabel(self.region_growing)
        self.label_184.setGeometry(QtCore.QRect(80, 65, 81, 31))
        self.label_184.setObjectName("label_184")
        self.Upload_region = QtWidgets.QPushButton(self.region_growing)
        self.Upload_region.setGeometry(QtCore.QRect(30, 390, 71, 31))
        font = QtGui.QFont()
        font.setBold(True)
        font.setWeight(75)
        self.Upload_region.setFont(font)
        self.Upload_region.setStyleSheet("background-color: rgb(0, 170, 255);")
        self.Upload_region.setObjectName("Upload_region")
        self.beffore_region = QtWidgets.QLabel(self.region_growing)
        self.beffore_region.setGeometry(QtCore.QRect(10, 110, 271, 271))
        self.beffore_region.setStyleSheet("background-color: rgb(0, 0, 0);")
        self.beffore_region.setText("")
        self.beffore_region.setScaledContents(True)
        self.beffore_region.setObjectName("beffore_region")
        self.after_region = QtWidgets.QLabel(self.region_growing)
        self.after_region.setGeometry(QtCore.QRect(290, 110, 291, 271))
        self.after_region.setStyleSheet("background-color: rgb(0, 0, 0);")
        self.after_region.setText("")
        self.after_region.setScaledContents(True)
        self.after_region.setObjectName("after_region")
        self.label_185 = QtWidgets.QLabel(self.region_growing)
        self.label_185.setGeometry(QtCore.QRect(400, 70, 121, 31))
        self.label_185.setObjectName("label_185")
        self.label_186 = QtWidgets.QLabel(self.region_growing)
        self.label_186.setGeometry(QtCore.QRect(210, 10, 191, 41))
        font = QtGui.QFont()
        font.setPointSize(12)
        font.setBold(True)
        font.setWeight(75)
        self.label_186.setFont(font)
        self.label_186.setObjectName("label_186")
        self.apply_region = QtWidgets.QPushButton(self.region_growing)
        self.apply_region.setGeometry(QtCore.QRect(230, 390, 81, 31))
        font = QtGui.QFont()
        font.setBold(True)
        font.setWeight(75)
        self.apply_region.setFont(font)
        self.apply_region.setStyleSheet("background-color: rgb(85, 255, 255);")
        self.apply_region.setObjectName("apply_region")
        self.stackedWidget.addWidget(self.region_growing)
        self.mean_shift = QtWidgets.QWidget()
        self.mean_shift.setObjectName("mean_shift")
        self.apply_meanshift = QtWidgets.QPushButton(self.mean_shift)
        self.apply_meanshift.setGeometry(QtCore.QRect(240, 390, 81, 31))
        font = QtGui.QFont()
        font.setBold(True)
        font.setWeight(75)
        self.apply_meanshift.setFont(font)
        self.apply_meanshift.setStyleSheet("background-color: rgb(85, 255, 255);")
        self.apply_meanshift.setObjectName("apply_meanshift")
        self.label_252 = QtWidgets.QLabel(self.mean_shift)
        self.label_252.setGeometry(QtCore.QRect(70, 45, 101, 31))
        self.label_252.setObjectName("label_252")
        self.after_meanshift = QtWidgets.QLabel(self.mean_shift)
        self.after_meanshift.setGeometry(QtCore.QRect(300, 90, 281, 291))
        self.after_meanshift.setStyleSheet("background-color: rgb(0, 0, 0);")
        self.after_meanshift.setText("")
        self.after_meanshift.setScaledContents(True)
        self.after_meanshift.setObjectName("after_meanshift")
        self.beffore_meanshift = QtWidgets.QLabel(self.mean_shift)
        self.beffore_meanshift.setGeometry(QtCore.QRect(10, 90, 281, 291))
        self.beffore_meanshift.setStyleSheet("background-color: rgb(0, 0, 0);")
        self.beffore_meanshift.setText("")
        self.beffore_meanshift.setScaledContents(True)
        self.beffore_meanshift.setObjectName("beffore_meanshift")
        self.label_253 = QtWidgets.QLabel(self.mean_shift)
        self.label_253.setGeometry(QtCore.QRect(150, 0, 221, 41))
        font = QtGui.QFont()
        font.setPointSize(12)
        font.setBold(True)
        font.setWeight(75)
        self.label_253.setFont(font)
        self.label_253.setObjectName("label_253")
        self.Upload_meanshift = QtWidgets.QPushButton(self.mean_shift)
        self.Upload_meanshift.setGeometry(QtCore.QRect(40, 390, 71, 31))
        font = QtGui.QFont()
        font.setBold(True)
        font.setWeight(75)
        self.Upload_meanshift.setFont(font)
        self.Upload_meanshift.setStyleSheet("background-color: rgb(0, 170, 255);")
        self.Upload_meanshift.setObjectName("Upload_meanshift")
        self.save_meanshift = QtWidgets.QPushButton(self.mean_shift)
        self.save_meanshift.setGeometry(QtCore.QRect(450, 390, 81, 31))
        font = QtGui.QFont()
        font.setBold(True)
        font.setWeight(75)
        self.save_meanshift.setFont(font)
        self.save_meanshift.setStyleSheet("background-color: rgb(170, 0, 255);")
        self.save_meanshift.setObjectName("save_meanshift")
        self.label_254 = QtWidgets.QLabel(self.mean_shift)
        self.label_254.setGeometry(QtCore.QRect(380, 50, 121, 31))
        self.label_254.setObjectName("label_254")
        self.stackedWidget.addWidget(self.mean_shift)
        self.GrabCut = QtWidgets.QWidget()
        self.GrabCut.setObjectName("GrabCut")
        self.apply_GrabCut = QtWidgets.QPushButton(self.GrabCut)
        self.apply_GrabCut.setGeometry(QtCore.QRect(230, 390, 81, 31))
        font = QtGui.QFont()
        font.setBold(True)
        font.setWeight(75)
        self.apply_GrabCut.setFont(font)
        self.apply_GrabCut.setStyleSheet("background-color: rgb(85, 255, 255);")
        self.apply_GrabCut.setObjectName("apply_GrabCut")
        self.label_255 = QtWidgets.QLabel(self.GrabCut)
        self.label_255.setGeometry(QtCore.QRect(80, 60, 91, 31))
        self.label_255.setObjectName("label_255")
        self.after_GrabCut = QtWidgets.QLabel(self.GrabCut)
        self.after_GrabCut.setGeometry(QtCore.QRect(300, 100, 281, 281))
        self.after_GrabCut.setStyleSheet("background-color: rgb(0, 0, 0);")
        self.after_GrabCut.setText("")
        self.after_GrabCut.setScaledContents(True)
        self.after_GrabCut.setObjectName("after_GrabCut")
        self.beffore_GrabCut = QtWidgets.QLabel(self.GrabCut)
        self.beffore_GrabCut.setGeometry(QtCore.QRect(10, 100, 281, 281))
        self.beffore_GrabCut.setStyleSheet("background-color: rgb(0, 0, 0);")
        self.beffore_GrabCut.setText("")
        self.beffore_GrabCut.setScaledContents(True)
        self.beffore_GrabCut.setObjectName("beffore_GrabCut")
        self.label_256 = QtWidgets.QLabel(self.GrabCut)
        self.label_256.setGeometry(QtCore.QRect(200, 10, 151, 41))
        font = QtGui.QFont()
        font.setPointSize(12)
        font.setBold(True)
        font.setWeight(75)
        self.label_256.setFont(font)
        self.label_256.setObjectName("label_256")
        self.Upload_GrabCut = QtWidgets.QPushButton(self.GrabCut)
        self.Upload_GrabCut.setGeometry(QtCore.QRect(20, 390, 71, 31))
        font = QtGui.QFont()
        font.setBold(True)
        font.setWeight(75)
        self.Upload_GrabCut.setFont(font)
        self.Upload_GrabCut.setStyleSheet("background-color: rgb(0, 170, 255);")
        self.Upload_GrabCut.setObjectName("Upload_GrabCut")
        self.save_GrabCut = QtWidgets.QPushButton(self.GrabCut)
        self.save_GrabCut.setGeometry(QtCore.QRect(460, 390, 81, 31))
        font = QtGui.QFont()
        font.setBold(True)
        font.setWeight(75)
        self.save_GrabCut.setFont(font)
        self.save_GrabCut.setStyleSheet("background-color: rgb(170, 0, 255);")
        self.save_GrabCut.setObjectName("save_GrabCut")
        self.label_257 = QtWidgets.QLabel(self.GrabCut)
        self.label_257.setGeometry(QtCore.QRect(390, 60, 91, 31))
        self.label_257.setObjectName("label_257")
        self.stackedWidget.addWidget(self.GrabCut)
        self.Watershed = QtWidgets.QWidget()
        self.Watershed.setObjectName("Watershed")
        self.apply_Watershed = QtWidgets.QPushButton(self.Watershed)
        self.apply_Watershed.setGeometry(QtCore.QRect(240, 390, 81, 31))
        font = QtGui.QFont()
        font.setBold(True)
        font.setWeight(75)
        self.apply_Watershed.setFont(font)
        self.apply_Watershed.setStyleSheet("background-color: rgb(85, 255, 255);")
        self.apply_Watershed.setObjectName("apply_Watershed")
        self.label_258 = QtWidgets.QLabel(self.Watershed)
        self.label_258.setGeometry(QtCore.QRect(90, 45, 81, 31))
        self.label_258.setObjectName("label_258")
        self.after_Watershed = QtWidgets.QLabel(self.Watershed)
        self.after_Watershed.setGeometry(QtCore.QRect(300, 90, 281, 291))
        self.after_Watershed.setStyleSheet("background-color: rgb(0, 0, 0);")
        self.after_Watershed.setText("")
        self.after_Watershed.setScaledContents(True)
        self.after_Watershed.setObjectName("after_Watershed")
        self.beffore_Watershed = QtWidgets.QLabel(self.Watershed)
        self.beffore_Watershed.setGeometry(QtCore.QRect(20, 90, 271, 291))
        self.beffore_Watershed.setStyleSheet("background-color: rgb(0, 0, 0);")
        self.beffore_Watershed.setText("")
        self.beffore_Watershed.setScaledContents(True)
        self.beffore_Watershed.setObjectName("beffore_Watershed")
        self.label_259 = QtWidgets.QLabel(self.Watershed)
        self.label_259.setGeometry(QtCore.QRect(210, 10, 131, 41))
        font = QtGui.QFont()
        font.setPointSize(12)
        font.setBold(True)
        font.setWeight(75)
        self.label_259.setFont(font)
        self.label_259.setObjectName("label_259")
        self.Upload_Watershed = QtWidgets.QPushButton(self.Watershed)
        self.Upload_Watershed.setGeometry(QtCore.QRect(30, 390, 71, 31))
        font = QtGui.QFont()
        font.setBold(True)
        font.setWeight(75)
        self.Upload_Watershed.setFont(font)
        self.Upload_Watershed.setStyleSheet("background-color: rgb(0, 170, 255);")
        self.Upload_Watershed.setObjectName("Upload_Watershed")
        self.save_Watershed = QtWidgets.QPushButton(self.Watershed)
        self.save_Watershed.setGeometry(QtCore.QRect(440, 390, 81, 31))
        font = QtGui.QFont()
        font.setBold(True)
        font.setWeight(75)
        self.save_Watershed.setFont(font)
        self.save_Watershed.setStyleSheet("background-color: rgb(170, 0, 255);")
        self.save_Watershed.setObjectName("save_Watershed")
        self.label_260 = QtWidgets.QLabel(self.Watershed)
        self.label_260.setGeometry(QtCore.QRect(370, 50, 91, 31))
        self.label_260.setObjectName("label_260")
        self.stackedWidget.addWidget(self.Watershed)
        self.hough_lines = QtWidgets.QWidget()
        self.hough_lines.setObjectName("hough_lines")
        self.uplead_houglenes = QtWidgets.QPushButton(self.hough_lines)
        self.uplead_houglenes.setGeometry(QtCore.QRect(30, 390, 81, 31))
        font = QtGui.QFont()
        font.setBold(True)
        font.setWeight(75)
        self.uplead_houglenes.setFont(font)
        self.uplead_houglenes.setStyleSheet("background-color: rgb(0, 170, 255);")
        self.uplead_houglenes.setObjectName("uplead_houglenes")
        self.label_265 = QtWidgets.QLabel(self.hough_lines)
        self.label_265.setGeometry(QtCore.QRect(200, 50, 81, 20))
        self.label_265.setStyleSheet("color: rgb(0, 0, 0);")
        self.label_265.setObjectName("label_265")
        self.label_266 = QtWidgets.QLabel(self.hough_lines)
        self.label_266.setGeometry(QtCore.QRect(400, 49, 91, 31))
        self.label_266.setStyleSheet("color: rgb(0, 0, 0);")
        self.label_266.setObjectName("label_266")
        self.label_267 = QtWidgets.QLabel(self.hough_lines)
        self.label_267.setGeometry(QtCore.QRect(170, 10, 211, 31))
        font = QtGui.QFont()
        font.setPointSize(12)
        font.setBold(True)
        font.setWeight(75)
        self.label_267.setFont(font)
        self.label_267.setStyleSheet("color: rgb(0, 0, 0);")
        self.label_267.setObjectName("label_267")
        self.after_hough_lines = QtWidgets.QLabel(self.hough_lines)
        self.after_hough_lines.setGeometry(QtCore.QRect(290, 90, 291, 291))
        self.after_hough_lines.setStyleSheet("background-color: rgb(0, 0, 0);")
        self.after_hough_lines.setText("")
        self.after_hough_lines.setScaledContents(True)
        self.after_hough_lines.setObjectName("after_hough_lines")
        self.applyhoughlines = QtWidgets.QPushButton(self.hough_lines)
        self.applyhoughlines.setGeometry(QtCore.QRect(230, 390, 81, 31))
        font = QtGui.QFont()
        font.setBold(True)
        font.setWeight(75)
        self.applyhoughlines.setFont(font)
        self.applyhoughlines.setStyleSheet("background-color: rgb(85, 255, 255);")
        self.applyhoughlines.setObjectName("applyhoughlines")
        self.label_268 = QtWidgets.QLabel(self.hough_lines)
        self.label_268.setGeometry(QtCore.QRect(60, 55, 101, 21))
        self.label_268.setStyleSheet("color: rgb(0, 0, 0);")
        self.label_268.setObjectName("label_268")
        self.th_houglines = QtWidgets.QSpinBox(self.hough_lines)
        self.th_houglines.setGeometry(QtCore.QRect(290, 50, 91, 22))
        self.th_houglines.setStyleSheet("background-color: rgb(162, 162, 162);")
        self.th_houglines.setMaximum(30000)
        self.th_houglines.setSingleStep(1)
        self.th_houglines.setObjectName("th_houglines")
        self.beforehoughlines = QtWidgets.QLabel(self.hough_lines)
        self.beforehoughlines.setGeometry(QtCore.QRect(10, 90, 271, 291))
        self.beforehoughlines.setStyleSheet("background-color: rgb(0, 0, 0);")
        self.beforehoughlines.setText("")
        self.beforehoughlines.setScaledContents(True)
        self.beforehoughlines.setObjectName("beforehoughlines")
        self.save_houghlines = QtWidgets.QPushButton(self.hough_lines)
        self.save_houghlines.setGeometry(QtCore.QRect(460, 390, 81, 31))
        font = QtGui.QFont()
        font.setBold(True)
        font.setWeight(75)
        self.save_houghlines.setFont(font)
        self.save_houghlines.setStyleSheet("background-color: rgb(170, 0, 255);")
        self.save_houghlines.setObjectName("save_houghlines")
        self.stackedWidget.addWidget(self.hough_lines)
        self.hough_circeles = QtWidgets.QWidget()
        self.hough_circeles.setObjectName("hough_circeles")
        self.applyhoughlcircles = QtWidgets.QPushButton(self.hough_circeles)
        self.applyhoughlcircles.setGeometry(QtCore.QRect(230, 390, 101, 31))
        font = QtGui.QFont()
        font.setBold(True)
        font.setWeight(75)
        self.applyhoughlcircles.setFont(font)
        self.applyhoughlcircles.setStyleSheet("background-color: rgb(85, 255, 255);")
        self.applyhoughlcircles.setObjectName("applyhoughlcircles")
        self.label_269 = QtWidgets.QLabel(self.hough_circeles)
        self.label_269.setGeometry(QtCore.QRect(80, 60, 81, 16))
        self.label_269.setStyleSheet("color: rgb(0, 0, 0);")
        self.label_269.setObjectName("label_269")
        self.beforehoughcircles = QtWidgets.QLabel(self.hough_circeles)
        self.beforehoughcircles.setGeometry(QtCore.QRect(10, 90, 271, 291))
        self.beforehoughcircles.setStyleSheet("background-color: rgb(0, 0, 0);")
        self.beforehoughcircles.setText("")
        self.beforehoughcircles.setScaledContents(True)
        self.beforehoughcircles.setObjectName("beforehoughcircles")
        self.uplead_hougcircles = QtWidgets.QPushButton(self.hough_circeles)
        self.uplead_hougcircles.setGeometry(QtCore.QRect(30, 390, 81, 31))
        font = QtGui.QFont()
        font.setBold(True)
        font.setWeight(75)
        self.uplead_hougcircles.setFont(font)
        self.uplead_hougcircles.setStyleSheet("background-color: rgb(0, 170, 255);")
        self.uplead_hougcircles.setObjectName("uplead_hougcircles")
        self.save_houghcircles = QtWidgets.QPushButton(self.hough_circeles)
        self.save_houghcircles.setGeometry(QtCore.QRect(450, 390, 81, 31))
        font = QtGui.QFont()
        font.setBold(True)
        font.setWeight(75)
        self.save_houghcircles.setFont(font)
        self.save_houghcircles.setStyleSheet("background-color: rgb(170, 0, 255);")
        self.save_houghcircles.setObjectName("save_houghcircles")
        self.label_271 = QtWidgets.QLabel(self.hough_circeles)
        self.label_271.setGeometry(QtCore.QRect(410, 49, 81, 31))
        self.label_271.setStyleSheet("color: rgb(0, 0, 0);")
        self.label_271.setObjectName("label_271")
        self.after_hough_circles = QtWidgets.QLabel(self.hough_circeles)
        self.after_hough_circles.setGeometry(QtCore.QRect(290, 90, 291, 291))
        self.after_hough_circles.setStyleSheet("background-color: rgb(0, 0, 0);")
        self.after_hough_circles.setText("")
        self.after_hough_circles.setScaledContents(True)
        self.after_hough_circles.setObjectName("after_hough_circles")
        self.label_272 = QtWidgets.QLabel(self.hough_circeles)
        self.label_272.setGeometry(QtCore.QRect(210, 10, 181, 31))
        font = QtGui.QFont()
        font.setPointSize(12)
        font.setBold(True)
        font.setWeight(75)
        self.label_272.setFont(font)
        self.label_272.setStyleSheet("color: rgb(0, 0, 0);")
        self.label_272.setObjectName("label_272")
        self.stackedWidget.addWidget(self.hough_circeles)
        self.verticalLayout.addWidget(self.stackedWidget)
        ImageProcessingAPP.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(ImageProcessingAPP)
        self.menubar.setEnabled(True)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 609, 22))
        self.menubar.setMouseTracking(True)
        self.menubar.setFocusPolicy(QtCore.Qt.NoFocus)
        self.menubar.setStatusTip("")
        self.menubar.setObjectName("menubar")
        self.menufile = QtWidgets.QMenu(self.menubar)
        self.menufile.setToolTipsVisible(False)
        self.menufile.setObjectName("menufile")
        self.menuimage = QtWidgets.QMenu(self.menubar)
        self.menuimage.setObjectName("menuimage")
        self.menuThresholding = QtWidgets.QMenu(self.menuimage)
        self.menuThresholding.setObjectName("menuThresholding")
        self.menuHough = QtWidgets.QMenu(self.menuimage)
        self.menuHough.setObjectName("menuHough")
        self.menufiltrage = QtWidgets.QMenu(self.menubar)
        self.menufiltrage.setObjectName("menufiltrage")
        self.menucontour_extraction = QtWidgets.QMenu(self.menubar)
        self.menucontour_extraction.setObjectName("menucontour_extraction")
        self.menumorphology = QtWidgets.QMenu(self.menubar)
        self.menumorphology.setObjectName("menumorphology")
        self.menusegmentation = QtWidgets.QMenu(self.menubar)
        self.menusegmentation.setObjectName("menusegmentation")
        self.menuHelp = QtWidgets.QMenu(self.menubar)
        self.menuHelp.setObjectName("menuHelp")
        ImageProcessingAPP.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(ImageProcessingAPP)
        self.statusbar.setObjectName("statusbar")
        ImageProcessingAPP.setStatusBar(self.statusbar)
        self.actionOpen = QtWidgets.QAction(ImageProcessingAPP)
        self.actionOpen.setObjectName("actionOpen")
        self.actionsave = QtWidgets.QAction(ImageProcessingAPP)
        self.actionsave.setObjectName("actionsave")
        self.actionsave_as = QtWidgets.QAction(ImageProcessingAPP)
        self.actionsave_as.setObjectName("actionsave_as")
        self.actionGrayscale_Conversion = QtWidgets.QAction(ImageProcessingAPP)
        self.actionGrayscale_Conversion.setCheckable(False)
        font = QtGui.QFont()
        self.actionGrayscale_Conversion.setFont(font)
        self.actionGrayscale_Conversion.setObjectName("actionGrayscale_Conversion")
        self.actionCropping = QtWidgets.QAction(ImageProcessingAPP)
        self.actionCropping.setObjectName("actionCropping")
        self.actionResizing = QtWidgets.QAction(ImageProcessingAPP)
        self.actionResizing.setCheckable(False)
        self.actionResizing.setEnabled(True)
        self.actionResizing.setObjectName("actionResizing")
        self.actionHistogram = QtWidgets.QAction(ImageProcessingAPP)
        self.actionHistogram.setObjectName("actionHistogram")
        self.actionRotation_and_Flipping = QtWidgets.QAction(ImageProcessingAPP)
        self.actionRotation_and_Flipping.setObjectName("actionRotation_and_Flipping")
        self.actionMean_filtre = QtWidgets.QAction(ImageProcessingAPP)
        self.actionMean_filtre.setObjectName("actionMean_filtre")
        self.actionMedain_filtre = QtWidgets.QAction(ImageProcessingAPP)
        self.actionMedain_filtre.setObjectName("actionMedain_filtre")
        self.actionErosion = QtWidgets.QAction(ImageProcessingAPP)
        self.actionErosion.setObjectName("actionErosion")
        self.actionDelation = QtWidgets.QAction(ImageProcessingAPP)
        self.actionDelation.setObjectName("actionDelation")
        self.actionOpening = QtWidgets.QAction(ImageProcessingAPP)
        self.actionOpening.setObjectName("actionOpening")
        self.actionClosing = QtWidgets.QAction(ImageProcessingAPP)
        self.actionClosing.setObjectName("actionClosing")
        self.actionMean_Contour = QtWidgets.QAction(ImageProcessingAPP)
        self.actionMean_Contour.setObjectName("actionMean_Contour")
        self.actionexternal_contour = QtWidgets.QAction(ImageProcessingAPP)
        self.actionexternal_contour.setObjectName("actionexternal_contour")
        self.actionInner_contour = QtWidgets.QAction(ImageProcessingAPP)
        self.actionInner_contour.setObjectName("actionInner_contour")
        self.actionPrewit = QtWidgets.QAction(ImageProcessingAPP)
        self.actionPrewit.setObjectName("actionPrewit")
        self.actionSobel = QtWidgets.QAction(ImageProcessingAPP)
        self.actionSobel.setObjectName("actionSobel")
        self.actionRobinson = QtWidgets.QAction(ImageProcessingAPP)
        self.actionRobinson.setObjectName("actionRobinson")
        self.actionRoberts = QtWidgets.QAction(ImageProcessingAPP)
        self.actionRoberts.setObjectName("actionRoberts")
        self.actionLaplacien = QtWidgets.QAction(ImageProcessingAPP)
        self.actionLaplacien.setObjectName("actionLaplacien")
        self.actionOtsu = QtWidgets.QAction(ImageProcessingAPP)
        self.actionOtsu.setObjectName("actionOtsu")
        self.actionFix_Threshold = QtWidgets.QAction(ImageProcessingAPP)
        self.actionFix_Threshold.setObjectName("actionFix_Threshold")
        self.actionRegion_growing = QtWidgets.QAction(ImageProcessingAPP)
        font = QtGui.QFont()
        self.actionRegion_growing.setFont(font)
        self.actionRegion_growing.setObjectName("actionRegion_growing")
        self.actionWatershed = QtWidgets.QAction(ImageProcessingAPP)
        self.actionWatershed.setObjectName("actionWatershed")
        self.actionGrabCut = QtWidgets.QAction(ImageProcessingAPP)
        self.actionGrabCut.setObjectName("actionGrabCut")
        self.actionMean_Shift = QtWidgets.QAction(ImageProcessingAPP)
        self.actionMean_Shift.setObjectName("actionMean_Shift")
        self.actionLines = QtWidgets.QAction(ImageProcessingAPP)
        self.actionLines.setObjectName("actionLines")
        self.actioncircles = QtWidgets.QAction(ImageProcessingAPP)
        self.actioncircles.setObjectName("actioncircles")
        self.actionNew = QtWidgets.QAction(ImageProcessingAPP)
        self.actionNew.setObjectName("actionNew")
        self.actionSave_All = QtWidgets.QAction(ImageProcessingAPP)
        self.actionSave_All.setObjectName("actionSave_All")
        self.actionSave_Image = QtWidgets.QAction(ImageProcessingAPP)
        self.actionSave_Image.setObjectName("actionSave_Image")
        self.actionClose = QtWidgets.QAction(ImageProcessingAPP)
        self.actionClose.setObjectName("actionClose")
        self.actionQuit = QtWidgets.QAction(ImageProcessingAPP)
        self.actionQuit.setObjectName("actionQuit")
        self.actionContact_me = QtWidgets.QAction(ImageProcessingAPP)
        self.actionContact_me.setObjectName("actionContact_me")
        self.actionAbout_App = QtWidgets.QAction(ImageProcessingAPP)
        self.actionAbout_App.setObjectName("actionAbout_App")
        self.actionfrequent_questions = QtWidgets.QAction(ImageProcessingAPP)
        self.actionfrequent_questions.setObjectName("actionfrequent_questions")
        self.actionColor_Segmentation = QtWidgets.QAction(ImageProcessingAPP)
        self.actionColor_Segmentation.setObjectName("actionColor_Segmentation")
        self.actionGamma_filtre = QtWidgets.QAction(ImageProcessingAPP)
        self.actionGamma_filtre.setObjectName("actionGamma_filtre")
        self.menufile.addAction(self.actionNew)
        self.menufile.addAction(self.actionOpen)
        self.menufile.addSeparator()
        self.menufile.addAction(self.actionsave)
        self.menufile.addAction(self.actionsave_as)
        self.menufile.addAction(self.actionSave_All)
        self.menufile.addSeparator()
        self.menufile.addAction(self.actionSave_Image)
        self.menufile.addSeparator()
        self.menufile.addAction(self.actionClose)
        self.menufile.addSeparator()
        self.menufile.addAction(self.actionQuit)
        self.menuThresholding.addAction(self.actionOtsu)
        self.menuThresholding.addAction(self.actionFix_Threshold)
        self.menuHough.addAction(self.actionLines)
        self.menuHough.addAction(self.actioncircles)
        self.menuimage.addAction(self.actionGrayscale_Conversion)
        self.menuimage.addAction(self.actionHistogram)
        self.menuimage.addAction(self.menuThresholding.menuAction())
        self.menuimage.addAction(self.menuHough.menuAction())
        self.menufiltrage.addAction(self.actionMean_filtre)
        self.menufiltrage.addAction(self.actionMedain_filtre)
        self.menufiltrage.addAction(self.actionGamma_filtre)
        self.menucontour_extraction.addAction(self.actionPrewit)
        self.menucontour_extraction.addAction(self.actionSobel)
        self.menucontour_extraction.addAction(self.actionRobinson)
        self.menucontour_extraction.addAction(self.actionRoberts)
        self.menucontour_extraction.addAction(self.actionLaplacien)
        self.menumorphology.addAction(self.actionErosion)
        self.menumorphology.addAction(self.actionDelation)
        self.menumorphology.addAction(self.actionOpening)
        self.menumorphology.addAction(self.actionClosing)
        self.menumorphology.addAction(self.actionMean_Contour)
        self.menumorphology.addAction(self.actionexternal_contour)
        self.menumorphology.addAction(self.actionInner_contour)
        self.menusegmentation.addAction(self.actionRegion_growing)
        self.menusegmentation.addAction(self.actionWatershed)
        self.menusegmentation.addAction(self.actionGrabCut)
        self.menusegmentation.addAction(self.actionMean_Shift)
        self.menusegmentation.addAction(self.actionColor_Segmentation)
        self.menuHelp.addAction(self.actionContact_me)
        self.menuHelp.addAction(self.actionAbout_App)
        self.menuHelp.addAction(self.actionfrequent_questions)
        self.menubar.addAction(self.menufile.menuAction())
        self.menubar.addAction(self.menuimage.menuAction())
        self.menubar.addAction(self.menufiltrage.menuAction())
        self.menubar.addAction(self.menucontour_extraction.menuAction())
        self.menubar.addAction(self.menumorphology.menuAction())
        self.menubar.addAction(self.menusegmentation.menuAction())
        self.menubar.addAction(self.menuHelp.menuAction())

        self.retranslateUi(ImageProcessingAPP)
        self.stackedWidget.setCurrentIndex(0)
        QtCore.QMetaObject.connectSlotsByName(ImageProcessingAPP)

    def retranslateUi(self, ImageProcessingAPP):
        _translate = QtCore.QCoreApplication.translate
        ImageProcessingAPP.setWindowTitle(_translate("ImageProcessingAPP", "ImageProcessingApp"))
        self.firstimg.setText(_translate("ImageProcessingAPP", "                     Upload your image for processing "))
        self.uploadbutton.setText(_translate("ImageProcessingAPP", "Upload"))
        self.upload_gray.setText(_translate("ImageProcessingAPP", "Upload"))
        self.savegray.setText(_translate("ImageProcessingAPP", "Save"))
        self.label_6.setText(_translate("ImageProcessingAPP", "   Original image"))
        self.label_7.setText(_translate("ImageProcessingAPP", "Gray Image"))
        self.label_2.setText(_translate("ImageProcessingAPP", "Grayscale Conversion"))
        self.applygray.setText(_translate("ImageProcessingAPP", "Apply"))
        self.upload_hist.setText(_translate("ImageProcessingAPP", "Upload"))
        self.label_178.setText(_translate("ImageProcessingAPP", "Original image"))
        self.label_180.setText(_translate("ImageProcessingAPP", "Histogram"))
        self.applyhist.setText(_translate("ImageProcessingAPP", "Show"))
        self.label_177.setText(_translate("ImageProcessingAPP", "  Threshhold"))
        self.th.setText(_translate("ImageProcessingAPP", "        threshold:"))
        self.label_5.setText(_translate("ImageProcessingAPP", "    binarized image"))
        self.label_4.setText(_translate("ImageProcessingAPP", "   Origimal Image"))
        self.Upload_b.setText(_translate("ImageProcessingAPP", "Upload"))
        self.Binarize.setText(_translate("ImageProcessingAPP", "Binarize"))
        self.save_b.setText(_translate("ImageProcessingAPP", "Save"))
        self.label_175.setText(_translate("ImageProcessingAPP", "binarized image"))
        self.Binarize_6.setText(_translate("ImageProcessingAPP", "Binarize"))
        self.save_b_6.setText(_translate("ImageProcessingAPP", "Save"))
        self.label_176.setText(_translate("ImageProcessingAPP", "Origimal Image"))
        self.Upload_b_6.setText(_translate("ImageProcessingAPP", "Upload"))
        self.label.setText(_translate("ImageProcessingAPP", "   Otsu"))
        self.uplead_meanf.setText(_translate("ImageProcessingAPP", "Upload"))
        self.save_maenf.setText(_translate("ImageProcessingAPP", "Save"))
        self.applymeanf.setText(_translate("ImageProcessingAPP", "Apply Filtre"))
        self.label_3.setText(_translate("ImageProcessingAPP", "  Kernel Size"))
        self.label_8.setText(_translate("ImageProcessingAPP", "      Mean Filter"))
        self.label_181.setText(_translate("ImageProcessingAPP", "   Original Image"))
        self.label_182.setText(_translate("ImageProcessingAPP", "   Filtered Image"))
        self.label_261.setText(_translate("ImageProcessingAPP", "   Iteration"))
        self.upload_gamma.setText(_translate("ImageProcessingAPP", "Upload"))
        self.apply_gamma.setText(_translate("ImageProcessingAPP", "Apply Filtre"))
        self.label_26.setText(_translate("ImageProcessingAPP", "  Filtered Image"))
        self.save_gamma.setText(_translate("ImageProcessingAPP", "Save"))
        self.label_28.setText(_translate("ImageProcessingAPP", "   Orignal Image"))
        self.label_30.setText(_translate("ImageProcessingAPP", "Gamma"))
        self.label_31.setText(_translate("ImageProcessingAPP", "           Gamma Filter"))
        self.upload_medf.setText(_translate("ImageProcessingAPP", "Upload"))
        self.save_medf.setText(_translate("ImageProcessingAPP", "Save"))
        self.apply_medfil.setText(_translate("ImageProcessingAPP", "Apply Filtre"))
        self.label_19.setText(_translate("ImageProcessingAPP", "  Kernel Size"))
        self.label_20.setText(_translate("ImageProcessingAPP", "   Orignal Image"))
        self.label_21.setText(_translate("ImageProcessingAPP", "  Filtered Image"))
        self.label_22.setText(_translate("ImageProcessingAPP", "           Medain Filter"))
        self.label_262.setText(_translate("ImageProcessingAPP", "    Iteration"))
        self.label_101.setText(_translate("ImageProcessingAPP", "     Roberts"))
        self.save_robert.setText(_translate("ImageProcessingAPP", "Save"))
        self.Upload_robert.setText(_translate("ImageProcessingAPP", "Upload"))
        self.label_104.setText(_translate("ImageProcessingAPP", "  Roberts Contour"))
        self.dis_roberts.setText(_translate("ImageProcessingAPP", "Display Contour"))
        self.label_105.setText(_translate("ImageProcessingAPP", "   Orignal Image"))
        self.th_7.setText(_translate("ImageProcessingAPP", "       Threshold:"))
        self.checkBox_robert.setText(_translate("ImageProcessingAPP", "auto by Otsu"))
        self.hist_robert.setText(_translate("ImageProcessingAPP", "Histogram"))
        self.label_23.setText(_translate("ImageProcessingAPP", "   Orignal Image"))
        self.dis_prewitt.setText(_translate("ImageProcessingAPP", "Display Contour"))
        self.save_prewiit.setText(_translate("ImageProcessingAPP", "Save"))
        self.label_25.setText(_translate("ImageProcessingAPP", "              Prewit"))
        self.upload_prewiit.setText(_translate("ImageProcessingAPP", "Upload"))
        self.label_27.setText(_translate("ImageProcessingAPP", "    Prewit Contour"))
        self.th_8.setText(_translate("ImageProcessingAPP", "    Threshold:"))
        self.checkBox_prewitt.setText(_translate("ImageProcessingAPP", "auto by Otsu"))
        self.hist_prewiit.setText(_translate("ImageProcessingAPP", "Histogram"))
        self.label_29.setText(_translate("ImageProcessingAPP", "                      Sobel"))
        self.save_sobel.setText(_translate("ImageProcessingAPP", "Save"))
        self.upload_sobel.setText(_translate("ImageProcessingAPP", "Upload"))
        self.label_32.setText(_translate("ImageProcessingAPP", "     Sobel Contour"))
        self.display_sobel.setText(_translate("ImageProcessingAPP", "Display Contour"))
        self.label_24.setText(_translate("ImageProcessingAPP", "    Orignal Image"))
        self.th_9.setText(_translate("ImageProcessingAPP", "    Threshold:"))
        self.checkBox_sobel.setText(_translate("ImageProcessingAPP", "auto by Otsu"))
        self.hist_sobel.setText(_translate("ImageProcessingAPP", "Histogram"))
        self.label_33.setText(_translate("ImageProcessingAPP", "   Robinson Contour"))
        self.label_34.setText(_translate("ImageProcessingAPP", "           Robinson"))
        self.Save_robinson.setText(_translate("ImageProcessingAPP", "Save"))
        self.label_35.setText(_translate("ImageProcessingAPP", "    Orignal Image"))
        self.upload_robinson.setText(_translate("ImageProcessingAPP", "Upload"))
        self.disp_robinson.setText(_translate("ImageProcessingAPP", "Display Contour"))
        self.th_10.setText(_translate("ImageProcessingAPP", "       Threshold:"))
        self.checkBox_robinson.setText(_translate("ImageProcessingAPP", "auto by Otsu"))
        self.hist_robinson.setText(_translate("ImageProcessingAPP", "Histogram"))
        self.label_96.setText(_translate("ImageProcessingAPP", "Laplacien Contour"))
        self.label_97.setText(_translate("ImageProcessingAPP", "              Laplacien"))
        self.Save_laplacien.setText(_translate("ImageProcessingAPP", "Save"))
        self.label_98.setText(_translate("ImageProcessingAPP", "   Orignal Image"))
        self.upload_laplacien.setText(_translate("ImageProcessingAPP", "Upload"))
        self.dis_laplacien.setText(_translate("ImageProcessingAPP", "Display Contour"))
        self.th_11.setText(_translate("ImageProcessingAPP", "    Threshold:"))
        self.checkBox_laplacien.setText(_translate("ImageProcessingAPP", "auto by Otsu"))
        self.hist_robert_2.setText(_translate("ImageProcessingAPP", "Histogram"))
        self.label_106.setText(_translate("ImageProcessingAPP", "   Orignal Image"))
        self.apply_erosion.setText(_translate("ImageProcessingAPP", "Apply Erosion"))
        self.save_erosion.setText(_translate("ImageProcessingAPP", "Save"))
        self.label_107.setText(_translate("ImageProcessingAPP", "    Kernel Size"))
        self.label_108.setText(_translate("ImageProcessingAPP", "        Erosion"))
        self.Upload_erosion.setText(_translate("ImageProcessingAPP", "Upload"))
        self.label_110.setText(_translate("ImageProcessingAPP", "   Eroded Image"))
        self.label_263.setText(_translate("ImageProcessingAPP", "    Iteration"))
        self.label_112.setText(_translate("ImageProcessingAPP", "    Orignal Image"))
        self.apply_delation.setText(_translate("ImageProcessingAPP", "Apply delation"))
        self.save_delation.setText(_translate("ImageProcessingAPP", "Save"))
        self.label_113.setText(_translate("ImageProcessingAPP", "     Kernel Size"))
        self.label_114.setText(_translate("ImageProcessingAPP", "          Delation"))
        self.pushButton_60.setText(_translate("ImageProcessingAPP", "Upload"))
        self.label_116.setText(_translate("ImageProcessingAPP", "    Dilate Image"))
        self.label_264.setText(_translate("ImageProcessingAPP", "     Iteration"))
        self.label_118.setText(_translate("ImageProcessingAPP", "Orignal Image"))
        self.apply_opening.setText(_translate("ImageProcessingAPP", "Apply Opening"))
        self.save_opning.setText(_translate("ImageProcessingAPP", "Save"))
        self.label_119.setText(_translate("ImageProcessingAPP", "                Kernel Size"))
        self.label_120.setText(_translate("ImageProcessingAPP", "      Opening"))
        self.upload_opening.setText(_translate("ImageProcessingAPP", "Upload"))
        self.label_122.setText(_translate("ImageProcessingAPP", "    Opening"))
        self.label_124.setText(_translate("ImageProcessingAPP", "Orignal Image"))
        self.apply_closing.setText(_translate("ImageProcessingAPP", "Apply Closing"))
        self.save_closing.setText(_translate("ImageProcessingAPP", "Save"))
        self.label_125.setText(_translate("ImageProcessingAPP", "      Kernel Size"))
        self.label_126.setText(_translate("ImageProcessingAPP", "          Closing "))
        self.Upload_closing.setText(_translate("ImageProcessingAPP", "Upload"))
        self.label_128.setText(_translate("ImageProcessingAPP", "   Closing"))
        self.upload_meancountour.setText(_translate("ImageProcessingAPP", "Upload"))
        self.label_130.setText(_translate("ImageProcessingAPP", "      Orignal Image"))
        self.apply_meancountor.setText(_translate("ImageProcessingAPP", "Apply "))
        self.label_132.setText(_translate("ImageProcessingAPP", "Kernel Size"))
        self.save_meancountor.setText(_translate("ImageProcessingAPP", "Save"))
        self.label_133.setText(_translate("ImageProcessingAPP", "           Mean Contour "))
        self.label_134.setText(_translate("ImageProcessingAPP", "       Mean Contour "))
        self.Upload_ex_contors.setText(_translate("ImageProcessingAPP", "Upload"))
        self.label_136.setText(_translate("ImageProcessingAPP", "Orignal Image"))
        self.apply_ex_contour.setText(_translate("ImageProcessingAPP", "Apply "))
        self.label_138.setText(_translate("ImageProcessingAPP", "    Kernel Size"))
        self.save_ex_countor.setText(_translate("ImageProcessingAPP", "Save"))
        self.label_139.setText(_translate("ImageProcessingAPP", "    Extarnal Contour"))
        self.label_140.setText(_translate("ImageProcessingAPP", "Extarnal Contours"))
        self.Upload_inner.setText(_translate("ImageProcessingAPP", "Upload"))
        self.label_142.setText(_translate("ImageProcessingAPP", "        Orignal Image"))
        self.apply_inner.setText(_translate("ImageProcessingAPP", "Apply "))
        self.label_144.setText(_translate("ImageProcessingAPP", "   Kernel Size"))
        self.save_inner.setText(_translate("ImageProcessingAPP", "Save"))
        self.label_145.setText(_translate("ImageProcessingAPP", "      Inner Contour"))
        self.label_146.setText(_translate("ImageProcessingAPP", "            Inner Contours"))
        self.save_region.setText(_translate("ImageProcessingAPP", "Save"))
        self.label_184.setText(_translate("ImageProcessingAPP", "Orignal Image"))
        self.Upload_region.setText(_translate("ImageProcessingAPP", "Upload"))
        self.label_185.setText(_translate("ImageProcessingAPP", "    Region growing"))
        self.label_186.setText(_translate("ImageProcessingAPP", "      Region growing"))
        self.apply_region.setText(_translate("ImageProcessingAPP", "Apply "))
        self.apply_meanshift.setText(_translate("ImageProcessingAPP", "Apply "))
        self.label_252.setText(_translate("ImageProcessingAPP", "   Orignal Image"))
        self.label_253.setText(_translate("ImageProcessingAPP", "                 Mean Shift"))
        self.Upload_meanshift.setText(_translate("ImageProcessingAPP", "Upload"))
        self.save_meanshift.setText(_translate("ImageProcessingAPP", "Save"))
        self.label_254.setText(_translate("ImageProcessingAPP", "         Mean Shift"))
        self.apply_GrabCut.setText(_translate("ImageProcessingAPP", "Apply "))
        self.label_255.setText(_translate("ImageProcessingAPP", "  Orignal Image"))
        self.label_256.setText(_translate("ImageProcessingAPP", "           GrabCut"))
        self.Upload_GrabCut.setText(_translate("ImageProcessingAPP", "Upload"))
        self.save_GrabCut.setText(_translate("ImageProcessingAPP", "Save"))
        self.label_257.setText(_translate("ImageProcessingAPP", "    GrabCut"))
        self.apply_Watershed.setText(_translate("ImageProcessingAPP", "Apply "))
        self.label_258.setText(_translate("ImageProcessingAPP", "Orignal Image"))
        self.label_259.setText(_translate("ImageProcessingAPP", "      Watershed"))
        self.Upload_Watershed.setText(_translate("ImageProcessingAPP", "Upload"))
        self.save_Watershed.setText(_translate("ImageProcessingAPP", "Save"))
        self.label_260.setText(_translate("ImageProcessingAPP", "Watershed"))
        self.uplead_houglenes.setText(_translate("ImageProcessingAPP", "Upload"))
        self.label_265.setText(_translate("ImageProcessingAPP", "   Threshold :"))
        self.label_266.setText(_translate("ImageProcessingAPP", "   Lines"))
        self.label_267.setText(_translate("ImageProcessingAPP", "                  Hough Lines "))
        self.applyhoughlines.setText(_translate("ImageProcessingAPP", "Apply"))
        self.label_268.setText(_translate("ImageProcessingAPP", " Original Image"))
        self.save_houghlines.setText(_translate("ImageProcessingAPP", "Save"))
        self.applyhoughlcircles.setText(_translate("ImageProcessingAPP", "Apply "))
        self.label_269.setText(_translate("ImageProcessingAPP", "Original Image"))
        self.uplead_hougcircles.setText(_translate("ImageProcessingAPP", "Upload"))
        self.save_houghcircles.setText(_translate("ImageProcessingAPP", "Save"))
        self.label_271.setText(_translate("ImageProcessingAPP", "      Circles"))
        self.label_272.setText(_translate("ImageProcessingAPP", "       Hough Circles"))
        self.menufile.setTitle(_translate("ImageProcessingAPP", "File "))
        self.menuimage.setTitle(_translate("ImageProcessingAPP", "Image"))
        self.menuThresholding.setTitle(_translate("ImageProcessingAPP", "Thresholding"))
        self.menuHough.setTitle(_translate("ImageProcessingAPP", "Hough"))
        self.menufiltrage.setTitle(_translate("ImageProcessingAPP", "Filtrage "))
        self.menucontour_extraction.setTitle(_translate("ImageProcessingAPP", "Contour Extraction"))
        self.menumorphology.setTitle(_translate("ImageProcessingAPP", "Morphology"))
        self.menusegmentation.setTitle(_translate("ImageProcessingAPP", "Segmentation"))
        self.menuHelp.setTitle(_translate("ImageProcessingAPP", "Help"))
        self.actionOpen.setText(_translate("ImageProcessingAPP", "Open"))
        self.actionsave.setText(_translate("ImageProcessingAPP", "Save"))
        self.actionsave_as.setText(_translate("ImageProcessingAPP", "Save as "))
        self.actionGrayscale_Conversion.setText(_translate("ImageProcessingAPP", "Grayscale Conversion"))
        self.actionCropping.setText(_translate("ImageProcessingAPP", "Cropping"))
        self.actionResizing.setText(_translate("ImageProcessingAPP", "Resizing"))
        self.actionHistogram.setText(_translate("ImageProcessingAPP", "Histogram"))
        self.actionRotation_and_Flipping.setText(_translate("ImageProcessingAPP", "Rotation and Flipping"))
        self.actionMean_filtre.setText(_translate("ImageProcessingAPP", "Mean filtre"))
        self.actionMedain_filtre.setText(_translate("ImageProcessingAPP", "Medain filtre"))
        self.actionErosion.setText(_translate("ImageProcessingAPP", "Erosion"))
        self.actionDelation.setText(_translate("ImageProcessingAPP", "Delation"))
        self.actionOpening.setText(_translate("ImageProcessingAPP", "Opening"))
        self.actionClosing.setText(_translate("ImageProcessingAPP", "Closing"))
        self.actionMean_Contour.setText(_translate("ImageProcessingAPP", "Mean Contour "))
        self.actionexternal_contour.setText(_translate("ImageProcessingAPP", "External contour "))
        self.actionInner_contour.setText(_translate("ImageProcessingAPP", "Inner contour\n"
""))
        self.actionPrewit.setText(_translate("ImageProcessingAPP", "Prewit"))
        self.actionSobel.setText(_translate("ImageProcessingAPP", "Sobel"))
        self.actionRobinson.setText(_translate("ImageProcessingAPP", "Robinson"))
        self.actionRoberts.setText(_translate("ImageProcessingAPP", "Roberts"))
        self.actionLaplacien.setText(_translate("ImageProcessingAPP", "Laplacien"))
        self.actionOtsu.setText(_translate("ImageProcessingAPP", "Otsu"))
        self.actionFix_Threshold.setText(_translate("ImageProcessingAPP", "Fix Threshold"))
        self.actionRegion_growing.setText(_translate("ImageProcessingAPP", "Region growing"))
        self.actionRegion_growing.setIconText(_translate("ImageProcessingAPP", "Region growing"))
        self.actionWatershed.setText(_translate("ImageProcessingAPP", "Watershed"))
        self.actionGrabCut.setText(_translate("ImageProcessingAPP", "GrabCut"))
        self.actionMean_Shift.setText(_translate("ImageProcessingAPP", "Mean Shift"))
        self.actionLines.setText(_translate("ImageProcessingAPP", "Lines"))
        self.actioncircles.setText(_translate("ImageProcessingAPP", "circles"))
        self.actionNew.setText(_translate("ImageProcessingAPP", "New"))
        self.actionSave_All.setText(_translate("ImageProcessingAPP", "Save All"))
        self.actionSave_Image.setText(_translate("ImageProcessingAPP", "Save Image"))
        self.actionClose.setText(_translate("ImageProcessingAPP", "Close"))
        self.actionQuit.setText(_translate("ImageProcessingAPP", "Quit "))
        self.actionContact_me.setText(_translate("ImageProcessingAPP", "Contact me\n"
""))
        self.actionAbout_App.setText(_translate("ImageProcessingAPP", "About App"))
        self.actionfrequent_questions.setText(_translate("ImageProcessingAPP", "frequent questions"))
        self.actionColor_Segmentation.setText(_translate("ImageProcessingAPP", "Color Segmentation"))
        self.actionGamma_filtre.setText(_translate("ImageProcessingAPP", "Gamma filtre"))

        ################## navigatiion#################################################################
        self.actionGrayscale_Conversion.triggered.connect(lambda: self.stackedWidget.setCurrentWidget(self.grayconv))
        self.actionHistogram.triggered.connect(lambda: self.stackedWidget.setCurrentWidget(self.histogram))
        self.actionGamma_filtre.triggered.connect(lambda: self.stackedWidget.setCurrentWidget(self.gamma_F))
        self.actionFix_Threshold.triggered.connect(lambda: self.stackedWidget.setCurrentWidget(self.throu))
        self.actionOtsu.triggered.connect(lambda: self.stackedWidget.setCurrentWidget(self.otsu))
        self.actionLines.triggered.connect(lambda: self.stackedWidget.setCurrentWidget(self.hough_lines))
        self.actioncircles.triggered.connect(lambda: self.stackedWidget.setCurrentWidget(self.hough_circeles))
        self.actionMean_filtre.triggered.connect(lambda: self.stackedWidget.setCurrentWidget(self.Mean_f))
        self.actionMedain_filtre.triggered.connect(lambda: self.stackedWidget.setCurrentWidget(self.Medain_f))
        self.actionPrewit.triggered.connect(lambda: self.stackedWidget.setCurrentWidget(self.Prewit))
        self.actionSobel.triggered.connect(lambda: self.stackedWidget.setCurrentWidget(self.Sobel))
        self.actionRobinson.triggered.connect(lambda: self.stackedWidget.setCurrentWidget(self.Robinson))
        self.actionRoberts.triggered.connect(lambda: self.stackedWidget.setCurrentWidget(self.Roberts))
        self.actionLaplacien.triggered.connect(lambda: self.stackedWidget.setCurrentWidget(self.Laplacien))
        self.actionErosion.triggered.connect(lambda: self.stackedWidget.setCurrentWidget(self.Erosion))
        self.actionDelation.triggered.connect(lambda: self.stackedWidget.setCurrentWidget(self.dellation))
        self.actionOpening.triggered.connect(lambda: self.stackedWidget.setCurrentWidget(self.opening))
        self.actionClosing.triggered.connect(lambda: self.stackedWidget.setCurrentWidget(self.closing))
        self.actionMean_Contour.triggered.connect(lambda: self.stackedWidget.setCurrentWidget(self.maencontour))
        self.actionexternal_contour.triggered.connect(lambda: self.stackedWidget.setCurrentWidget(self.extarnalcontour))
        self.actionInner_contour.triggered.connect(lambda: self.stackedWidget.setCurrentWidget(self.inner))
        self.actionRegion_growing.triggered.connect(lambda: self.stackedWidget.setCurrentWidget(self.region_growing))
        self.actionWatershed.triggered.connect(lambda: self.stackedWidget.setCurrentWidget(self.Watershed))
        self.actionGrabCut.triggered.connect(lambda: self.stackedWidget.setCurrentWidget(self.GrabCut))
        self.actionMean_Shift.triggered.connect(lambda: self.stackedWidget.setCurrentWidget(self.mean_shift))
        self.actionNew.triggered.connect(lambda: self.stackedWidget.setCurrentWidget(self.homepage))
       ##############################################################################################################
        self.uploadbutton.clicked.connect(self.upload_homepage)
        self.upload_gray.clicked.connect(self.upload_grayy)
        self.actionColor_Segmentation.triggered.connect(self.open_color_segmentation_page)

        
        self.applygray.clicked.connect(self.apply_gray)
        self.savegray.clicked.connect(self.save_gray)

        
        
        self.Upload_b.clicked.connect(self.upload_binaire)
        self.Binarize.clicked.connect(self.apply_binarie_fix)
        
        self.Upload_b_6.clicked.connect(self.upload_binaire_otsu)
        self.Binarize_6.clicked.connect(self.apply_binarie_otsu)
        
        self.applyhist.clicked.connect(self.dis_histogram)
        self.upload_hist.clicked.connect(self.upload_histt)
        
        
        
        ##action filter 
        self.uplead_meanf.clicked.connect(self.upload_mean_f)
        self.applymeanf.clicked.connect(self.apply_mean_f)
        
        ##action filter 
        self.upload_medf.clicked.connect(self.upload_median_f)
        self.apply_medfil.clicked.connect(self.apply_medain_f)
        self.upload_gamma.clicked.connect(self.upload_gamma_f)
        self.apply_gamma.clicked.connect(self.apply_gamma_f)
        self.save_gamma.clicked.connect(self.save_gamma_f)
        ##morphology 
        self.Upload_erosion.clicked.connect(self.upload_erosion)
        self.apply_erosion.clicked.connect(self.apply_erosionn) 
             
        self.pushButton_60.clicked.connect(self.upload_delationn)
        self.apply_delation.clicked.connect(self.apply_delationn)  
        
        self.upload_opening.clicked.connect(self.upload_openingg)
        self.apply_opening.clicked.connect(self.apply_openingg) 
        
        self.Upload_closing.clicked.connect(self.upload_closingg)
        self.apply_closing.clicked.connect(self.apply_closingg)  
        
        self.Upload_inner.clicked.connect(self.upload_innerr)
        self.apply_inner.clicked.connect(self.apply_innerr)  
        
        self.Upload_ex_contors.clicked.connect(self.upload_extarnal)
        self.apply_ex_contour.clicked.connect(self.apply_extarnal)
        
        self.upload_meancountour.clicked.connect(self.upload_mean_co)
        self.apply_meancountor.clicked.connect(self.apply_mean_co)
        
        
        self.upload_prewiit.clicked.connect(self.upload_prewitt)
        self.dis_prewitt.clicked.connect(self.apply_prewitt)
        self.upload_sobel.clicked.connect(self.upload_sobell)
        self.display_sobel.clicked.connect(self.apply_sobell)
        
        self.upload_robinson.clicked.connect(self.upload_Robinson)
        self.disp_robinson.clicked.connect(self.apply_Robinson)
        self.Upload_robert.clicked.connect(self.upload_roberts)
        self.dis_roberts.clicked.connect(self.apply_roberts)
        self.upload_laplacien.clicked.connect(self.upload_Laplacien)
        self.dis_laplacien.clicked.connect(self.apply_Laplacienn)
        
        
        
        self.uplead_houglenes.clicked.connect(self.upload_hough_lines)
        self.applyhoughlines.clicked.connect(self.apply_hough_lines)
        
        self.uplead_hougcircles.clicked.connect(self.upload_hough_circeles)
        self.applyhoughlcircles.clicked.connect(self.apply_hough_circeles)
        
        
        
        self.apply_Watershed.clicked.connect(self.apply_watershed)
        self.Upload_Watershed.clicked.connect(self.upload_watershed)
        
        
        self.apply_GrabCut.clicked.connect(self.apply_grabcut)
        self.Upload_GrabCut.clicked.connect(self.upload_grabcut)
        
        self.apply_meanshift.clicked.connect(self.apply_meanshifftt)
        self.Upload_meanshift.clicked.connect(self.upload_meansshiftt)
        
        self.apply_region.clicked.connect(self.apply_growing)
        self.Upload_region.clicked.connect(self.upload_growing)
        
        self.save_closing.clicked.connect(self.save_closingg)
        self.save_opning.clicked.connect(self.save_openingg)
        
        self.save_erosion.clicked.connect(self.save_erosionn)
        self.save_delation.clicked.connect(self.save_delationn)
        self.save_meancountor.clicked.connect(self.save_mean_co)
        self.save_ex_countor.clicked.connect(self.save_extarnal)
        self.save_inner.clicked.connect(self.save_innerr)
        
        
        self.save_maenf.clicked.connect(self.save_mean_f)
        self.save_medf.clicked.connect(self.save_medain_f)
        
        self.save_robert.clicked.connect(self.save_roberts)
        self.save_prewiit.clicked.connect(self.save_prewitt)
        self.save_sobel.clicked.connect(self.save_sobell)
        self.Save_robinson.clicked.connect(self.save_Robinson)
        self.Save_laplacien.clicked.connect(self.save_Laplacienn)
        
        self.save_houghlines.clicked.connect(self.save_hough_lines)
        self.save_houghcircles.clicked.connect(self.save_hough_circeles)

        self.hist_prewiit.clicked.connect(self.hist_prewitt)
        self.hist_robert.clicked.connect(self.hisst_roberts)
        self.hist_robinson.clicked.connect(self.histtt_Robinson)
        self.hist_sobel.clicked.connect(self.histtt_sobell)
        self.hist_robert_2.clicked.connect(self.hisst_Laplacienn)
        self.checkbox_state_prewitt = Qt.Unchecked  # Initial state
        self.checkbox_state_sobel = Qt.Unchecked  # Initial state
        self.checkbox_state_robert = Qt.Unchecked  # Initial state
        self.checkbox_state_robinson = Qt.Unchecked  # Initial state
        self.checkbox_state_laplacien = Qt.Unchecked  # Initial state
        
        self.checkBox_robert.stateChanged.connect(self.on_checkbox_robert)
        self.checkBox_prewitt.stateChanged.connect(self.on_checkbox_prewitt)
        self.checkBox_robinson.stateChanged.connect(self.on_checkbox_robinson)
        self.checkBox_sobel.stateChanged.connect(self.on_checkbox_sobel)
        self.checkBox_laplacien.stateChanged.connect(self.on_checkbox_laplacien)
       


        



        self.file_name=' '    

        
        
    def upload_homepage(self):
        self.file_name=self.upload_()
        if self.file_name:
            # Read the original image using OpenCV
            original_image = cv2.imread(self.file_name)

            if original_image is not None:
                # Convert the original image to RGB format
                original_image_rgb = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
                # Display the original image in the first QLabel
                self.display_image(original_image_rgb, self.firstimg)
                self.display_image(original_image_rgb, self.aftergray)
                
                
 #################################################################################################  
    def show_message_box(self, message, icon=QMessageBox.Information, buttons=QMessageBox.Ok):
                    app = QApplication.instance() or QApplication(sys.argv)
                    message_box = QMessageBox()
                    message_box.setText(message)
                    message_box.setIcon(icon)
                    message_box.setStandardButtons(buttons)
                    result = message_box.exec_()
                    if result == QMessageBox.Ok:
                        print("Ok button clicked")
                    else:
                        print("Cancel button clicked")
                    return result

# Example usage:
             
    def upload_grayy(self):
        self.file_name=self.upload_()
        if self.file_name:
            # Read the original image using OpenCV
            original_image = cv2.imread(self.file_name)
            if original_image is not None:
                # Convert the original image to RGB format
                original_image_rgb = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
                # Display the original image in the first QLabel
                self.display_image(original_image_rgb, self.aftergray)
    def apply_gray(self):
           if self.file_name:
               original_image = cv2.imread(self.file_name)
               if original_image is not None:
                          gray_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
                          self.display_image(gray_image, self.beforegray)
    def save_gray(self):
           if self.file_name:
               original_image = cv2.imread(self.file_name)
               if original_image is not None:
                          gray_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
                          self.save_image(gray_image)                    




#################################################################################################################        
    def upload_binaire(self):
        self.file_name=self.upload_()
        if self.file_name:
            # Read the original image using OpenCV
            original_image = cv2.imread(self.file_name)
            if original_image is not None:
                # Convert the original image to RGB format
                original_image_rgb = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
                # Display the original image in the first QLabel
                self.display_image(original_image_rgb, self.before_b)
                
    def apply_binarie_fix(self):
           print(self.file_name)
           if self.file_name:
               original_image = cv2.imread(self.file_name)
               if original_image is not None:
                          value = self.threshhold.value()
                          gray_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
                          _ , binary_image = cv2.threshold(gray_image, value, 255, cv2.THRESH_BINARY)
                          self.display_image(binary_image, self.after_b)
    def save_binarie_fix(self):
           if self.file_name:
               original_image = cv2.imread(self.file_name)
               if original_image is not None:
                          value = self.threshhold.value()
                          gray_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
                          _, binary_image = cv2.threshold(gray_image, value, 255, cv2.THRESH_BINARY)
                          self.save_image(binary_image)
    
    
    
    def upload_binaire_otsu(self):
        self.file_name=self.upload_()
        if self.file_name:
            # Read the original image using OpenCV
            original_image = cv2.imread(self.file_name)
            if original_image is not None:
                # Convert the original image to RGB format
                original_image_rgb = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
                # Display the original image in the first QLabel
                self.display_image(original_image_rgb, self.before_b_6)
                
    def apply_binarie_otsu(self):
           if self.file_name:
               original_image = cv2.imread(self.file_name)
               if original_image is not None:
                          gray_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
                          _, binary_image = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                          self.display_image(binary_image, self.after_b_6)
    def save_binarie_otsu(self):
           if self.file_name:
               original_image = cv2.imread(self.file_name)
               if original_image is not None:
                          value = self.threshhold.value()
                          gray_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
                          _, binary_image = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                          self.save_image(binary_image)          

    def display_image(self, image, label):
        height, width = image.shape[:2]  # Use slicing to get height and width
        channels = image.shape[2] if len(image.shape) == 3 else 1  # Check the number of channels

        bytes_per_line = channels * width
        q_image = QImage(image.data, width, height, bytes_per_line, QImage.Format_RGB888 if channels == 3 else QImage.Format_Grayscale8)

        pixmap = QPixmap.fromImage(q_image)
        label.setPixmap(pixmap)
        label.setScaledContents(True)
        label.setAlignment(Qt.AlignCenter)
####################################################################################################################################################
    def upload_histt (self):
        self.file_name=self.upload_()
        if self.file_name:
            # Read the original image using OpenCV
            original_image = cv2.imread(self.file_name)
            if original_image is not None:
                          gray_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
                          self.display_image(gray_image, self.before_hist)  
    def show_histogram(self,image):
     if image is not None:
        plt.figure()
        plt.title("Image Histogram")
        plt.xlabel("Pixel Value")
        plt.ylabel("Frequency")
        plt.hist(image.flatten(), bins=256, range=[0, 256], color=(0, 191/255, 255/255), edgecolor='black')
        plt.grid()

        canvas = FigureCanvas(plt.gcf())
        layout = QVBoxLayout(self.centralwidget)
        layout.addWidget(canvas)
        plt.show()

    def dis_histogram(self):
        if self.file_name:
            # Read the original image using OpenCV
            original_image = cv2.imread(self.file_name)
            if original_image is not None:
                        gray_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
                        self.show_histogram(gray_image)

                   



#####################################################################Meanfilter############################################
    def upload_mean_f(self):
        self.file_name=self.upload_()
        if self.file_name:
            original_image = cv2.imread(self.file_name)
            if original_image is not None:
                gray_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
                self.display_image(gray_image, self.beforemeanfil)
                
    def apply_mean_f(self):
           print(self.file_name)
           if self.file_name:
               original_image = cv2.imread(self.file_name)
               if original_image is not None:
                          value = self.k_mean.value()
                          itrr = self.mean_f_iteration.value()
                          gray_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
                          if value%2 != 0:
                                im =self.filter_moy_gen(gray_image,value,itrr)
                                self.display_image(im, self.aftermean_f)
                          else :
                                 result = self.show_message_box("Hello,the kernal must be even!", icon=QMessageBox.Information, buttons=QMessageBox.Ok | QMessageBox.Cancel)
    def save_mean_f(self):
           if self.file_name:
               original_image = cv2.imread(self.file_name)
               if original_image is not None:
                          value = self.k_mean.value()
                          itrr = self.mean_f_iteration.value()
                          gray_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
                          if value%2 != 0:
                                im =self.filter_moy_gen(gray_image,value,itrr)
                                self.save_image(im)
                          else :
                                 result = self.show_message_box("Hello,the kernal must be even!", icon=QMessageBox.Information, buttons=QMessageBox.Ok | QMessageBox.Cancel)

    def filter_moy_gen(self,image,f_size,iteration):
                iterr=(f_size-1)//2
                m,n = image.shape[0],image.shape[1]
                imageA=np.zeros_like(image)
                for k in range(iteration) :
                     for i in range(iterr,m-iterr):
                        for j in range(iterr,n-iterr):
                                imageA[i,j]= np.mean( image[i-iterr:i+iterr+1,j-iterr:j+iterr+1])
                     image=imageA
                return image 
        
        
        
    def upload_gamma_f(self):
        self.file_name=self.upload_()
        if self.file_name:
            original_image = cv2.imread(self.file_name)
            if original_image is not None:
                original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
                self.display_image(original_image, self.before_gamma)
                
    def apply_gamma_f(self):
           print(self.file_name)
           if self.file_name:
               original_image = cv2.imread(self.file_name)
               if original_image is not None:
                          original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
                          value = self.gamma.value()
                          im =self.adjust_gamma(original_image,value)
                          self.display_image(im, self.after_gamma)
                          
    def save_gamma_f(self):
           if self.file_name:
               original_image = cv2.imread(self.file_name)
               if original_image is not None:
                          original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
                          value = self.gamma.value()
                          im =self.adjust_gamma(original_image,value)
                          self.save_image(im)

    def adjust_gamma(self,image, gamma=1.0):
                        # build a lookup table mapping the pixel values [0, 255] to their adjusted gamma values
                        inv_gamma = 1.0 / gamma
                        table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
                        
                        # apply gamma correction using the lookup table
                        return cv2.LUT(image, table)
####################################################MEdian_filter#######################################################
    def upload_median_f(self):
        self.file_name=self.upload_()
        if self.file_name:
            original_image = cv2.imread(self.file_name)
            if original_image is not None:
                gray_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
                self.display_image(gray_image, self.before_medf)
                
    def apply_medain_f(self):
           print(self.file_name)
           if self.file_name:
               original_image = cv2.imread(self.file_name)
               if original_image is not None:
                          value = self.k_medfil.value()
                          itr = self.medain_f_iteration.value()
                          gray_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
                          if value%2 != 0:
                                im =self.fil_median(gray_image,value,itr)
                                self.display_image(im, self.after_medf)
                          else :
                                 result = self.show_message_box("Hello,the kernal must be even!", icon=QMessageBox.Information, buttons=QMessageBox.Ok | QMessageBox.Cancel)
    def save_medain_f(self):
           print(self.file_name)
           if self.file_name:
               original_image = cv2.imread(self.file_name)
               if original_image is not None:
                          value = self.k_medfil.value()
                          itr = self.medain_f_iteration.value()
                          gray_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
                          if value%2 != 0:
                                im =self.fil_median(gray_image,value,itr)
                                self.save_image(im)
                          else :
                                 result = self.show_message_box("Hello,the kernal must be even!", icon=QMessageBox.Information, buttons=QMessageBox.Ok | QMessageBox.Cancel)
    def fil_median(self,image,f_size,iteration):
                iterr=(f_size-1)//2
                m,n = image.shape[0],image.shape[1]
                imageA=np.zeros_like(image)
                for k in range(iteration) :
                   for i in range(iterr,m-iterr):
                        for j in range(iterr,n-iterr):
                                arry= image[i-iterr:i+iterr+1,j-iterr:j+iterr+1]
                                arry=arry.flatten()
                                arry=sorted(arry)
                                imageA[i,j]= np.median(arry)
                   image=imageA

                return image  
##############################################################Morphology##############################################
    def upload_erosion(self):
        self.file_name=self.upload_()
        if self.file_name:
            original_image = cv2.imread(self.file_name)
            if original_image is not None:
                gray_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
                _, binary_image = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                self.display_image(binary_image, self.before_erosion)
                
    def apply_erosionn(self):
           print(self.file_name)
           if self.file_name:
               original_image = cv2.imread(self.file_name)
               if original_image is not None:
                          value = self.k_erosion.value()
                          itr = self.erosion_iteration.value()
                          gray_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
                          _, binary_image = cv2.threshold(gray_image, 0, 1, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                          if value%2 != 0:
                                im =self.erosion(binary_image,value,itr)
                                im[im == 1] = 255
                                self.display_image(im, self.after_erosion)
                          else :
           
                              result = self.show_message_box("Hello,the kernal must be even!", icon=QMessageBox.Information, buttons=QMessageBox.Ok | QMessageBox.Cancel)
   
   
    def save_erosionn(self):
           print(self.file_name)
           if self.file_name:
               original_image = cv2.imread(self.file_name)
               if original_image is not None:
                          value = self.k_erosion.value()
                          itr = self.erosion_iteration.value()
                          gray_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
                          _, binary_image = cv2.threshold(gray_image, 0, 1, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                          if value%2 != 0:
                                im =self.erosion(binary_image,value,itr)
                                im[im == 1] = 255
                                self.save_image(im)
                          else :
                                 result = self.show_message_box("Hello,the kernal must be even!", icon=QMessageBox.Information, buttons=QMessageBox.Ok | QMessageBox.Cancel)
   
   
    def erosion(self,image,kernel_size,itration):
                                m,n = image.shape[0],image.shape[1]
                                iterr=(kernel_size-1)//2
                                imageA=np.zeros_like(image)
                                for g in range(itration):
                                    for i in range(iterr,m-iterr):
                                        for j in range(iterr,n-iterr):
                                                part= image[i-iterr:i+iterr+1,j-iterr:j+iterr+1]
                                                k=np.sum(part)
                                                if k==kernel_size*kernel_size:
                                                       imageA[i,j] = 1
                                                else:
                                                        imageA[i,j] = 0
                                    image=imageA
                                return image  
        
   
   
    def upload_delationn(self):
        self.file_name=self.upload_()
        if self.file_name:
            original_image = cv2.imread(self.file_name)
            if original_image is not None:
                gray_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
                _, binary_image = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                self.display_image(binary_image, self.before_delation)
                
    def apply_delationn(self):
           print(self.file_name)
           if self.file_name:
               original_image = cv2.imread(self.file_name)
               if original_image is not None:
                          value = self.k_delarion.value()
                          itr = self.delatation_iteration.value()
                          gray_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
                          _, binary_image = cv2.threshold(gray_image, 0, 1, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                          if value%2 != 0:
                                im =self.delation(binary_image,value,itr)
                                im[im == 1] = 255
                                self.display_image(im, self.after_delation)
                          else :
                                 result = self.show_message_box("Hello,the kernal must be even!", icon=QMessageBox.Information, buttons=QMessageBox.Ok | QMessageBox.Cancel)
    def save_delationn(self):
           print(self.file_name)
           if self.file_name:
               original_image = cv2.imread(self.file_name)
               if original_image is not None:
                          value = self.k_delarion.value()
                          itr = self.delatation_iteration.value()
                          gray_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
                          _, binary_image = cv2.threshold(gray_image, 0, 1, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                          if value%2 != 0:
                                im =self.delation(binary_image,value,itr)
                                im[im == 1] = 255
                                self.save_image(im)
                          else :
                                 result = self.show_message_box("Hello,the kernal must be even!", icon=QMessageBox.Information, buttons=QMessageBox.Ok | QMessageBox.Cancel)                             
                                 
        
    def delation(self,image,kernel_size,itration):
                                m,n = image.shape[0],image.shape[1]
                                iterr=(kernel_size-1)//2
                                imageA=np.zeros_like(image)
                                for g in range(itration):
                                    for i in range(iterr,m-iterr):
                                        for j in range(iterr,n-iterr):
                                             part= image[i-iterr:i+iterr+1,j-iterr:j+iterr+1]
                                             k=np.sum(part)
                                             if k==0:
                                                imageA[i,j] = 0
                                             else:
                                                imageA[i,j] = 1
                                    image=imageA
                                return image  
################################### opening ########

    def upload_closingg(self):
        self.file_name=self.upload_()
        if self.file_name:
            original_image = cv2.imread(self.file_name)
            if original_image is not None:
                gray_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
                _, binary_image = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                self.display_image(binary_image, self.before_closing)
                
    def apply_closingg(self):
           print(self.file_name)
           if self.file_name:
               original_image = cv2.imread(self.file_name)
               if original_image is not None:
                          value = self.k_closing.value()
                          gray_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
                          _, binary_image = cv2.threshold(gray_image, 0, 1, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                          if value%2 != 0:
                                im =self.closingg(binary_image,value,1)
                                im[im == 1] = 255
                                self.display_image(im, self.after_closing)
                          else :
                                 result = self.show_message_box("Hello,the kernal must be even!", icon=QMessageBox.Information, buttons=QMessageBox.Ok | QMessageBox.Cancel)
    def save_closingg(self):
           print(self.file_name)
           if self.file_name:
               original_image = cv2.imread(self.file_name)
               if original_image is not None:
                          value = self.k_closing.value()
                          gray_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
                          _, binary_image = cv2.threshold(gray_image, 0, 1, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                          if value%2 != 0:
                                im =self.closingg(binary_image,value,1)
                                im[im == 1] = 255
                                self.save_image(im)
                          else :
                                 result = self.show_message_box("Hello,the kernal must be even!", icon=QMessageBox.Information, buttons=QMessageBox.Ok | QMessageBox.Cancel)
    def closingg(self,image,kernel_size,iteration):
        imageA=self.delation(image,kernel_size,iteration)
        img=self.erosion(imageA,kernel_size,iteration)
        return img

    def upload_openingg(self):
        self.file_name=self.upload_()
        if self.file_name:
            original_image = cv2.imread(self.file_name)
            if original_image is not None:
                gray_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
                _, binary_image = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                self.display_image(binary_image, self.before_opening)
                
    def apply_openingg(self):
           print(self.file_name)
           if self.file_name:
               original_image = cv2.imread(self.file_name)
               if original_image is not None:
                          value = self.k_opning.value()
                          gray_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
                          _, binary_image = cv2.threshold(gray_image, 0, 1, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                          if value%2 != 0:
                                im =self.openingg(binary_image,value,1)
                                im[im == 1] = 255
                                self.display_image(im, self.after_opening)
                          else :
                                 result = self.show_message_box("Hello,the kernal must be even!", icon=QMessageBox.Information, buttons=QMessageBox.Ok | QMessageBox.Cancel)
    def save_openingg(self):
           if self.file_name:
               original_image = cv2.imread(self.file_name)
               if original_image is not None:
                          value = self.k_opning.value()
                          gray_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
                          _, binary_image = cv2.threshold(gray_image, 0, 1, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                          if value%2 != 0:
                                im =self.openingg(binary_image,value,1)
                                im[im == 1] = 255
                                self.save_image(im)
                          else :
                                 result = self.show_message_box("Hello,the kernal must be even!", icon=QMessageBox.Information, buttons=QMessageBox.Ok | QMessageBox.Cancel)
       
    
    def openingg(self,image,kernel_size,iteration):
        img=self.erosion(image,kernel_size,iteration)
        imageA=self.delation(img,kernel_size,iteration)
        return imageA
      
      
      
      

    def upload_innerr(self):
        self.file_name=self.upload_()
        if self.file_name:
            original_image = cv2.imread(self.file_name)
            if original_image is not None:
                gray_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
                _, binary_image = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                self.display_image(binary_image, self.beffore_inner)
                
    def apply_innerr(self):
           print(self.file_name)
           if self.file_name:
               original_image = cv2.imread(self.file_name)
               if original_image is not None:
                          value = self.k_inner.value()
                          gray_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
                          _, binary_image = cv2.threshold(gray_image, 0, 1, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                          if value%2 != 0:
                                im =self.innercontour(binary_image,value,1)
                                im[im == 1] = 255
                                self.display_image(im, self.after_inner)
                          else :
                                 result = self.show_message_box("Hello,the kernal must be even!", icon=QMessageBox.Information, buttons=QMessageBox.Ok | QMessageBox.Cancel)
    def save_innerr(self):
           print(self.file_name)
           if self.file_name:
               original_image = cv2.imread(self.file_name)
               if original_image is not None:
                          value = self.k_inner.value()
                          gray_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
                          _, binary_image = cv2.threshold(gray_image, 0, 1, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                          if value%2 != 0:
                                im =self.innercontour(binary_image,value,1)
                                im[im == 1] = 255
                                self.save_image(im)
                          else :
                                 result = self.show_message_box("Hello,the kernal must be even!", icon=QMessageBox.Information, buttons=QMessageBox.Ok | QMessageBox.Cancel)   
    
    def innercontour(self,image,kernel_size,iteration):
        img=self.erosion(image,kernel_size,iteration)
        return image-img 
      



    def upload_extarnal(self):
        self.file_name=self.upload_()
        if self.file_name:
            original_image = cv2.imread(self.file_name)
            if original_image is not None:
                gray_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
                _, binary_image = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                self.display_image(binary_image, self.before_ex_contours)
                
    def apply_extarnal(self):
           print(self.file_name)
           if self.file_name:
               original_image = cv2.imread(self.file_name)
               if original_image is not None:
                          value = self.k_exe_contours.value()
                          gray_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
                          _, binary_image = cv2.threshold(gray_image, 0, 1, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                          if value%2 != 0:
                                im =self.extarnalcontourr(binary_image,value,1)
                                im[im == 1] = 255
                                self.display_image(im, self.after_exer_contour)
                          else :
                                 result = self.show_message_box("Hello,the kernal must be even!", icon=QMessageBox.Information, buttons=QMessageBox.Ok | QMessageBox.Cancel)
    def save_extarnal(self):
           print(self.file_name)
           if self.file_name:
               original_image = cv2.imread(self.file_name)
               if original_image is not None:
                          value = self.k_exe_contours.value()
                          gray_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
                          _, binary_image = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                          if value%2 != 0:
                                im =self.extarnalcontourr(binary_image,value,1)
                                im[im == 1] = 255
                                self.save_image(im)
                          else :
                                 result = self.show_message_box("Hello,the kernal must be even!", icon=QMessageBox.Information, buttons=QMessageBox.Ok | QMessageBox.Cancel)
       
    
    def extarnalcontourr(self,image,kernel_size,iteration):
        img=self.delation(image,kernel_size,iteration)
        return image-img





    def upload_mean_co(self):
        self.file_name=self.upload_()
        if self.file_name:
            original_image = cv2.imread(self.file_name)
            if original_image is not None:
                gray_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
                _, binary_image = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                self.display_image(binary_image, self.before_contor)
                
    def apply_mean_co(self):
           print(self.file_name)
           if self.file_name:
               original_image = cv2.imread(self.file_name)
               if original_image is not None:
                          value = self.k_meancontour.value()
                          gray_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
                          _, binary_image = cv2.threshold(gray_image, 0, 1, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                          if value%2 != 0:
                                im =self.meancon(binary_image,value,1)
                                im[im == 1] = 255
                                self.display_image(im, self.after_meancountor)
                          else :
                                 result = self.show_message_box("Hello,the kernal must be even!", icon=QMessageBox.Information, buttons=QMessageBox.Ok | QMessageBox.Cancel)
    def save_mean_co(self):
           print(self.file_name)
           if self.file_name:
               original_image = cv2.imread(self.file_name)
               if original_image is not None:
                          value = self.k_meancontour.value()
                          gray_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
                          _, binary_image = cv2.threshold(gray_image, 0, 1, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                          if value%2 != 0:
                                im =self.meancon(binary_image,value,1)
                                im[im == 1] = 255
                                self.save_image(im)
                          else :
                                 result = self.show_message_box("Hello,the kernal must be even!", icon=QMessageBox.Information, buttons=QMessageBox.Ok | QMessageBox.Cancel)
   
   
    def meancon(self,image,kernel_size,iteration):
        imageA=self.delation(image,kernel_size,iteration)
        img=self.erosion(image,kernel_size,iteration)
        return imageA-img


#######################################################countors 
    def on_checkbox_robert(self, state):
        self.checkbox_state_robert = state
    def on_checkbox_prewitt(self, state):
        self.checkbox_state_prewitt = state
    def on_checkbox_robinson(self, state):
        self.checkbox_state_robinson = state
    def on_checkbox_sobel(self, state):
        self.checkbox_state_sobel = state
    def on_checkbox_laplacien(self, state):
        self.checkbox_state_laplacien = state
        
        
    def model_ggg(self,listt):
        image=np.zeros_like(listt[0])
        for li in listt :
                image=image+np.square(li)
        image=np.sqrt(image)  
        return np.uint8(image)
    def G_prewit(self,image):
            mask1=[[1,1, 1],
                [0,0, 0],
                [-1,-1,-1]]
            mask2=[[-1,0,1],
                [-1,0,1],
                [-1,0,1]]
            m,n = image.shape[0],image.shape[1]
            Gx=np.zeros_like(image)
            Gy=np.zeros_like(image)
            for i in range(1,m-1):
                for j in range(1,n-1):
                        part= image[i-1:i+1+1,j-1:j+1+1]
                        Gx[i,j]=abs(np.sum(part*mask1))            
                        Gy[i,j]=abs(np.sum(part*mask2))   
            return [Gx,Gy]
    
          
    def upload_prewitt(self):
        self.file_name=self.upload_()
        if self.file_name:
            original_image = cv2.imread(self.file_name)
            if original_image is not None:
                gray_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
                _, binary_image = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                self.display_image(binary_image, self.before_prewitt)
                
    def apply_prewitt(self):
           print(self.file_name)
           if self.file_name:
               original_image = cv2.imread(self.file_name)
               if original_image is not None:
                          value = self.prewit_th.value()
                          gray_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
                          im =self.prewitt(gray_image)
                          if self.checkbox_state_prewitt==False:  
                                ret, im_binary_image = cv2.threshold(im, value, 255, cv2.THRESH_BINARY)
                                self.display_image(im_binary_image, self.after_prewiit)
                          else: 
                               _, im_binary_image = cv2.threshold(im, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                               self.display_image(im_binary_image, self.after_prewiit)
    def save_prewitt(self):
           print(self.file_name)
           if self.file_name:
               original_image = cv2.imread(self.file_name)
               if original_image is not None:
                          value = self.prewit_th.value()
                          gray_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
                          im =self.prewitt(gray_image)                          
                          if self.checkbox_state_prewitt==False:  
                                ret, im_binary_image = cv2.threshold(im, value, 255, cv2.THRESH_BINARY)
                                self.save_image(im_binary_image)                         
                          else: 
                               _, im_binary_image = cv2.threshold(im, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                               self.save_image(im_binary_image)                         
                        
    def hist_prewitt(self):
           print(self.file_name)
           if self.file_name:
               original_image = cv2.imread(self.file_name)
               if original_image is not None:
                          value = self.prewit_th.value()
                          gray_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
                          im =self.prewitt(gray_image)
                          self.show_histogram(im)
    def prewitt(self,image):
        li=self.G_prewit(image)
        img=self.model_ggg(li)
        return img

##############################################################################
    def G_sobel(self,image):
                        mask1=[[1,2,1],
                                [0,0,0],
                                [-1,-2,-1]]
                        mask2=[[-1,0,1],
                                [-2,0,2],
                                [-1,0,1]]
                        m,n = image.shape[0],image.shape[1]
                        image =np.float32(image)
                        Gx=np.zeros_like(image)
                        Gy=np.zeros_like(image)
                        for i in range(1,m-1):
                                for j in range(1,n-1):
                                        part= image[i-1:i+1+1,j-1:j+1+1]
                                        Gx[i,j]=abs(np.sum(part*mask1))
                                        Gy[i,j]=abs(np.sum(part*mask2))       
                        return [Gx,Gy]
    def sobell(self,image):
        li=self.G_sobel(image)
        img=self.model_ggg(li)
        return img
          
    def upload_sobell(self):
        self.file_name=self.upload_()
        if self.file_name:
            original_image = cv2.imread(self.file_name)
            if original_image is not None:
                gray_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
                _, binary_image = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                self.display_image(binary_image, self.before_sobel)
    def apply_sobell(self):  
           print(self.file_name)
           if self.file_name:
               original_image = cv2.imread(self.file_name)
               if original_image is not None:
                          value = self.sobel_th.value()
                          gray_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
                          im = self.sobell(gray_image)
                          if self.checkbox_state_sobel==False:  
                                ret, im_binary_image = cv2.threshold(im, value, 255, cv2.THRESH_BINARY)
                                self.display_image(im_binary_image, self.after_prewiit)
                          else:
                                _, im_binary_image = cv2.threshold(im, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                                self.display_image(im_binary_image, self.after_sobel)
                              
                          
    def histtt_sobell(self):
           print(self.file_name)
           if self.file_name:
               original_image = cv2.imread(self.file_name)
               if original_image is not None:
                          value = self.sobel_th.value()
                          gray_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
                          im = self.sobell(gray_image)
                          self.show_histogram(im)
    def save_sobell(self):
           print(self.file_name)
           if self.file_name:
               original_image = cv2.imread(self.file_name)
               if original_image is not None:
                          value = self.sobel_th.value()
                          gray_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
                          im = self.sobell(gray_image)
                          if self.checkbox_state_sobel==False:  
                                ret, im_binary_image = cv2.threshold(im, value, 255, cv2.THRESH_BINARY)
                                self.save_image(im_binary_image)
                          else:
                                _, im_binary_image = cv2.threshold(im, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                                self.save_image(im_binary_image)

                
    def mas(self):
                mask_no=[[1,1,0],
                [1,0,-1],
                [0,-1,-1]]

                mask_o=[[1,0,-1],
                [1,0,-1],
                [1,0,-1]]

                mask_so=[[0,-1,-1],
                        [1,0,-1],
                        [1,1,0]]
                mask_s=[[-1,-1,-1],
                        [0,0,0],
                        [1,1,1]]

                mask_se=[[-1,-1,0],
                [-1,0,1],
                [0,1,1]]

                mask_E=[[-1,0,1],
                        [-1,0,1],
                        [-1,0,1]]
                mask_NE=[[0,1,1],
                        [-1,0,1],
                        [-1,-1,0]]
                mask_s=[[1,1,1],
                        [0,0,0],
                        [-1,-1,-1]]
                masks =[mask_no,mask_o,mask_so,mask_s,mask_se,mask_E,mask_NE,mask_s]
                return masks
                
    def G(self,image,masks):
                m,n = image.shape[0],image.shape[1]
                G=[]
                for mask in masks:
                        Gy=np.zeros_like(image)
                        for i in range(1,m-1):
                           for j in range(1,n-1):
                                part= image[i-1:i+1+1,j-1:j+1+1]
                                k=mask*part
                                Gy[i,j]=abs(np.sum(k))
                        G.append(Gy)       
                return G  

    def  Robinsonn(self, image):
            masks=self.mas()
            ima_masks = self.G(image,masks)
            i=self.model_ggg(ima_masks)
            return i
    def upload_Robinson(self):
        self.file_name=self.upload_()
        if self.file_name:
            original_image = cv2.imread(self.file_name)
            if original_image is not None:
                gray_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
                _, binary_image = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                self.display_image(binary_image, self.before_robinson)
                
    def apply_Robinson(self):
           print(self.file_name)
           if self.file_name:
               original_image = cv2.imread(self.file_name)
               if original_image is not None:
                          value = self.robinson_th.value()
                          gray_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
                          im = self.Robinsonn(gray_image)
                          if self.checkbox_state_robinson==False:  
                                  ret, im_binary_image = cv2.threshold(im, value, 255, cv2.THRESH_BINARY)
                                  self.display_image(im_binary_image, self.after_robinson)
                          else:
                                _, im_binary_image = cv2.threshold(im, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                                self.display_image(im_binary_image, self.after_robinson)                         
                          
    def histtt_Robinson(self):
           print(self.file_name)
           if self.file_name:
               original_image = cv2.imread(self.file_name)
               if original_image is not None:
                          value = self.robinson_th.value()
                          gray_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
                          im = self.Robinsonn(gray_image)
                          self.show_histogram(im)

    def save_Robinson(self):
           if self.file_name:
               original_image = cv2.imread(self.file_name)
               if original_image is not None:
                          value = self.robinson_th.value()
                          gray_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
                          im = self.Robinsonn(gray_image)
                          if self.checkbox_state_robinson==False:  
                                  ret, im_binary_image = cv2.threshold(im, value, 255, cv2.THRESH_BINARY)
                                  self.save_image(im_binary_image)

                          else:
                                _, im_binary_image = cv2.threshold(im, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                                self.save_image(im_binary_image)
   
   
   
    def G_roberts(self,image):
                                        mask1=[[1,0],
                                                [0,-1]]
                                        mask2=[[-1,0],
                                                [0,1]]
                                        m,n = image.shape[0],image.shape[1]
                                        image =np.float32(image)
                                        Gx=np.zeros_like(image)
                                        Gy=np.zeros_like(image)
                                        for i in range(1,m-1):
                                                for j in range(1,n-1):
                                                        part= image[i-1:i+1,j-1:j+1]
                                                        Gx[i,j]=abs(np.sum(part*mask1))
                                                        Gy[i,j]=abs(np.sum(part*mask2))             
                                        return [Gx,Gy]
    def robertss(self,image):
        li=self.G_roberts(image)
        img=self.model_ggg(li)
        return img
    def upload_roberts(self):
        self.file_name=self.upload_()
        if self.file_name:
            original_image = cv2.imread(self.file_name)
            if original_image is not None:
                gray_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
                _, binary_image = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                self.display_image(binary_image, self.brfore_robert)
                    
    def apply_roberts(self):
           print(self.file_name)
           if self.file_name:
               original_image = cv2.imread(self.file_name)
               if original_image is not None:
                          value = self.roberts_th.value()
                          gray_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
                          im = self.robertss(gray_image)
                          if self.checkbox_state_robert==False:
                                    ret, im_binary_image = cv2.threshold(im, value, 255, cv2.THRESH_BINARY)
                                    self.display_image(im_binary_image, self.after_robert)
                          else: 
                                _, im_binary_image = cv2.threshold(im, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                                self.display_image(im_binary_image, self.after_robert)
                              
    def hisst_roberts(self):
           print(self.file_name)
           if self.file_name:
               original_image = cv2.imread(self.file_name)
               if original_image is not None:
                          value = self.roberts_th.value()
                          gray_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
                          im = self.robertss(gray_image)
                          self.show_histogram(im)
    def save_roberts(self):
           print(self.file_name)
           if self.file_name:
               original_image = cv2.imread(self.file_name)
               if original_image is not None:
                          value = self.roberts_th.value()
                          gray_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
                          im = self.robertss(gray_image)
                          if self.checkbox_state_robert==False:
                                    ret, im_binary_image = cv2.threshold(im, value, 255, cv2.THRESH_BINARY)
                                    self.save_image(im_binary_image)
                          else: 
                                _, im_binary_image = cv2.threshold(im, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                                self.save_image(im_binary_image)


    def G_Laplacien(self,image):
        mask=[[0,-1,0],
                [-1,4,-1],
                [0,-1,0]]
        m,n = image.shape[0],image.shape[1]
        res=np.zeros_like(image)
        for i in range(1,m-1):
                for j in range(1,n-1):
                        part= image[i-1:i+1+1,j-1:j+1+1]
                        res[i,j]=abs(np.sum(part*mask))
        return res


    def Laplacienn(self,image):
        imgg=self.G_Laplacien(image)
        return imgg


    def upload_Laplacien(self):
        self.file_name=self.upload_()
        if self.file_name:
            original_image = cv2.imread(self.file_name)
            if original_image is not None:
                gray_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
                _, binary_image = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                self.display_image(binary_image, self.before_laplacien)
    
                
    def apply_Laplacienn(self):
           print(self.file_name)
           if self.file_name:
               original_image = cv2.imread(self.file_name)
               if original_image is not None:
                          value = self.laplacien_th.value()
                          gray_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
                          im = self.Laplacienn(gray_image)

                          if self.checkbox_state_laplacien==False:  
                                  ret, im_binary_image = cv2.threshold(im, value, 255, cv2.THRESH_BINARY)
                                  self.display_image(im_binary_image, self.after_laplacien)
                          else:
                                _, im_binary_image = cv2.threshold(im, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                                self.display_image(im_binary_image, self.after_laplacien)  
    def hisst_Laplacienn(self):
           print(self.file_name)
           if self.file_name:
               original_image = cv2.imread(self.file_name)
               if original_image is not None:
                          value = self.laplacien_th.value()
                          gray_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
                          im = self.Laplacienn(gray_image)
                          self.show_histogram(im)
                          
                          
    def save_Laplacienn(self):
           print(self.file_name)
           if self.file_name:
               original_image = cv2.imread(self.file_name)
               if original_image is not None:
                          value = self.laplacien_th.value()
                          gray_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
                          im = self.Laplacienn(gray_image)
                          if self.checkbox_state_laplacien==False:  
                                  ret, im_binary_image = cv2.threshold(im, value, 255, cv2.THRESH_BINARY)
                                  self.save_image(im_binary_image)
                          else:
                                _, im_binary_image = cv2.threshold(im, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                                self.save_image(im_binary_image)




           
######################################################################hough#######################################
    def upload_hough_lines(self):
        self.file_name=self.upload_()
        if self.file_name:
            original_image = cv2.imread(self.file_name)
            if original_image is not None:
                gray_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
                _, binary_image = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                self.display_image(binary_image, self.beforehoughlines)
                
    def apply_hough_lines(self):
           print(self.file_name)
           if self.file_name:
               original_image = cv2.imread(self.file_name)
               if original_image is not None:
                          value = self.th_houglines.value()
                          im =self.hough_liness(original_image,value)
                          self.display_image(im, self.after_hough_lines)
    def save_hough_lines(self):
           if self.file_name:
               original_image = cv2.imread(self.file_name)
               if original_image is not None:
                          value = self.th_houglines.value()
                          im =self.hough_liness(original_image,value)
                          self.save_image(im)
     
  
    def hough_liness (self, image ,thershhold):
            gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) 
            gray_image=255-gray_image
            _, binary_image = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            lines = cv2.HoughLines(binary_image, 2, np.pi / 180,thershhold)
            if lines is not None:
                for line in lines:
                    rho, theta = line[0]
                    a = np.cos(theta)
                    b = np.sin(theta)
                    x0 = a * rho
                    y0 = b * rho
                    x1 = int(x0 + 1000 * (-b))
                    y1 = int(y0 + 1000 * (a))
                    x2 = int(x0 - 1000 * (-b))
                    y2 = int(y0 - 1000 * (a))
                    cv2.line(image, (x1, y1), (x2, y2), (255, 0, 255), 1)
            return  image




    def upload_hough_circeles(self):
        self.file_name=self.upload_()
        if self.file_name:
            original_image = cv2.imread(self.file_name)
            if original_image is not None:
                self.display_image(original_image, self.beforehoughcircles)
                
    def apply_hough_circeles(self):
           print(self.file_name)
           if self.file_name:
               original_image = cv2.imread(self.file_name)
               if original_image is not None:
                          im =self.hough_circeless(original_image)
                          self.display_image(im, self.after_hough_circles)
    def save_hough_circeles(self):
           if self.file_name:
               original_image = cv2.imread(self.file_name)
               if original_image is not None:
                          value = self.th_houglines.value()
                          im =self.hough_circeless(original_image)
                          self.save_image(im)
     
  
    def hough_circeless (self, image):
            gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) 
            gray_image=255-gray_image
            circles = cv2.HoughCircles(gray_image, cv2.HOUGH_GRADIENT, 1, 20,
                               param1=20, param2=60,
                               minRadius=40, maxRadius=200)
            if circles is not None:
                      circles = np.uint16(np.around(circles))
                      for i in circles[0, :]:
                             center = (i[0], i[1])
                             cv2.circle(image, center, 1, (0, 100, 100), 3)
                             radius = i[2]
                             cv2.circle(image, center, radius, (255, 0, 255), 3)
            return  image
        
        
        
        
    def upload_watershed(self):
        self.file_name=self.upload_()
        if self.file_name:
            original_image = cv2.imread(self.file_name)
            if original_image is not None:
                original_image_rgb = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
                self.display_image(original_image_rgb, self.beffore_Watershed)
                
    def apply_watershed(self):
           print(self.file_name)
           if self.file_name:
               original_image = cv2.imread(self.file_name)
               if original_image is not None:
                          original_image_rgb = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
                      
                          im =self.apply_watershedd(original_image_rgb)
                          self.display_image(im, self.after_Watershed)
                          
    
    def apply_watershedd(self,image):
            # Convert the image to grayscale
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            # Apply Otsu's thresholding
            _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

            # Noise removal using morphological opening
            kernel = np.ones((3, 3), np.uint8)
            opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)

            # Sure background area
            sure_bg = cv2.dilate(opening, kernel, iterations=3)

            # Finding sure foreground area using distance transform
            dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
            ret, sure_fg = cv2.threshold(dist_transform, 0.7 * dist_transform.max(), 255, 0)
            sure_fg = np.uint8(sure_fg)

            # Finding unknown region
            unknown = cv2.subtract(sure_bg, sure_fg)

            # Marker labelling
            ret, markers = cv2.connectedComponents(sure_fg)

            # Add one to all labels so that sure background is not 0, but 1
            markers = markers + 1

            # Now, mark the region of unknown with zero
            markers[unknown == 255] = 0

            # Apply Watershed algorithm
            markers = cv2.watershed(image, markers)

            # Mark Watershed region in the original image
            image[markers == -1] = [255, 0, 0]

            return image
    
    def upload_grabcut(self):
        self.file_name=self.upload_()
        if self.file_name:
            original_image = cv2.imread(self.file_name)
            if original_image is not None:
                original_image_rgb = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
                self.display_image(original_image_rgb, self.beffore_GrabCut)
                
    def apply_grabcut(self):
           print(self.file_name)
           if self.file_name:
               original_image = cv2.imread(self.file_name)
               if original_image is not None:
                          im =self.grabcut_segmentation(original_image)
                          self.display_image(im, self.after_GrabCut)
    def grabcut_segmentation(self,image):
            # Read the image
            # Create a mask initialized with zeros
            mask = np.zeros(image.shape[:2], np.uint8)
            
            # Create foreground and background models
            bgd_model = np.zeros((1, 65), np.float64)
            fgd_model = np.zeros((1, 65), np.float64)
            
            # Define a rectangle around the object to help GrabCut segment the image
            rect = (50, 50, image.shape[1]-50, image.shape[0]-50)
            
            # Apply GrabCut algorithm
            cv2.grabCut(image, mask, rect, bgd_model, fgd_model, 5, cv2.GC_INIT_WITH_RECT)
            
            # Modify the mask to create a binary mask for the foreground
            mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
            
            # Multiply the original image by the binary mask to keep the foreground
            segmented_image = image * mask2[:, :, np.newaxis]
            
            # Assign different colors to the segmented regions
            colored_image = np.zeros_like(image)
            colored_image[mask2 == 1] = np.random.randint(0, 255, 3)
            
            return colored_image

    def upload_meansshiftt(self):
        self.file_name=self.upload_()
        if self.file_name:
            original_image = cv2.imread(self.file_name)
            if original_image is not None:
                original_image_rgb = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
                self.display_image(original_image_rgb, self.beffore_meanshift)
                
    def apply_meanshifftt(self):
           print(self.file_name)
           if self.file_name:
               original_image = cv2.imread(self.file_name)
               if original_image is not None:
                          original_image_rgb = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
                          im =self.mean_shift_segmentation(original_image_rgb)
                          self.display_image(im, self.after_meanshift)
                          
                          
    def mean_shift_segmentation(self,image):

        shifted = cv2.pyrMeanShiftFiltering(image, 21, 51)
        return  shifted




    def upload_growing(self):
        self.file_name=self.upload_()
        if self.file_name:
            original_image = cv2.imread(self.file_name)
            if original_image is not None:
                original_image_rgb = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
                self.display_image(original_image_rgb, self.beffore_region)
                
    def apply_growing(self):
           print(self.file_name)
           if self.file_name:
               original_image = cv2.imread(self.file_name)
               if original_image is not None:
                          gray_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
                          im =self.region_growingg(gray_image)
                          self.display_image(im, self.after_region)
                          
    def region_growingg(self,image, seed = (150, 150)):
                        # Create an empty mask to store the segmented region
                        mask = np.zeros_like(image, dtype=np.uint8)

                        # Parameters for region growing (adjust as needed)
                        tolerance = 50 # Intensity difference threshold
                        region_size = 120  # Minimum region size
                        height, width = image.shape[:2]

                        # Queue to store pixel coordinates to be processed
                        queue = [seed]

                        # Seed pixel intensity
                        seed_intensity = image[seed]

                        while queue:
                                # Get the next pixel coordinates from the queue
                                current_pixel = queue.pop(0)
                                x, y = current_pixel

                                # Check if the pixel is within the image bounds
                                if 0 <= x < height and 0 <= y < width:
                                # Check if the pixel is unprocessed and similar to the seed
                                   if mask[x, y] == 0 and abs(int(image[x, y]) - int(seed_intensity)) < tolerance:
                                        # Add the pixel to the segmented region
                                        mask[x, y] = 255

                                        queue.append((x + 1, y))
                                        queue.append((x - 1, y))
                                        queue.append((x, y + 1))
                                        queue.append((x, y - 1))

                        return mask


       
    def open_color_segmentation_page(self):
        # Open the ColorSegmentationApp as a separate window
        self.color_segmentation_app = ColorSegmentationApp()
        self.color_segmentation_app.show()

    def upload_(self):
        options = QFileDialog.Options()
        options |= QFileDialog.ReadOnly
        file_name, _ = QFileDialog.getOpenFileName(self, "Open Image File", "", "Image Files (*.png *.jpg *.bmp *.jpeg);;All Files (*)", options=options)

        if file_name:
              return file_name
    def save_image(self, image):
        if image is not None:
            options = QFileDialog.Options()
            options |= QFileDialog.DontUseNativeDialog
            file_name, _ = QFileDialog.getSaveFileName(self, "Save Image", "", "Images (*.png *.xpm *.jpg *.bmp);;All Files (*)", options=options)

            if file_name:
                if not file_name.endswith(('.png', '.xpm', '.jpg', '.bmp')):
                         file_name += '.png'
                cv2.imwrite(file_name, image)


