# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'UiMain.ui'
#
# Created by: PyQt5 UI code generator 5.15.2
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(1320, 830)
        MainWindow.setMinimumSize(QtCore.QSize(1320, 830))
        MainWindow.setMaximumSize(QtCore.QSize(1320, 830))
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.frame = QtWidgets.QFrame(self.centralwidget)
        self.frame.setGeometry(QtCore.QRect(10, 100, 791, 711))
        self.frame.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame.setObjectName("frame")
        self.frame_2 = QtWidgets.QFrame(self.frame)
        self.frame_2.setGeometry(QtCore.QRect(10, 0, 771, 481))
        self.frame_2.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame_2.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_2.setObjectName("frame_2")
        self.label_show = QtWidgets.QLabel(self.frame_2)
        self.label_show.setGeometry(QtCore.QRect(0, 0, 770, 480))
        self.label_show.setMinimumSize(QtCore.QSize(770, 480))
        self.label_show.setMaximumSize(QtCore.QSize(770, 480))
        self.label_show.setStyleSheet("border-image: url(:/icons/ui_imgs/icons/violence.png)0 0 0 0 round;\n"
"background-color: transparent;")
        self.label_show.setText("")
        self.label_show.setObjectName("label_show")
        self.frame_3 = QtWidgets.QFrame(self.frame)
        self.frame_3.setGeometry(QtCore.QRect(10, 480, 771, 221))
        self.frame_3.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame_3.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_3.setObjectName("frame_3")
        self.groupBox_3 = QtWidgets.QGroupBox(self.frame_3)
        self.groupBox_3.setGeometry(QtCore.QRect(0, 10, 771, 221))
        font = QtGui.QFont()
        font.setFamily("微软雅黑 Light")
        font.setPointSize(14)
        self.groupBox_3.setFont(font)
        self.groupBox_3.setStyleSheet("border: 2px solid #ADD8E6;\n"
"border-radius: 10px;\n"
"")
        self.groupBox_3.setObjectName("groupBox_3")
        self.tableWidget = QtWidgets.QTableWidget(self.groupBox_3)
        self.tableWidget.setGeometry(QtCore.QRect(10, 30, 751, 181))
        font = QtGui.QFont()
        font.setFamily("华文楷体")
        font.setPointSize(14)
        self.tableWidget.setFont(font)
        self.tableWidget.setStyleSheet("")
        self.tableWidget.setObjectName("tableWidget")
        self.tableWidget.setColumnCount(5)
        self.tableWidget.setRowCount(0)
        item = QtWidgets.QTableWidgetItem()
        self.tableWidget.setHorizontalHeaderItem(0, item)
        item = QtWidgets.QTableWidgetItem()
        self.tableWidget.setHorizontalHeaderItem(1, item)
        item = QtWidgets.QTableWidgetItem()
        self.tableWidget.setHorizontalHeaderItem(2, item)
        item = QtWidgets.QTableWidgetItem()
        self.tableWidget.setHorizontalHeaderItem(3, item)
        item = QtWidgets.QTableWidgetItem()
        self.tableWidget.setHorizontalHeaderItem(4, item)
        self.frame_4 = QtWidgets.QFrame(self.centralwidget)
        self.frame_4.setGeometry(QtCore.QRect(810, 100, 431, 711))
        self.frame_4.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame_4.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_4.setObjectName("frame_4")
        self.groupBox = QtWidgets.QGroupBox(self.frame_4)
        self.groupBox.setGeometry(QtCore.QRect(0, 0, 431, 171))
        font = QtGui.QFont()
        font.setFamily("仿宋_GB2312")
        font.setPointSize(14)
        self.groupBox.setFont(font)
        self.groupBox.setStyleSheet("border: 2px solid #ADD8E6;\n"
"border-radius: 10px;\n"
"")
        self.groupBox.setTitle("")
        self.groupBox.setObjectName("groupBox")
        self.PiclineEdit = QtWidgets.QLineEdit(self.groupBox)
        self.PiclineEdit.setGeometry(QtCore.QRect(11, 6, 400, 40))
        self.PiclineEdit.setMinimumSize(QtCore.QSize(400, 40))
        self.PiclineEdit.setMaximumSize(QtCore.QSize(400, 40))
        font = QtGui.QFont()
        font.setPointSize(10)
        self.PiclineEdit.setFont(font)
        self.PiclineEdit.setStyleSheet("border-radius: 1px;")
        self.PiclineEdit.setInputMask("")
        self.PiclineEdit.setObjectName("PiclineEdit")
        self.VideolineEdit = QtWidgets.QLineEdit(self.groupBox)
        self.VideolineEdit.setGeometry(QtCore.QRect(11, 66, 400, 40))
        self.VideolineEdit.setMinimumSize(QtCore.QSize(400, 40))
        self.VideolineEdit.setMaximumSize(QtCore.QSize(400, 40))
        font = QtGui.QFont()
        font.setPointSize(10)
        self.VideolineEdit.setFont(font)
        self.VideolineEdit.setStyleSheet("border-radius: 1px;")
        self.VideolineEdit.setObjectName("VideolineEdit")
        self.CaplineEdit = QtWidgets.QLineEdit(self.groupBox)
        self.CaplineEdit.setGeometry(QtCore.QRect(11, 126, 400, 40))
        self.CaplineEdit.setMinimumSize(QtCore.QSize(400, 40))
        self.CaplineEdit.setMaximumSize(QtCore.QSize(400, 40))
        font = QtGui.QFont()
        font.setPointSize(10)
        self.CaplineEdit.setFont(font)
        self.CaplineEdit.setStyleSheet("border-radius: 1px;")
        self.CaplineEdit.setObjectName("CaplineEdit")
        self.groupBox_2 = QtWidgets.QGroupBox(self.frame_4)
        self.groupBox_2.setGeometry(QtCore.QRect(0, 180, 431, 371))
        font = QtGui.QFont()
        font.setFamily("微软雅黑 Light")
        font.setPointSize(14)
        self.groupBox_2.setFont(font)
        self.groupBox_2.setObjectName("groupBox_2")
        self.layoutWidget = QtWidgets.QWidget(self.groupBox_2)
        self.layoutWidget.setGeometry(QtCore.QRect(10, 40, 281, 331))
        self.layoutWidget.setObjectName("layoutWidget")
        self.verticalLayout = QtWidgets.QVBoxLayout(self.layoutWidget)
        self.verticalLayout.setContentsMargins(0, 0, 0, 0)
        self.verticalLayout.setObjectName("verticalLayout")
        self.horizontalLayout_7 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_7.setObjectName("horizontalLayout_7")
        self.label_10 = QtWidgets.QLabel(self.layoutWidget)
        font = QtGui.QFont()
        font.setFamily("等线 Light")
        font.setPointSize(16)
        self.label_10.setFont(font)
        self.label_10.setObjectName("label_10")
        self.horizontalLayout_7.addWidget(self.label_10)
        self.time_lb = QtWidgets.QLabel(self.layoutWidget)
        palette = QtGui.QPalette()
        brush = QtGui.QBrush(QtGui.QColor(255, 0, 0))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Active, QtGui.QPalette.WindowText, brush)
        brush = QtGui.QBrush(QtGui.QColor(0, 0, 0))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Active, QtGui.QPalette.Text, brush)
        brush = QtGui.QBrush(QtGui.QColor(255, 0, 0, 128))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Active, QtGui.QPalette.PlaceholderText, brush)
        brush = QtGui.QBrush(QtGui.QColor(255, 0, 0))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Inactive, QtGui.QPalette.WindowText, brush)
        brush = QtGui.QBrush(QtGui.QColor(0, 0, 0))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Inactive, QtGui.QPalette.Text, brush)
        brush = QtGui.QBrush(QtGui.QColor(255, 0, 0, 128))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Inactive, QtGui.QPalette.PlaceholderText, brush)
        brush = QtGui.QBrush(QtGui.QColor(120, 120, 120))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Disabled, QtGui.QPalette.WindowText, brush)
        brush = QtGui.QBrush(QtGui.QColor(120, 120, 120))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Disabled, QtGui.QPalette.Text, brush)
        brush = QtGui.QBrush(QtGui.QColor(0, 0, 0, 128))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Disabled, QtGui.QPalette.PlaceholderText, brush)
        self.time_lb.setPalette(palette)
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(16)
        font.setBold(True)
        self.time_lb.setFont(font)
        self.time_lb.setText("")
        self.time_lb.setObjectName("time_lb")
        self.horizontalLayout_7.addWidget(self.time_lb)
        self.verticalLayout.addLayout(self.horizontalLayout_7)
        self.horizontalLayout_5 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_5.setObjectName("horizontalLayout_5")
        self.label = QtWidgets.QLabel(self.layoutWidget)
        font = QtGui.QFont()
        font.setFamily("等线 Light")
        font.setPointSize(16)
        self.label.setFont(font)
        self.label.setObjectName("label")
        self.horizontalLayout_5.addWidget(self.label)
        self.label_nums = QtWidgets.QLabel(self.layoutWidget)
        palette = QtGui.QPalette()
        brush = QtGui.QBrush(QtGui.QColor(255, 0, 0))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Active, QtGui.QPalette.WindowText, brush)
        brush = QtGui.QBrush(QtGui.QColor(0, 0, 0))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Active, QtGui.QPalette.Text, brush)
        brush = QtGui.QBrush(QtGui.QColor(255, 0, 0, 128))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Active, QtGui.QPalette.PlaceholderText, brush)
        brush = QtGui.QBrush(QtGui.QColor(255, 0, 0))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Inactive, QtGui.QPalette.WindowText, brush)
        brush = QtGui.QBrush(QtGui.QColor(0, 0, 0))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Inactive, QtGui.QPalette.Text, brush)
        brush = QtGui.QBrush(QtGui.QColor(255, 0, 0, 128))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Inactive, QtGui.QPalette.PlaceholderText, brush)
        brush = QtGui.QBrush(QtGui.QColor(120, 120, 120))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Disabled, QtGui.QPalette.WindowText, brush)
        brush = QtGui.QBrush(QtGui.QColor(120, 120, 120))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Disabled, QtGui.QPalette.Text, brush)
        brush = QtGui.QBrush(QtGui.QColor(0, 0, 0, 128))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Disabled, QtGui.QPalette.PlaceholderText, brush)
        self.label_nums.setPalette(palette)
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(16)
        font.setBold(True)
        self.label_nums.setFont(font)
        self.label_nums.setText("")
        self.label_nums.setObjectName("label_nums")
        self.horizontalLayout_5.addWidget(self.label_nums)
        self.verticalLayout.addLayout(self.horizontalLayout_5)
        self.horizontalLayout_8 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_8.setObjectName("horizontalLayout_8")
        self.label_11 = QtWidgets.QLabel(self.layoutWidget)
        font = QtGui.QFont()
        font.setFamily("等线 Light")
        font.setPointSize(16)
        self.label_11.setFont(font)
        self.label_11.setObjectName("label_11")
        self.horizontalLayout_8.addWidget(self.label_11)
        self.label_conf = QtWidgets.QLabel(self.layoutWidget)
        palette = QtGui.QPalette()
        brush = QtGui.QBrush(QtGui.QColor(255, 0, 0))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Active, QtGui.QPalette.WindowText, brush)
        brush = QtGui.QBrush(QtGui.QColor(255, 0, 0))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Inactive, QtGui.QPalette.WindowText, brush)
        brush = QtGui.QBrush(QtGui.QColor(120, 120, 120))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Disabled, QtGui.QPalette.WindowText, brush)
        self.label_conf.setPalette(palette)
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(16)
        font.setBold(True)
        self.label_conf.setFont(font)
        self.label_conf.setText("")
        self.label_conf.setObjectName("label_conf")
        self.horizontalLayout_8.addWidget(self.label_conf)
        self.verticalLayout.addLayout(self.horizontalLayout_8)
        self.horizontalLayout_9 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_9.setObjectName("horizontalLayout_9")
        self.label_13 = QtWidgets.QLabel(self.layoutWidget)
        self.label_13.setMaximumSize(QtCore.QSize(60, 16777215))
        font = QtGui.QFont()
        font.setFamily("等线 Light")
        font.setPointSize(16)
        self.label_13.setFont(font)
        self.label_13.setObjectName("label_13")
        self.horizontalLayout_9.addWidget(self.label_13)
        self.type_lb = QtWidgets.QLabel(self.layoutWidget)
        palette = QtGui.QPalette()
        brush = QtGui.QBrush(QtGui.QColor(255, 0, 0))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Active, QtGui.QPalette.WindowText, brush)
        brush = QtGui.QBrush(QtGui.QColor(255, 0, 0))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Inactive, QtGui.QPalette.WindowText, brush)
        brush = QtGui.QBrush(QtGui.QColor(120, 120, 120))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Disabled, QtGui.QPalette.WindowText, brush)
        self.type_lb.setPalette(palette)
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(16)
        font.setBold(True)
        self.type_lb.setFont(font)
        self.type_lb.setText("")
        self.type_lb.setObjectName("type_lb")
        self.horizontalLayout_9.addWidget(self.type_lb)
        self.verticalLayout.addLayout(self.horizontalLayout_9)
        self.groupBox_4 = QtWidgets.QGroupBox(self.frame_4)
        self.groupBox_4.setGeometry(QtCore.QRect(0, 560, 431, 141))
        font = QtGui.QFont()
        font.setFamily("微软雅黑 Light")
        font.setPointSize(14)
        self.groupBox_4.setFont(font)
        self.groupBox_4.setStyleSheet("border: 2px solid #ADD8E6;\n"
"border-image: url(:/icons/ui_imgs/icons/location.png);\n"
"border-radius: 10px;\n"
"")
        self.groupBox_4.setObjectName("groupBox_4")
        self.frame_5 = QtWidgets.QFrame(self.centralwidget)
        self.frame_5.setGeometry(QtCore.QRect(10, 10, 1231, 91))
        self.frame_5.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame_5.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_5.setObjectName("frame_5")
        self.label_3 = QtWidgets.QLabel(self.frame_5)
        self.label_3.setGeometry(QtCore.QRect(160, 10, 985, 55))
        font = QtGui.QFont()
        font.setFamily("微软雅黑")
        font.setPointSize(30)
        self.label_3.setFont(font)
        self.label_3.setObjectName("label_3")
        self.layoutWidget1 = QtWidgets.QWidget(self.centralwidget)
        self.layoutWidget1.setGeometry(QtCore.QRect(1250, 70, 52, 591))
        self.layoutWidget1.setObjectName("layoutWidget1")
        self.verticalLayout_2 = QtWidgets.QVBoxLayout(self.layoutWidget1)
        self.verticalLayout_2.setContentsMargins(0, 0, 0, 0)
        self.verticalLayout_2.setObjectName("verticalLayout_2")
        self.PicBtn = QtWidgets.QPushButton(self.layoutWidget1)
        self.PicBtn.setMinimumSize(QtCore.QSize(50, 50))
        self.PicBtn.setMaximumSize(QtCore.QSize(50, 50))
        self.PicBtn.setStyleSheet("border: 0px solid #ADD8E6; \n"
"border-radius: 0px;\n"
"border-image: url(:/icons/ui_imgs/icons/img.png);")
        self.PicBtn.setText("")
        self.PicBtn.setObjectName("PicBtn")
        self.verticalLayout_2.addWidget(self.PicBtn)
        self.VideoBtn = QtWidgets.QPushButton(self.layoutWidget1)
        self.VideoBtn.setMinimumSize(QtCore.QSize(50, 50))
        self.VideoBtn.setMaximumSize(QtCore.QSize(50, 50))
        self.VideoBtn.setStyleSheet("border-image: url(:/icons/ui_imgs/icons/video.png);")
        self.VideoBtn.setText("")
        self.VideoBtn.setObjectName("VideoBtn")
        self.verticalLayout_2.addWidget(self.VideoBtn)
        self.CapBtn = QtWidgets.QPushButton(self.layoutWidget1)
        self.CapBtn.setMinimumSize(QtCore.QSize(50, 50))
        self.CapBtn.setMaximumSize(QtCore.QSize(50, 50))
        self.CapBtn.setStyleSheet("border-image: url(:/icons/ui_imgs/icons/camera.png);")
        self.CapBtn.setText("")
        self.CapBtn.setObjectName("CapBtn")
        self.verticalLayout_2.addWidget(self.CapBtn)
        self.TabBtn = QtWidgets.QPushButton(self.layoutWidget1)
        self.TabBtn.setMinimumSize(QtCore.QSize(50, 50))
        self.TabBtn.setMaximumSize(QtCore.QSize(50, 50))
        self.TabBtn.setStyleSheet("background-color: transparent;")
        self.TabBtn.setText("")
        icon = QtGui.QIcon()
        icon.addPixmap(QtGui.QPixmap(":/icons/ui_imgs/icons/history.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.TabBtn.setIcon(icon)
        self.TabBtn.setIconSize(QtCore.QSize(50, 50))
        self.TabBtn.setObjectName("TabBtn")
        self.verticalLayout_2.addWidget(self.TabBtn)
        self.SaveBtn = QtWidgets.QPushButton(self.layoutWidget1)
        self.SaveBtn.setMinimumSize(QtCore.QSize(50, 50))
        self.SaveBtn.setMaximumSize(QtCore.QSize(50, 50))
        self.SaveBtn.setStyleSheet("border-image: url(:/icons/ui_imgs/icons/保存.png);\n"
"background-color: transparent;")
        self.SaveBtn.setText("")
        self.SaveBtn.setIconSize(QtCore.QSize(30, 30))
        self.SaveBtn.setObjectName("SaveBtn")
        self.verticalLayout_2.addWidget(self.SaveBtn)
        self.ExitBtn = QtWidgets.QPushButton(self.layoutWidget1)
        self.ExitBtn.setMinimumSize(QtCore.QSize(50, 50))
        self.ExitBtn.setMaximumSize(QtCore.QSize(50, 50))
        self.ExitBtn.setStyleSheet("border-image: url(:/icons/ui_imgs/icons/退出.png);\n"
"background-color: transparent;")
        self.ExitBtn.setText("")
        self.ExitBtn.setIconSize(QtCore.QSize(30, 30))
        self.ExitBtn.setObjectName("ExitBtn")
        self.verticalLayout_2.addWidget(self.ExitBtn)
        MainWindow.setCentralWidget(self.centralwidget)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        self.groupBox_3.setTitle(_translate("MainWindow", "存储信息情况"))
        item = self.tableWidget.horizontalHeaderItem(0)
        item.setText(_translate("MainWindow", "序号"))
        item = self.tableWidget.horizontalHeaderItem(1)
        item.setText(_translate("MainWindow", "文件路径"))
        item = self.tableWidget.horizontalHeaderItem(2)
        item.setText(_translate("MainWindow", "类别"))
        item = self.tableWidget.horizontalHeaderItem(3)
        item.setText(_translate("MainWindow", "置信度"))
        item = self.tableWidget.horizontalHeaderItem(4)
        item.setText(_translate("MainWindow", "坐标位置"))
        self.PiclineEdit.setPlaceholderText(_translate("MainWindow", "请选择图片文件"))
        self.VideolineEdit.setPlaceholderText(_translate("MainWindow", "请选择视频文件"))
        self.CaplineEdit.setPlaceholderText(_translate("MainWindow", "摄像头未开启"))
        self.groupBox_2.setTitle(_translate("MainWindow", "检测结果"))
        self.label_10.setText(_translate("MainWindow", "<html><head/><body><p><span style=\" font-weight:600;\">用时：</span></p></body></html>"))
        self.label.setText(_translate("MainWindow", "<html><head/><body><p><span style=\" font-weight:600;\">目标数目：</span></p></body></html>"))
        self.label_11.setText(_translate("MainWindow", "<html><head/><body><p><span style=\" font-weight:600;\">置信度：</span></p></body></html>"))
        self.label_13.setText(_translate("MainWindow", "<html><head/><body><p><span style=\" font-weight:600;\">类型：</span></p></body></html>"))
        self.groupBox_4.setTitle(_translate("MainWindow", "事发位置"))
        self.label_3.setText(_translate("MainWindow", "基于yolo11的校园暴力预警系统"))
import ui_sources_rc
