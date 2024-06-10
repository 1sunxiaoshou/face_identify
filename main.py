import sys
from PyQt5.QtWidgets import QMainWindow,QApplication,QWidget
from PyQt5 import QtCore
from PyQt5.QtGui import QIcon,QImage,QPixmap,QKeySequence
from PyQt5.QtWidgets import QMessageBox,QShortcut,QInputDialog
from PyQt5.QtCore import QTimer
from Ui_登录 import Ui_Form
from Ui_识别 import Ui_Form as Ui_Form2  #导入界面类
import model as md

import os
import cv2
import time
import pickle
import random
import threading
import tensorflow as tf
 
 
# user = '202101170329'
# password = '孙佳和'
user = '1'
password = '2'
model_path = 'face_model_5.keras'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # 设置TensorFlow日志级别为ERROR
threshold = 0.75


#登录窗口
class MyMainWindow(QWidget,Ui_Form):
    def __init__(self,parent =None):
        super(MyMainWindow,self).__init__(parent)
        self.setupUi(self)
        self.Method_bindings()
        self.setWindowIcon(QIcon('logo.ico'))

        self.label_2.setWordWrap(True)
    
        self.grate = False
        self.current_index=0
        self.talk = ['什么都不说吗？','笨蛋雏草姬！','再这样我可走咯？','卧槽，原！','你说得对，但是《原神》.......']
    
    #信号/槽绑定
    def Method_bindings(self):
        self.pushButton.clicked.connect(self.login_button)
        
    
    #文字刷新
    def show_next_letter(self,text,label):
        if self.current_index <= len(text):
            label.setText(text[:self.current_index])
            self.current_index += 1
            QTimer.singleShot(100, lambda: self.show_next_letter(text, label))
        else:
            self.current_index=0

    #身份认证
    def login_button(self):
        input = self.lineEdit.text()
        if input=='':
            self.show_next_letter(random.choice(self.talk),self.label_2)
        elif input == user:
            self.show_next_letter('请输入用户名',self.label_2)
            self.lineEdit.clear()
            self.grate = True
        elif input == password and self.grate:
            self.show_next_letter('欢迎回来，主人~',self.label_2)
            self.lineEdit.clear()
            self.open_window()
        else:
            self.show_next_letter('错了哦~',self.label_2)
            self.lineEdit.clear()
    #打开子窗口
    def open_window(self):
        myWin2.show()
        myWin2.load_model()
        self.close()


 #人脸识别窗口
class MyMainWindow2(QWidget,Ui_Form2):
    def __init__(self,parent =None):
        super(MyMainWindow2,self).__init__(parent)
        self.setupUi(self)
        self.Method_bindings()#信号槽
        self.setWindowIcon(QIcon('logo.ico'))

        self.is_train = False
        self.is_pred = False
        self.image = None
        self.faces = None
        self.save_dir = None
        self.dst_size = (256, 256)

        # 加载保存的特征向量库文件
        with open('feature_library.pkl', 'rb') as file:
            self.feature_library = pickle.load(file)


    #信号/槽绑定
    def Method_bindings(self):
        self.pushButton.clicked.connect(self.button)
        self.pushButton_2.clicked.connect(self.button_2)
        self.pushButton_4.clicked.connect(self.button_4)
        quit_shortcut = QShortcut(QKeySequence(' '), self)  # 绑定 "空格" 快捷键
        quit_shortcut.activated.connect(self.extract)
        
        self.bg = QPixmap('bg.jpg').scaled(self.label.size(), aspectRatioMode=False)



    def button(self):
        if self.is_train == False:
            self.pushButton.setStyleSheet("background-color: rgb(200, 0, 30);")
            self.pushButton.setText('结束')
            self.train_th()
        else:
            self.pushButton.setStyleSheet("background-color: rgb(255, 255, 255);")
            self.pushButton.setText('训练')
            self.end_train()
    
    def button_2(self):
        if self.is_pred == False:
            self.pushButton_2.setStyleSheet("background-color: rgb(200, 0, 30);")
            self.pushButton_2.setText('结束')
            self.start_predict()
        else:
            self.pushButton_2.setStyleSheet("background-color: rgb(255, 255, 255);")
            self.pushButton_2.setText('推理')
            self.end_predict()

    def button_4(self):
        # 建立特征向量库
        self.feature_library = md.build_feature_library(self.model,'face')
        print("Feature library built successfully!")
        # 保存特征向量库到文件
        with open('feature_library.pkl', 'wb') as file:
            pickle.dump(self.feature_library, file)
    #启动训练
    def train_th(self):
        self.pushButton_2.setEnabled(False)
        name, ok = QInputDialog.getText(self, '新建角色', '请输入角色名称：')
        if not ok:
            print('用户取消输入')
            return
        else:
            # 创建保存图像的目录
            self.save_dir = os.path.join('face', name)
            self.is_train = True
            if not os.path.exists(self.save_dir):
                os.makedirs(self.save_dir, exist_ok=True)
        added_thread = threading.Thread(target=self.train_model)  
        # 启动线程
        added_thread.start()
    #结束训练
    def end_train(self):
        self.pushButton_2.setEnabled(True)
        self.is_train = False
    
    #启动推理
    def start_predict(self):
        self.pushButton.setEnabled(False)
        self.pre_thread = threading.Thread(target=self.predict_model)  
        # 启动线程
        self.pre_thread.start()
    #结束推理
    def end_predict(self):
        self.pushButton.setEnabled(True)
        self.is_pred = False

    
    #视频帧刷新
    def video_updata(self,image):
            # 显示视频流
            height, width, depth = image.shape
            cvimg = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            cvimg = QImage(cvimg.data, width, height, width * depth, QImage.Format_RGB888)
            self.label.setPixmap(QPixmap(cvimg))
    
    #截取人脸
    def extract(self):
        if self.is_train and len(self.faces)==1:
                x, y, width, height = self.faces[0]
                face_region = self.image[y:y+height, x:x+width]
                face_region = cv2.resize(face_region,self.dst_size)
                # 生成文件名
                timestamp = time.strftime("%Y%m%d%H%M%S")
                img_name = os.path.join(self.save_dir, '{}.png'.format(timestamp))
                file_name = os.path.basename(self.save_dir)
                # 保存图像
                cv2.imwrite(img_name, face_region)
                print("保存位置：", img_name)
                feature = md.extract_features(self.model, face_region)
                if file_name not in self.feature_library:
                    self.feature_library[file_name] = [feature]
                else:
                    self.feature_library[file_name].append(feature)

    
    #初始化模型
    def load_model(self):
        try:
            self.model = tf.keras.models.load_model(model_path)
            self.model.pop()
        except:
            error_box = QMessageBox()
            error_box.setIcon(QMessageBox.Critical)
            error_box.setText("Error")
            error_box.setInformativeText('未能找到模型'+model_path)
            error_box.setWindowTitle("Error")
            error_box.exec_()
    
    #模型训练
    def train_model(self):

        # 打开摄像头
        cap = cv2.VideoCapture(0)
        face_detector = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_alt2.xml")

        while self.is_train:
            # 读取视频流
            _, frame = cap.read()
            self.image = frame.copy()

            #级联检测
            faces = face_detector.detectMultiScale(frame, scaleFactor=1.05, minNeighbors=5, 
                                            minSize=(100, 100), maxSize=(500, 500))
            self.faces = faces
            for x, y, width, height in faces:
                cv2.rectangle(frame, (x, y), (x+width, y+height), (0, 0, 255), 2, cv2.LINE_8, 0)
            self.video_updata(frame)
        # 释放摄像头并关闭所有窗口
        cap.release()
        self.label.setText('正在训练中')
        # 保存特征向量库到文件
        with open('feature_library.pkl', 'wb') as file:
            pickle.dump(self.feature_library, file)
        self.label.setPixmap(self.bg)



    #模型推理
    def predict_model(self):
        # 打开摄像头
        self.is_pred = True
        cap = cv2.VideoCapture(0)
        face_detector = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_alt2.xml")
        while self.is_pred:
            _,frame = cap.read()
            #级联检测
            faces = face_detector.detectMultiScale(frame, scaleFactor=1.05, minNeighbors=5, 
                                            minSize=(100, 100), maxSize=(500, 500))
            if len(faces) > 0:
                for x, y, width, height in faces:
                    cv2.rectangle(frame, (x, y), (x+width, y+height), (0, 0, 255), 2, cv2.LINE_8, 0)
                    face_region = frame[y:y+height, x:x+width]

                    # 进行特征匹配
                    test_feature = md.extract_features(self.model,face_region)
                    matched_person, similarity_score = md.match_feature(test_feature, self.feature_library, threshold)
                    frame = md.cv2ImgAddText(frame, str(matched_person), x+width//2-50, y+height//+20,textSize=40)
            self.video_updata(frame)
        cap.release()
        self.label.setPixmap(self.bg)
    

if __name__ == "__main__":
    QtCore.QCoreApplication.setAttribute(QtCore.Qt.AA_EnableHighDpiScaling)
    app = QApplication(sys.argv)
    myWin = MyMainWindow()
    myWin2 = MyMainWindow2()
    myWin.show()
    sys.exit(app.exec_())