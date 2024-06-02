import cv2
import os
import time

# 创建保存图像的目录
save_dir = 'face\op_2'
dst_size = (256, 256)
os.makedirs(save_dir, exist_ok=True)

# 打开摄像头
cap = cv2.VideoCapture(0)
face_detector = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_alt2.xml")
while True:
    # 读取视频流
    ret, frame = cap.read()
    img = frame.copy()

    #级联检测
    faces = face_detector.detectMultiScale(frame, scaleFactor=1.05, minNeighbors=5, 
                                       minSize=(100, 100), maxSize=(500, 500))
    for x, y, width, height in faces:
        cv2.rectangle(frame, (x, y), (x+width, y+height), (0, 0, 255), 2, cv2.LINE_8, 0)
        print(height/width,end='\r')
    # 显示视频流
    cv2.imshow('Video', frame)
    
    # 检测按键按下事件
    key = cv2.waitKey(1)
    
    
    # 如果按下空格键（ASCII码为32）
    if key == 32:
        count = 0
        # 截取人脸区域
        for x, y, width, height in faces:
            face_region = img[y:y+height, x:x+width]
            face_region = cv2.resize(face_region,dst_size)
            # gray_face = cv2.cvtColor(face_region, cv2.COLOR_BGR2GRAY)
            # 生成文件名
            timestamp = time.strftime("%Y%m%d%H%M%S")
            img_name = os.path.join(save_dir, '{}_{}.png'.format(timestamp,count))
            # 保存图像
            cv2.imwrite(img_name, face_region)
            count+=1
            print("Image saved as", img_name)
    
    # 如果按下ESC键（ASCII码为27），则退出循环
    elif key == 27:
        break

# 释放摄像头并关闭所有窗口
cap.release()
cv2.destroyAllWindows()
