import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential # type: ignore
from tensorflow.keras import layers,models # type: ignore
import cv2
from PIL import Image, ImageDraw, ImageFont
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('agg')

 
#第一层卷积+批量归一化
def ConvCall(x,filtten,xx,yy,strides = (1,1)):
    x = layers.Conv2D(filtten,(xx,yy),strides=strides,padding='same')(x)
    x = layers.BatchNormalization()(x)
    return x
 
#残差模块
def ResNetblock(input,filtten,strides = (1,1)):
    x = ConvCall(input,filtten,3,3,strides=strides)
    x = layers.Activation("relu")(x)
 
    x = ConvCall(x,filtten,3,3,strides=(1,1))
    if strides != (1,1):
        residual = ConvCall(input,filtten,1,1,strides=strides)
    else:
        residual = input
 
    x = x + residual
    x = layers.Activation("relu")(x)
 
    return x
 
#前向传播
def forward(inputs,num_classes):
    x = ConvCall(inputs, 64, 3, 3, strides=(1, 1))
    x = layers.Activation('relu')(x)
 
    x = ResNetblock(x, 64, strides=(1, 1))
    x = ResNetblock(x, 64, strides=(1, 1))
 
    x = ResNetblock(x, 128, strides=(2, 2))
    x = ResNetblock(x, 128, strides=(1, 1))
 
    x = ResNetblock(x, 256, strides=(2, 2))
    x = ResNetblock(x, 256, strides=(1, 1))
 
    x = ResNetblock(x, 512, strides=(2, 2))
    x = ResNetblock(x, 512, strides=(1, 1))
    x = layers.GlobalAveragePooling2D()(x)  # 全局平均池化
    x = layers.Dense(256, activation='relu')(x)
    output = layers.Dense(num_classes, "softmax")(x)

    return output

#模型组建
def ResNet18(num_classes = 10):

    inputs = keras.Input((256,256,3))
    output = forward(inputs,num_classes)

    model = models.Model(inputs,output)

    model.compile(loss = keras.losses.CategoricalCrossentropy(),
                  optimizer=keras.optimizers.Adam(0.001),
                  metrics=['accuracy'])

    return model

#数据增强
def augment(datas:tf.data.Dataset,epochs=1):
    for i in range(epochs):
        datas = datas.concatenate(datas.map(augment_images))
    return datas

def augment_images(image, label):
    # image = tf.image.random_flip_left_right(image)
    image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
    # image = tf.image.rot90(image, k=tf.random.uniform(shape=[], minval=0, maxval=4, dtype=tf.int32))
    image = tf.image.random_brightness(image, max_delta=0.2)
    image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
    return image, label

#目录索引
def get_folder_names():
    folder_names = {}
    index = 0
    # 遍历目录
    for entry in os.scandir('./face'):
        if entry.is_dir():
            folder_names[index] = entry.name
            index += 1
    return folder_names

def cv2ImgAddText(img, text, left, top, textColor=(0, 255, 0), textSize=20):
    if (isinstance(img, np.ndarray)):  # 判断是否OpenCV图片类型
        img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    # 创建一个可以在给定图像上绘图的对象
    draw = ImageDraw.Draw(img)
    # 字体的格式
    fontStyle = ImageFont.truetype(
        "simsun.ttc", textSize, encoding="utf-8")
    # 绘制文本
    draw.text((left, top), text, textColor, font=fontStyle)
    # 转换回OpenCV格式
    return cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)

    
def save_history(history):
    # 创建一个新的 Matplotlib 图表
    plt.figure()

    # 绘制训练损失和验证损失
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')

    # 添加标题和标签
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    # 保存图表为图片
    plt.savefig('training_loss.png')

    # 创建另一个 Matplotlib 图表来绘制准确率
    plt.figure()

    # 绘制训练准确率和验证准确率
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')

    # 添加标题和标签
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    # 保存图表为图片
    plt.savefig('training_accuracy.png')

def train():
    batch_size = 32
    data_dir = 'face_train'

    subdirectories = next(os.walk(data_dir))[1]
    num_classes = len(subdirectories)

    #数据预处理
    train_ds = tf.keras.utils.image_dataset_from_directory(
    data_dir,
    validation_split=0.3,
    subset="training",
    seed=123,
    image_size=(256, 256),
    label_mode='categorical',
    batch_size=batch_size)

    val_ds = tf.keras.utils.image_dataset_from_directory(
    data_dir,
    validation_split=0.3,
    subset="validation",
    seed=123,
    image_size=(256, 256),
    label_mode='categorical',
    batch_size=batch_size)

    # train_ds = augment(train_ds,4)
    # val_ds = augment(val_ds,4)

    #模型实例化
    # if os.path.exists('face_model.keras'):
    #     print(num_classes)
    #     model = tf.keras.models.load_model('face_model.keras')
    #     model.summary()
    #     # 删除原先的输出层
    #     model.pop()
    #     # 添加新的输出层
    #     new_output_layer = layers.Dense(num_classes, activation='softmax',name='Dense_output')
    #     model.add(new_output_layer)

    #     model.summary()
    #     #冻结卷积基
    #     for layer in model.layers[:-2]:
    #         layer.trainable = False
    # else:
    model = Sequential([
            layers.Rescaling(1./255, input_shape=(256,256, 3)),
            layers.Conv2D(16, 3, padding='same', activation='relu'),
            layers.MaxPooling2D(),
            layers.Conv2D(32, 3, padding='same', activation='relu'),
            layers.MaxPooling2D(),
            layers.Conv2D(64, 3, padding='same', activation='relu'),
            layers.MaxPooling2D(),
            layers.Flatten(),
            layers.Dense(1024, activation='relu'),
            layers.Dense(256, activation='relu'),
            layers.Dense(num_classes,activation= "softmax")
            ])
    # model = ResNet18(num_classes)
    
    model.compile(loss = keras.losses.CategoricalCrossentropy(),
                optimizer=keras.optimizers.Adam(0.001),
                metrics=['accuracy'])

    
    
    history = model.fit(train_ds,validation_data=val_ds,epochs=5)
    save_history(history)

    model.save('face_model.keras')

    return model
def extract_features(model,image):
    image =  cv2.resize(image,(256,256))
    image = tf.expand_dims(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), axis=0)
    pre_label = model.predict(image)
    return pre_label

# 定义一个函数来构建特征向量库，将每张图的特征都添加到字典列表中
def build_feature_library(model, data_dir):
    feature_library = {}
    for person_dir in os.listdir(data_dir):
        person_path = os.path.join(data_dir, person_dir)
        if os.path.isdir(person_path):
            print(f"Processing images for {person_dir}")
            for file in os.listdir(person_path):
                if file.endswith(".png"):
                    image_path = os.path.join(person_path, file)
                    image = cv2.imread(image_path)
                    feature = extract_features(model, image)
                    if person_dir not in feature_library:
                        feature_library[person_dir] = [feature]
                    else:
                        feature_library[person_dir].append(feature)
    return feature_library

# 特征匹配
def match_feature(feature, feature_library, threshold):
    max_sim = -1
    matched_person = None
    for person, features in feature_library.items():
        for avg_feature in features:
            similarity = cosine_similarity(feature.reshape(1, -1), avg_feature.reshape(1, -1))
            if similarity > max_sim:
                max_sim = similarity
                matched_person = person
    if max_sim > threshold:
        return matched_person, max_sim
    else:
        return "Unknown", max_sim

#训练编码模型
# train()