# encoding:utf-8
import os
import cv2
import numpy as np
import tensorflow as tf
from imgaug.augmentables.bbs import BoundingBox, BoundingBoxesOnImage
from imgaug import augmenters as iaa
from keras.applications import VGG16
from keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense ,Dropout
from keras.models import Model, load_model
from keras.optimizers import Adam
#使用 TensorFlow 内置的 MobileNetV2
from keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input
#导入配置文件
from config import model_choice, input_shape, learning_rate, loss_function, epochs, model_file, images_folder
#pip install tensorflow opencv-python-headless imgaug

def create_model():
    if model_choice == 'VGG16':
        base_model = VGG16(weights='imagenet', include_top=False, input_shape=input_shape)
        x = base_model.output
        x = Flatten()(x)
        x = Dense(128, activation='relu')(x)
        outputs = Dense(4, activation='linear')(x)
        model = Model(inputs=base_model.input, outputs=outputs)
    elif model_choice == 'MobileNetV2':
        base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=input_shape)
        x = base_model.output
        x = Flatten()(x)
        x = Dense(512, activation='relu')(x)
        x = Dropout(0.5)(x)
        outputs = Dense(4, activation='linear')(x)
        model = Model(inputs=base_model.input, outputs=outputs)
    elif model_choice == 'Normal':
        inputs = Input(shape=input_shape)
        x = Conv2D(32, (3, 3), activation='relu')(inputs)
        x = MaxPooling2D(pool_size=(2, 2))(x)
        x = Conv2D(64, (3, 3), activation='relu')(x)
        x = MaxPooling2D(pool_size=(2, 2))(x)
        x = Flatten()(x)
        x = Dense(128, activation='relu')(x)
        outputs = Dense(4, activation='linear')(x)
        model = Model(inputs=inputs, outputs=outputs)
 # 使用config中的学习率和损失函数
    model.compile(optimizer=Adam(learning_rate=learning_rate), loss=loss_function)
    return model

def load_images_and_annotations(images_folder):
    images = []
    annotations = []

    for image_file in os.listdir(images_folder):
        if image_file.endswith(('.png', '.jpg', '.jpeg')):
            image_path = os.path.join(images_folder, image_file)
            annotation_path = os.path.splitext(image_path)[0] + '.txt'

            if os.path.exists(annotation_path):
                image = cv2.imread(image_path)
                images.append(image)

                with open(annotation_path, 'r') as f:
                    x1, y1, x2, y2 = map(float, f.readline().split())
                    annotations.append([x1, y1, x2, y2])

    return np.array(images), np.array(annotations)

def train(images_folder, model_file=model_file, epochs=epochs):
    model = create_model()
    images, annotations = load_images_and_annotations(images_folder)

    seq = iaa.Sequential([
        iaa.Resize({"height": input_shape[0], "width": input_shape[1]}),
    ])

    for epoch in range(epochs):
        for image, annotation in zip(images, annotations):
            bbs = BoundingBoxesOnImage([
                BoundingBox(x1=annotation[0], y1=annotation[1], x2=annotation[2], y2=annotation[3])
            ], shape=image.shape)

            image_aug, bbs_aug = seq(image=image, bounding_boxes=bbs)
            annotation_aug = [bbs_aug.bounding_boxes[0].x1, bbs_aug.bounding_boxes[0].y1, bbs_aug.bounding_boxes[0].x2, bbs_aug.bounding_boxes[0].y2]

            model.train_on_batch(np.expand_dims(image_aug, axis=0), np.expand_dims(annotation_aug, axis=0))

    model.save(model_file)

if __name__ == '__main__':
   train(images_folder)
