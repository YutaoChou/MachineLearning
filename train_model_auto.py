# encoding:utf-8
import os
import time
import cv2
import numpy as np
import tensorflow as tf
from tkinter import Tk, filedialog
from screen_capture import capture_screen
from keras.models import load_model
#导入配置文件
from config import model_choice, input_shape, learning_rate, loss_function, epochs, model_file, images_folder
from memory_check import MemoryCheck
from image_processing_ml import preprocess_image

def preprocess_imageT(image):
    return cv2.resize(image, (input_shape[0], input_shape[1]))

def find_reference_object(model, image):
    processed_image = preprocess_imageT(image)
    prediction = model.predict(np.expand_dims(processed_image, axis=0))[0]
    x1, y1, x2, y2 = prediction
    return int(x1), int(y1), int(x2), int(y2)

def find_target(image, template, threshold=0.7):
    """
    在图像中寻找参考物
    
    参数:
    image: numpy array, 输入的图像
    template: numpy array, 参考物的模板图像
    threshold: float, 相似度阈值
    
    返回值:
    (x, y): tuple, 参考物在图像中的位置
    """
    result = cv2.matchTemplate(image, template, cv2.TM_CCOEFF_NORMED)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
    x, y = max_loc
    if max_val > threshold:
        return x,y,x+input_shape[0],y+input_shape[1]
    else:
        return None

def update_model(model, model_file, new_image, new_annotation):
    processed_image = preprocess_imageT(new_image)
    model.fit(np.expand_dims(processed_image, axis=0),
              np.expand_dims(new_annotation, axis=0),
              epochs=1, batch_size=1,
              callbacks=[MemoryCheck()])
    model.save(model_file)

def main(model_file='model.h5'):
    runTimes = 1000
    cuurrT = 0
    if os.path.exists(model_file):
        model = load_model(model_file)
    else:
        print("模型文件未找到，请先运行 train_model.py")
        return
    template = cv2.imread("Target\T1.png", cv2.IMREAD_GRAYSCALE)

    root = Tk()
    root.withdraw()
    images_folder = 'Images'

    image_files = [f for f in os.listdir(images_folder) if f.endswith(('.png', '.jpg', '.jpeg'))]

    if not image_files:
        print("所选文件夹没有图片")
        return
    
    while True:
        # 获取屏幕截图
        #screen = capture_screen()
        # 读取文件夹图片

        for image_file in image_files:
            image_path = os.path.join(images_folder, image_file)
            screen = cv2.imread(image_path)
            
            # 使用模型在屏幕截图中查找参考物
            x1, y1, x2, y2 = find_reference_object(model, screen)
            cv2.rectangle(screen, (x1, y1), (x2, y2), (0, 255, 0), 2)

            print("预测位置 X1: " + str(x1) + " Y1: " + str(y1) + " X2: " + str(x2) + " X2: " + str(y2))

            # 获取实际参照物位置
            gray = cv2.cvtColor(screen, cv2.COLOR_BGR2GRAY)
            blur_size=5
            blurred = cv2.GaussianBlur(gray, (blur_size, blur_size), 0)
            pos = find_target(blurred,template)
            if pos is None:
                print("Find Nothing")
                continue
            target_x1, target_y1, target_x2, target_y2 = find_target(blurred,template)
            cv2.rectangle(screen, (target_x1, target_y1), (target_x2, target_y2), (0, 0, 255), 2)

            print("真实位置 X1: " + str(target_x1) + " Y1: " + str(target_y1) + " X2: " + str(target_x2) + " X2: " + str(target_y2))

            # 使用实际参照物位置更新模型
            update_model(model, model_file, screen, [target_x1, target_y1, target_x2, target_y2])

            '''
            cuurrT+=1
            if cuurrT >= runTimes:
                os._exit(0)
            '''
            #'''
            # 显示结果
            cv2.imshow(u"自动训练", screen)
            while True:
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    cv2.destroyAllWindows()
                    os._exit(0)
                elif key == ord('c'):
                    cv2.destroyAllWindows()
                    break
            #'''   
if __name__ == '__main__':
    main()