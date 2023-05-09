# encoding:utf-8
import os
import cv2
import numpy as np
from tkinter import Tk, filedialog
from screen_capture import capture_screen
from keras.models import load_model
from image_processing_ml import preprocess_image, find_target

refPt = []
cropping = False

def preprocess_imageT(image):
    return cv2.resize(image, (128, 128))

def click_and_crop(event, x, y, flags, param):
    global refPt, cropping, input_shape
    input_shape = (128, 128)
    if event == cv2.EVENT_LBUTTONDOWN:
        cropping = True
        half_width = input_shape[1] // 2
        half_height = input_shape[0] // 2
        x1 = max(0, x - half_width)
        y1 = max(0, y - half_height)
        x2 = min(param.shape[1], x + half_width)
        y2 = min(param.shape[0], y + half_height)
        refPt = [(x1, y1), (x2, y2)]

        cv2.rectangle(param, refPt[0], refPt[1], (0, 255, 0), 2)
        cv2.imshow("image", param)
		
def find_reference_object(model, image):
    processed_image = preprocess_imageT(image)
    prediction = model.predict(np.expand_dims(processed_image, axis=0))[0]
    x1, y1, x2, y2 = prediction
    return int(x1), int(y1), int(x2), int(y2)

def update_model(model, model_file, new_image, new_annotation):
    processed_image = preprocess_imageT(new_image)
    model.train_on_batch(np.expand_dims(processed_image, axis=0), np.expand_dims(new_annotation, axis=0))
    model.save(model_file)

def get_new_annotation(image, model, model_file):
    global refPt

    clone = image.copy()
    cv2.namedWindow("image")
    cv2.setMouseCallback("image", click_and_crop, clone)
    refPt = []
    while True:
        image_temp = image.copy()
        if len(refPt) == 2:
            cv2.rectangle(image_temp, refPt[0], refPt[1], (0, 255, 0), 2)

        cv2.imshow("image", image_temp)
        key = cv2.waitKey(1) & 0xFF

        if key == ord("c"):
            break
        elif key == ord("r"):
            refPt = []
        elif key == ord("q"):
            refPt = []
            cv2.destroyAllWindows()
            return "Exit"
            

    cv2.destroyAllWindows()

    if len(refPt) == 2:
        x1, y1 = refPt[0]
        x2, y2 = refPt[1]
        new_annotation = [x1, y1, x2, y2]
        print("更新模型")
        update_model(model, model_file, image, new_annotation)
        return new_annotation

    return None


def main(model_file='model.h5'):
    if os.path.exists(model_file):
        model = load_model(model_file)
    else:
        print("模型文件未找到，请先运行 train_model.py 生产模型")
        return

    root = Tk()
    root.withdraw()
    images_folder = filedialog.askdirectory(title="请选择图片文件夹")
    root.destroy()

    if not images_folder:
        print("未选择文件夹")
        return

    image_files = [f for f in os.listdir(images_folder) if f.endswith(('.png', '.jpg', '.jpeg'))]

    if not image_files:
        print("所选文件夹没有图片")
        return

    for image_file in image_files:
        image_path = os.path.join(images_folder, image_file)
        image = cv2.imread(image_path)

        new_annotation = get_new_annotation(image, model, model_file)
        if new_annotation is None:
            print("未标记参照物，跳过当前图片")
        else:
            if new_annotation == "Exit" :
                print("跳过手动更新模型")
                break
            else:
                print("已标记参照物并更新模型")
    print("所有图片已处理完毕")

    while True:
        screen = capture_screen()
        screenCopy =  np.copy(screen)
        x1, y1, x2, y2 = find_reference_object(model, screen)
        cv2.rectangle(screen, (x1, y1), (x2, y2), (0, 255, 0), 2)
        # 显示结果
        cv2.imshow(u"模拟训练", screen)

        while True:
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                cv2.destroyAllWindows()
                os._exit(0)
            elif key == ord('a'):
            # 人为地标记新的参考物
                new_annotation = get_new_annotation(screenCopy, model, model_file)
                if new_annotation:
                    print(u"已添加标记：", new_annotation)
                else:
                    print(u"未添加标记")
                cv2.destroyAllWindows()
                break
            elif key == ord('c'):
                cv2.destroyAllWindows()
                break

if __name__ == '__main__':
    main()