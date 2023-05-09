# encoding:utf-8
import cv2
import pyautogui
from screen_capture import capture_screen
from image_processing_ml import preprocess_image, find_target, ml_model, predict_target_position, save_model, load_model
from mouse_control import move_mouse_to, click_left_button
import os
import time
from datetime import datetime

def save_screenshot(image, folder='Images', image_format='png'):
    """
    将抓取的图片存储到指定文件夹

    参数:
    image: numpy array, 输入的图像
    folder: str, 存储图片的文件夹名称
    image_format: str, 图片格式
    """
    if not os.path.exists(folder):
        os.makedirs(folder)

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    image_path = os.path.join(folder, f"{timestamp}.{image_format}")
    cv2.imwrite(image_path, image)


def main():
    # 加载多个参考物模板
    templates = [
        cv2.imread("Target\T1.png", cv2.IMREAD_GRAYSCALE),
        #cv2.imread("Target\template2.png", cv2.IMREAD_GRAYSCALE),
        # ...
    ]
    
    # 加载数据集
    #images = [cv2.imread(f"image{i}.png", cv2.IMREAD_GRAYSCALE) for i in range(1, 11)]
    #labels = [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]  # 示例标签，具体内容根据实际情况设置

    #model_file = 'model.pkl'
    #if os.path.exists(model_file):
        # 从本地加载模型
    #    model = load_model(model_file)
    #else:
        # 训练机器学习模型
    #    model = ml_model(images, labels)
        # 保存模型到本地
    #    save_model(model, model_file)

    i = 0

    # 如果数据集大小超过一个阈值，重新训练模型
    #if len(images) > 10:
    #    model = ml_model(images[-10:], labels[-10:])
        # 保存模型到本地
    #    save_model(model, model_file)

    while True:
        # 抓取屏幕画面
        screen = capture_screen()
        # 保存屏幕截图
        #save_screenshot(screen)
        # 预处理画面
        processed_screen = preprocess_image(screen)

        # 记录找到的所有参考物位置
        positions = []

        # 在画面中寻找所有参考物
        for template in templates:
            position = find_target(processed_screen, template)
            if position is not None:
                positions.append(position)

        # 如果未找到任何参考物，使用机器学习模型预测位置
        if not positions:
            print("Find Nothing")
            continue
        #    position = predict_target_position(processed_screen, model)
        #    positions.append(position)

        # 计算鼠标与参考物之间的距离
        mouse_x, mouse_y = pyautogui.position()
        distances = [((x - mouse_x) ** 2 + (y - mouse_y) ** 2) ** 0.5 for x, y in positions]

        # 找到距离最近的参考物
        min_distance_index = distances.index(min(distances))
        nearest_position = positions[min_distance_index]

        # 将鼠标移动到最近的参考物上并点击
        if nearest_position:
            x, y = nearest_position
            #print("MOUSE Move To X: " + str(x) + " Y: " + str(y))
            move_mouse_to(x + 40, y + 40)
            click_left_button()
        #time.sleep(0.1)  # 每秒抓取10张图片，间

if __name__ == "__main__":
    main()

