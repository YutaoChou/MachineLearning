# encoding:utf-8
import os
import subprocess
import tempfile
import cv2

def capture_android_screen():
    """
    使用 adb 工具截取安卓手机屏幕

    返回值:
    screen: numpy array, 截取的屏幕图像
    """
    with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f:
        tmp_file = f.name

    cmd = f"adb exec-out screencap -p > {tmp_file}"
    os.system(cmd)

    screen = cv2.imread(tmp_file, cv2.IMREAD_COLOR)
    os.remove(tmp_file)

    return screen

def move_android_touch(x, y):
    """
    使用 adb 工具在安卓手机上模拟触摸事件

    参数:
    x, y: int, 触摸事件的屏幕坐标 
    50 毫秒
    """
    cmd = f"adb shell input touchscreen swipe {x} {y} {x} {y} 50"
    os.system(cmd)
