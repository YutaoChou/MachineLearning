# encoding:utf-8
import numpy as np
import cv2
import pyautogui
import time
from config import screen_choice

def capture_screen():
    # 隐藏当前窗口
    cv2.destroyAllWindows()
    time.sleep(0.5)  # 给系统足够的时间来隐藏窗口

    if screen_choice == 0:
        screenshot = pyautogui.screenshot()
    elif screen_choice == 1:
        screens = pyautogui.screens()
        if len(screens) > 1:
            screenshot = pyautogui.screenshot(screen=screens[1])
        else:
            print("只有一个屏幕，无法截取拓展屏幕。")
            screenshot = pyautogui.screenshot()
    else:
        print("无效的屏幕选择。请在 config.py 中设置正确的 screen_choice 值（0 或 1）。")
        return

    return cv2.cvtColor(np.array(screenshot), cv2.COLOR_RGB2BGR)