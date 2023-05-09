# encoding:utf-8
import pyautogui

# 获取本机分辨率
print("本台计算机分辨率为：",pyautogui.size(),type(pyautogui.size()))     # Size(width=1920, height=1080) <class 'pyautogui.Size'>
 
# 获取当前鼠标位置
x, y = pyautogui.position()
print("目前光标的位置：",pyautogui.position(),type(pyautogui.position()))       # Point(x=842, y=346) <class 'pyautogui.Point'>


def move_mouse_to(x, y, duration=0):
    """
    将鼠标移动到指定位置
    
    参数:
    x: int, 屏幕上的x坐标
    y: int, 屏幕上的y坐标
    duration: float, 鼠标移动的持续时间（秒）
    """
    pyautogui.moveTo(x, y, duration=duration)
    
def click_left_button():
    """
    点击鼠标左键
    """
    pyautogui.click()