# encoding:utf-8
import cv2
import pickle
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

def preprocess_image(image, blur_size=5):
    """
    图像预处理：灰度化、模糊化
    
    参数:
    image: numpy array, 输入的图像
    blur_size: int, 高斯模糊的核大小
    
    返回值:
    numpy array, 预处理后的图像
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (blur_size, blur_size), 0)
    return blurred

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

    if max_val > threshold:
        return max_loc
    else:
        return None

def ml_model():
    """
    创建机器学习模型并训练
    
    返回值:
    model: sklearn model, 训练好的机器学习模型
    """
    # 加载数据集（具体数据集的加载方式取决于实际数据）
    X, y = load_data()

    # 数据预处理
    scaler = StandardScaler()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # 创建模型
    knn = KNeighborsClassifier(n_neighbors=5)
    model = make_pipeline(scaler, knn)
    
    # 训练模型
    model.fit(X_train, y_train)

    # 模型评估
    score = model.score
    score = model.score(X_test, y_test)
    print(f"模型准确率：{score:.2f}")
    
    return model

def predict_target_position(image, model):
    """
    使用机器学习模型预测参考物的位置
    
    参数:
    image: numpy array, 输入的图像
    model: sklearn model, 训练好的机器学习模型
    
    返回值:
    (x, y): tuple, 参考物在图像中的位置
    """
    features = extract_features(image)  # 提取图像特征
    position = model.predict([features])
    return position[0]

def extract_features(image, n_features=500):
    """
    提取图像的特征

    参数:
    image: numpy array, 输入的图像
    n_features: int, 提取特征的数量

    返回值:
    descs: numpy array, 提取的特征
    """
    orb = cv2.ORB_create(nfeatures=n_features)
    _, descs = orb.detectAndCompute(image, None)
    return descs

def load_data(images, labels):
    """
    加载数据集

    参数:
    images: list, 输入的图像列表
    labels: list, 对应的标签列表

    返回值:
    X: numpy array, 特征数组
    y: numpy array, 标签数组
    """
    X = []
    y = []

    for image, label in zip(images, labels):
        features = extract_features(image)
        if features is not None:
            X.append(features)
            y.append(label)

    return np.array(X), np.array(y)

def save_model(model, file_name='model.pkl'):
    """
    保存模型到文件

    参数:
    model: sklearn model, 训练好的机器学习模型
    file_name: str, 模型文件名
    """
    with open(file_name, 'wb') as f:
        pickle.dump(model, f)

def load_model(file_name='model.pkl'):
    """
    从文件加载模型

    参数:
    file_name: str, 模型文件名

    返回值:
    model: sklearn model, 加载的机器学习模型
    """
    with open(file_name, 'rb') as f:
        model = pickle.load(f)
    return model
