# encoding:utf-8
# 屏幕选择（0：当前程序所在屏幕，1：拓展屏幕）
screen_choice = 0
# 模型的选择
model_choice = 'VGG16'
# 图像的宽度和高度
input_shape = (96, 96, 3)

# 训练模型时的学习率
learning_rate = 0.0001

# 训练模型时的损失函数
loss_function = 'mse'

# 训练模型的迭代次数
epochs = 100

# 保存模型的文件名
model_file = 'model.h5'

# 图像文件夹的名称
images_folder = 'Images'
