import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy 	# 浅拷贝 copy 深拷贝 deepcopy

######################################################################
# 数据加载
# ---------
# 训练集：数据增强+标准化
# 验证集：标准化
data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(224),	# 将图片随机裁切成224*224的大小
        transforms.RandomHorizontalFlip(),	# 将图片进行随机水平翻转
        transforms.ToTensor(),				# 将图片转换成tensor数据格式
        # 把一个取值范围是[0,255]的PIL.Image或者shape为(H,W,C)的numpy.ndarray转换成
        #   形状为[C,H,W]取值范围是[0,1.0]的torch.FloatTensor
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        									# 给定均值：(R,G,B) 方差：(R,G,B),将Tensor正则化
    ]),
    'val': transforms.Compose([
        transforms.Resize(256),				# 将图片随机裁切成256*256的大小
        transforms.CenterCrop(224),			# 图片中心裁切成224*224的大小
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

data_dir = 'hymenoptera_data'				# 数据集路径名
image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),data_transforms[x])
							# ImageFolder : 通用数据加载器，根据文件夹的名称给其分类
								# para1: 根文件夹路径train/val  join方法:路径拼接
                                # para2: 调用data_transforms的图片预处理
                  	for x in ['train', 'val']}
dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x],
				# 数据集组合采样器，并在数据集上提供单进程或多进程迭代器
							batch_size=16, 	# mini-batch的每个batch包含的数据大小
							shuffle=True, 	# 随机打乱
							num_workers=0) 	# 用多少个子进程加载数据。0表示数据将在主进程中加载(默认: 0)
              		for x in ['train', 'val']}
dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}	# 记录train/var的数据集大小
class_names = image_datasets['train'].classes # 数据的类型名

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


######################################################################
# 图片数据显示
# ----------------------
def imshow(inp, title=None):
    """Imshow for Tensor."""
    inp = inp.numpy().transpose((1, 2, 0)) 	# 三维转置,交换维度, (C,H,W)->(H,W,C)
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean					# 归一化
    inp = np.clip(inp, 0, 1) 				# 将矩阵中的元素限制在a_min, a_max之间, 标准化
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)  						# 暂停一段时间使得plot得以正常更新


# Get a batch of training data
inputs, classes = next(iter(dataloaders['train']))
	# 通过iter()函数获取这些可迭代对象的迭代器,对获取到的迭代器不断使用
	# next()函数来获取下一条数据
	# (iter()函数实际上就是调用了可迭代对象的__iter__方法)
	# inputs为一个batch的图片集合 (B x C x H x W)
# 制作图片阵列
out = torchvision.utils.make_grid(inputs) # 类似于做图像拼接，横向摆放方便显示(自带padding)
torchvision.utils.save_image(inputs, 'inputInstances.JPG') # 图像保存
imshow(out, title=[class_names[x] for x in classes])	   # 图像展示


######################################################################
# 模型训练
# ------------------
# -  Scheduling the learning rate
# -  Saving the best model
def train_model(model, criterion, optimizer, scheduler, num_epochs=25):
    since = time.time()		# 计时

    best_model_wts = copy.deepcopy(model.state_dict())  # state_dict: 返回包含模块整个状态的字典
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # 对每一个epoch都有训练和验证阶段
        for phase in ['train', 'val']:
            if phase == 'train':
                scheduler.step()
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # 对数据进行迭代
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()		# 参数梯度归零

                # with 语句：适用于对资源进行访问的场合，确保不管使用过程中是否发生异常
                # 都会执行必要的“清理”操作，释放资源
                with torch.set_grad_enabled(phase == 'train'):  # 根据其参数模式启用或禁用grads
                	# forward
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)  # 返回给定维度dim中输入张量的每一行的最大值,即预测值
                    loss = criterion(outputs, labels) # loss值计算

                    # backward + 仅在训练阶段进行参数更新
                    if phase == 'train':
                        loss.backward()			# 计算当前张量的梯度并逆向传播更新参数
                        optimizer.step()		# 执行单个优化步骤

                # information statistics
                running_loss += loss.item() * inputs.size(0) 	# item() → number
                												# size(0): size结果的第一位
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]
            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
          		# deep copy the model
            	# 深度复制：完全复制整个模型，修改新模型不会影响原模型
        print() # \n

    time_elapsed = time.time() - since # 计算总用时
    print('Training complete in {:.0f}m {:.0f}s'
    	.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # 加载最佳模型的参数/权重
    model.load_state_dict(best_model_wts)
    return model


######################################################################
# 模型预测效果显示函数
# ---------------------------------
# Generic function to display predictions for a few images
def visualize_model(model, num_images=6):
    was_training = model.training
    model.eval()      # Sets the module in evaluation mode.
    images_so_far = 0
    fig = plt.figure()

    with torch.no_grad():
        for i, (inputs, labels) in enumerate(dataloaders['val']):
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)	# 预测

            for j in range(inputs.size()[0]):
                images_so_far += 1
                ax = plt.subplot(num_images//2, 2, images_so_far)       # 将多个图画到一个平面
                ax.axis('off')		# 不显示坐标轴
                ax.set_title('predicted: {}'.format(class_names[preds[j]]))
                imshow(inputs.cpu().data[j])

                if images_so_far == num_images:
                    model.train(mode=was_training)          # Sets the module in training mode
                    return
        model.train(mode=was_training)


#############################################################################################
# 卷积网络参数微调
# ----------------------
# 加载训练好的resnet18的模型，并修改最后一层使得成为二分类问题
model_ft = models.resnet18(pretrained=True)	# 加载训练好的resnet18的模型
num_ftrs = model_ft.fc.in_features  		# in_features: 返回全连接层的输入尺寸
model_ft.fc = nn.Linear(num_ftrs, 2)		# 修改全连接层，使得最终输出尺寸为2

model_ft = model_ft.to(device)

criterion = nn.CrossEntropyLoss()	# 交叉熵损失函数

# 使用SGD进行参数更新， lr设低，微调即可
optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.001, momentum=0.9)

# 学习率衰减   Decay LR by a factor of 0.1 every 7 epochs
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)

#############################
# 训练+验证
# ------------------
# It should take around 15-25 min on CPU. On GPU though, it takes less than a
# minute.
model_ft = train_model(model_ft, criterion, optimizer_ft, exp_lr_scheduler, num_epochs=25)

visualize_model(model_ft)
torch.save(model_ft, 'model_Finetuning.pkl')


############################################################################################
# ConvNet作为固定特征提取器
# ----------------------------------
# Here, we need to freeze all the network except the final layer. We need
# to set ``requires_grad == False`` to freeze the parameters so that the
# gradients are not computed in ``backward()``.
#
# You can read more about this in the documentation
# `here <https://pytorch.org/docs/notes/autograd.html#excluding-subgraphs-from-backward>`__.

model_conv = torchvision.models.resnet18(pretrained=True)	# 加载训练好的resnet18的模型
for param in model_conv.parameters():  	# 冻结参数使得在backward时不会更新参数
	param.requires_grad = False

# 默认情况下，新构造的模块的参数具有requires_grad = True
# 使得在训练时只更新最后全连接层的参数
num_ftrs = model_conv.fc.in_features	# 同finetuning
model_conv.fc = nn.Linear(num_ftrs, 2)

model_conv = model_conv.to(device)

criterion = nn.CrossEntropyLoss()

optimizer_conv = optim.SGD(model_conv.fc.parameters(), lr=0.001, momentum=0.9)

# Decay LR by a factor of 0.1 every 7 epochs
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_conv, step_size=7, gamma=0.1)

#############################
# 训练+验证
# ------------------
# On CPU this will take about half the time compared to previous scenario.
# This is expected as gradients don't need to be computed for most of the
# network. However, forward does need to be computed.
#
model_conv = train_model(model_conv, criterion, optimizer_conv,
                         exp_lr_scheduler, num_epochs=25)

visualize_model(model_conv)
torch.save(model_conv, 'model_Fixed.pkl')

plt.ioff()
plt.show()