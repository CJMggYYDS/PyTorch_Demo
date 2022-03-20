import torch
import torch.nn as nn
import torch.optim as optim
from d2l import torch as d2l
from getData import load_my_dataset
from model import resnet_34

# ***模型训练脚本***
# 设备:
# CPU 6核12线程,内存16GB
# GPU GTX 1650ti,显存4GB
# CUDA version: 11.1
# 一个epoch计算时间: 4分钟左右


# 调用设备的支持CUDA的nvidia GPU
def try_gpu(i=0):
    if torch.cuda.device_count() >= i + 1:
        return torch.device(f'cuda:{i}')
    return torch.device('cpu')


# 使用GPU计算模型在数据集上的精度
def evaluate_accuracy_gpu(net, data_iter, device=None):
    if isinstance(net, nn.Module):
        net.eval()
        if not device:
            device = next(iter(net.parameters())).device
    metric = d2l.Accumulator(2)
    with torch.no_grad():
        for X, y in data_iter:
            if isinstance(X, list):
                X = [x.to(device) for x in X]
            else:
                X = X.to(device)
            y = y.to(device)
            metric.add(d2l.accuracy(net(X), y), y.numel())
    return metric[0] / metric[1]


# 训练函数
def train_my_net():
    device = try_gpu()
    print(device)

    batch_size = 32
    train_loader, val_loader, val_num = load_my_dataset(batch_size)

    # 使用pytorch官方的resnet-34模型预训练参数完成初始化,再根据自己数据集的类别数更改输出层的out_channel,减少训练迭代周期
    net = resnet_34()
    model_weight_path = "./resnet34-fromTorch.pth"
    net.load_state_dict(torch.load(model_weight_path), strict=False)
    in_channel = net.fc.in_features
    net.fc = nn.Linear(in_channel, 60)

    # 使用交叉熵损失函数,优化器使用经典的随机梯度下降SGD,并使用L2正则化来减轻过拟合
    loss_fun = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.0001, momentum=0.9, weight_decay=0.001)
    # 训练结束后保存模型参数的文件路径名称
    save_path = "./ResNet34-byMyTrainV2.pth"

    num_epochs = 50
    best_acc = 0.0
    net.to(device)
    for epoch in range(num_epochs):
        train_l_sum, train_acc_sum, n = 0.0, 0.0, 0
        net.train()
        for step, data in enumerate(train_loader, start=0):
            images, labels = data
            optimizer.zero_grad()
            images, labels = images.to(device), labels.to(device)
            y = net(images)
            loss = loss_fun(y, labels)
            loss.backward()
            optimizer.step()

            train_l_sum += loss.item()
            train_acc_sum += (y.argmax(dim=1) == labels).sum().item()
            n += labels.shape[0]

            rate = (step + 1) / len(train_loader)
            a = "*" * int(rate * 50)
            b = "." * int((1 - rate) * 50)
            print("\rtrain loss: {:^3.0f}%[{}->{}]{:.4f}".format(int(rate * 100), a, b, loss), end="")
        print()

        net.eval()
        acc = 0.0
        with torch.no_grad():
            for data_set in val_loader:
                test_images, test_labels = data_set
                outputs = net(test_images.to(device))
                predict_y = torch.max(outputs, dim=1)[1]
                acc += (predict_y == test_labels.to(device)).sum().item()
            accurate_test = acc / val_num
            if accurate_test > best_acc:
                best_acc = accurate_test
                torch.save(net.state_dict(), save_path)
            print('[epoch %d] train_loss: %.3f,  train_acc: %.3f,  test_acc: %.3f' %
                  (epoch + 1, train_l_sum / step, train_acc_sum / n, acc / val_num))


if __name__ == "__main__":
    train_my_net()
