import torch
import json
from torchvision import transforms
from PIL import Image
from model import resnet_34


# ***一个简单的预测脚本***
# 输入一张图片，返回模型预测的类别

# 输入图片的预处理方式,转成可以输入进模型的大小并转换成张量
images_transform = transforms.Compose(
    [transforms.Resize(256),
     transforms.CenterCrop(224),
     transforms.ToTensor(),
     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]
)

# 要进行预测的图片的路径
img_path = "D:/Coding/ClassificationModel/data/有害垃圾_废旧灯管灯泡/baidu000007.png"
# 如果图片不是RGB图片则转换成RGB图片
img = Image.open(img_path)
if img.mode != 'RGB':
    img = img.convert("RGB")
img = images_transform(img)
img = torch.unsqueeze(img, dim=0)

# 使用之前建立的json文件得到图片类别列表
try:
    json_file = open('./class_indices.json', 'r')
    class_indict = json.load(json_file)
except Exception as e:
    print(e)
    exit(-1)

# 定义模型结构,并载入之前训练好的模型参数
model = resnet_34(num_classes=60)
weight_path = "./ResNet34-byMyTrainV2.pth"
model.load_state_dict(torch.load(weight_path))

# 将模型转为评测模式,关闭参数梯度跟踪,得到计算的结果并输出
model.eval()
with torch.no_grad():
    output = torch.squeeze(model(img))
    predict = torch.softmax(output, dim=0)
    predict_class = torch.argmax(predict).numpy()
# 输出预测的类别和模型认为这张图片是这个类别的概率
print(class_indict[str(predict_class)], predict[predict_class].numpy())

