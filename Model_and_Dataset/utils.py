import os
import json
import random
import matplotlib.pyplot as plt
import matplotlib as mpl

# ***定义处理原数据集文件的函数***
# 划分成训练集和验证集


# 处理原数据集文件的函数,按0.2的比例划分出验证集,并遍历所有的图片文件得到其路径和对应的类别标签
def read_split_data(root: str, val_rate: float = 0.2):
    random.seed(0)  # 设置随机种子
    assert os.path.exists(root), "Model_and_Dataset root: {} does not exist.".format(root)

    # 遍历文件夹,将一个文件夹对应一个类别
    garbage_class = [cla for cla in os.listdir(root) if os.path.isdir(os.path.join(root, cla))]
    # 排序，保持顺序一致
    garbage_class.sort()
    # 生成类别名称以及对应的数字索引,并以一个json文件保存
    class_indices = dict((k, v) for v, k in enumerate(garbage_class))
    json_str = json.dumps(dict((val, key) for key, val in class_indices.items()), indent=4, ensure_ascii=False)
    with open('class_indices.json', 'w') as json_file:
        json_file.write(json_str)

    train_images_path = []  # 存储训练集的所以图片路径
    train_images_label = []  # 存储训练集图片对应的索引信息
    val_images_path = []  # 存储验证集的所有图片信息
    val_images_label = []  # 存储验证集图片对应的索引信息
    every_class_num = []  # 存储每个类别的样本总数
    supported = [".jpg", ".JPG", ".jpeg"]  # 支持的文件后缀类型
    # 遍历每个文件夹下的图片文件
    for cla in garbage_class:
        cla_path = os.path.join(root, cla)
        # 遍历获取supported支持的所有文件路径
        images = [os.path.join(root, cla, i) for i in os.listdir(cla_path) if os.path.splitext(i)[-1] in supported]
        # 获取该类别对应的索引
        image_class = class_indices[cla]
        # 记录该类别样本数量
        every_class_num.append(len(images))
        # 按比例随机采样验证样本
        eval_path = random.sample(images, k=int(len(images) * val_rate))

        for img_path in images:
            if img_path in eval_path:
                val_images_path.append(img_path)
                val_images_label.append(image_class)
            else:
                train_images_path.append(img_path)
                train_images_label.append(image_class)

    print("{} images were found in the Model_and_Dataset.".format(sum(every_class_num)))

    # 画柱状图显示数据集每种类别的图片数量
    plot_image = False
    if plot_image:
        plt.bar(range(len(garbage_class)), every_class_num, align='center')
        plt.xticks(range(len(garbage_class)), garbage_class)

        for i, v in enumerate(every_class_num):
            plt.text(x=i, y=v + 5, s=str(v), ha='center')

        plt.xlabel('image class')
        plt.ylabel('number of images')
        plt.title('garbage class distribution')
        plt.show()

    return train_images_path, train_images_label, val_images_path, val_images_label


def plot_data_loader_image(data_loader):
    batch_size = data_loader.batch_size
    plot_num = min(batch_size, 4)

    json_path = './class_indices.json'
    assert os.path.exists(json_path), json_path + " doesn't exist."
    json_file = open(json_path, 'r')
    class_indices = json.load(json_file)

    mpl.rcParams['font.family'] = 'SimHei'
    for data in data_loader:
        images, labels = data
        for i in range(plot_num):
            img = images[i].numpy().transpose(1, 2, 0)
            img = (img * [0.229, 0.224, 0.225] + [0.485, 0.456, 0.406]) * 255
            label = labels[i].item()
            plt.subplot(1, plot_num, i+1)
            plt.xlabel(class_indices[str(label)])
            plt.xticks([])
            plt.yticks([])
            plt.imshow(img.astype('uint8'))
        plt.show()

