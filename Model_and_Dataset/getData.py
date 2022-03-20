from torchvision import transforms
from torch.utils import data
from my_dataset import MyDataSet
from utils import read_split_data

# 数据集根目录
root = "D:/Coding/ClassificationModel/data"


# 得到dataloader的函数,参数batch_size为每一批次读入的图片数量
def load_my_dataset(batch_size):
    # 通过使用utils.py中定义的函数来得到训练集和验证集以及对应的类别标签
    train_images_path, train_images_label, val_images_path, val_images_label = read_split_data(root)
    # transform,图片数据预处理方式,变成224*224的图片样式并转换成张量形式
    data_transform = {
        "train": transforms.Compose([transforms.RandomResizedCrop(224),
                                     transforms.RandomHorizontalFlip(),
                                     transforms.ToTensor(),
                                     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),
        "val": transforms.Compose([transforms.Resize(256),
                                  transforms.CenterCrop(224),
                                  transforms.ToTensor(),
                                  transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    }
    # 将训练集和验证集载入到dataset中
    train_data_set = MyDataSet(images_path=train_images_path,
                               images_class=train_images_label,
                               transform=data_transform["train"])
    val_data_set = MyDataSet(images_path=val_images_path,
                             images_class=val_images_label,
                             transform=data_transform["val"])
    val_num = len(val_data_set)

    # 由dataset得到torchvision提供的数据加载器DataLoader,加载数据线程数为6,并随机打乱
    batch_size = batch_size
    train_loader = data.DataLoader(train_data_set,
                                   batch_size=batch_size,
                                   shuffle=True,
                                   num_workers=6,
                                   collate_fn=train_data_set.collate_fn)
    val_loader = data.DataLoader(val_data_set,
                                 batch_size=batch_size,
                                 shuffle=True,
                                 num_workers=6,
                                 collate_fn=val_data_set.collate_fn)

    return train_loader, val_loader, val_num




