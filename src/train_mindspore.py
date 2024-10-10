import argparse
import os
import gzip
import time
from sipbuild.generator.outputs import output_api
from tqdm import tqdm
import numpy as np
import mindspore as ms
import mindspore.nn as nn
import mindspore.dataset.vision as vision
import mindspore.dataset.transforms as transforms
from mindspore import Model, ModelCheckpoint, CheckpointConfig, LossMonitor, TimeMonitor, Accuracy, History, EarlyStopping
from matplotlib import pyplot as plt

ms.context.set_context(mode=ms.context.GRAPH_MODE, device_target="GPU", device_id=5)


class LeNet(nn.Cell):
    def __init__(self):
        super(LeNet, self).__init__()
        self.sequences = nn.SequentialCell(
            nn.Conv2d(1, 6, 5, 1, "pad", padding=2),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(6, 16, 5, 1, "pad"),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Flatten(),
            # nn.Dropout(0.5),
            nn.Dense(400, 120),
            nn.ReLU(),
            nn.Dropout(p=0.1),
            nn.Dense(120, 84),
            nn.ReLU(),
            nn.Dropout(p=0.05),
            nn.Dense(84, 10),
            nn.LogSoftmax(1)
        )

    def construct(self, x):
        return self.sequences(x)


class Network(nn.Cell):
    def __init__(self):
        super(Network, self).__init__()
        self.sequences = nn.SequentialCell(
            nn.Conv2d(1, 20, 5, 1, "pad"),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Flatten(),
            nn.Dense(12 * 12 * 20, 500),
            nn.ReLU(),
            nn.Dense(500, 10),
            nn.LogSoftmax(1)
        )

    def construct(self, x):
        return self.sequences(x)

class ImageList:
    def __init__(self, path, kind, transform=None):
        super(ImageList, self).__init__()
        self.labels_path = os.path.join(path, "%s-labels-idx1-ubyte.gz" % kind)
        self.images_path = os.path.join(path, "%s-images-idx3-ubyte.gz" % kind)
        self.images = None
        self.labels = None
        self.transform = transform
        self.__load_data()

    def __load_data(self):
        with gzip.open(self.labels_path, 'rb') as lbpath:
            self.labels = np.frombuffer(lbpath.read(), dtype=np.uint8, offset=8)
        with gzip.open(self.images_path, 'rb') as imgpath:
            self.images = np.frombuffer(imgpath.read(), dtype=np.uint8, offset=16).reshape(len(self.labels), 28, 28)
        print("Images and Labels Loaded Successfully.")

    def __getitem__(self, index):
        image, label = self.images[index], int(self.labels[index])
        if self.transform is not None:
            image = self.transform(image)
        return image, ms.Tensor(label, ms.int32)

    def __len__(self):
        assert len(self.images) == len(self.labels)
        return len(self.images)


epochs = 15
batch_size = 64
#learning_rate = 0.5

network = Network()
loss_fn = nn.CrossEntropyLoss()
#optimizer = nn.Adam(network.trainable_params())
optimizer = nn.SGD(network.trainable_params(), learning_rate=0.1, momentum=0.5)
model = Model(network, loss_fn, optimizer, metrics={'accuracy'})


transform = transforms.Compose([
    # vision.Resize((32, 32)),  # 将图片调整到32x32大小
    # vision.RandomHorizontalFlip(prob=0.5),  # 50%的几率水平翻转
    # vision.RandomRotation(degrees=15),  # 随机旋转图像，角度范围为±15度
    # vision.RandomResizedCrop((28, 28), scale=(0.8, 1.0)),  # 随机裁剪图像并调整回28x28大小
    # vision.RandomColorAdjust(brightness=0.2, contrast=0.2),  # 随机调整图像亮度和对比度
    vision.ToTensor(),  # 将图像转换为张量
    vision.Normalize(mean=(0.5, ), std=(0.5, ))  # 正则化
])

train_data = ImageList("../data/oracle", kind="train", transform=transform)
test_data = ImageList("../data/oracle", kind="t10k", transform=transforms.Compose([vision.ToTensor(), vision.Normalize((0.5, ), (0.5, ))]))
train_loader = ms.dataset.GeneratorDataset(train_data, column_names=["image", "label"], shuffle=True).batch(batch_size, drop_remainder=True)
test_loader = ms.dataset.GeneratorDataset(test_data, column_names=["image", "label"], shuffle=False).batch(batch_size * 2)

loss_callback = LossMonitor()
# model.train(epochs, train_loader, callbacks=[time_callback, loss_callback, history_callback])
# model.eval(test_loader, callbacks=[time_callback, loss_callback, history_callback])
model.fit(epochs, train_loader, test_loader, callbacks=[TimeMonitor(), loss_callback], dataset_sink_mode=True)

