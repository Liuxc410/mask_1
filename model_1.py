import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import csv
# 这段代码设置了一些变量，用于训练一个机器学习模型。
# 其中，train_data_dir、test_data_dir和val_data_dir分别指定了训练、测试和验证数据集所在的目录，
# img_height和img_width分别指定了图像的高度和宽度，batch_size指定了每个batch的大小，num_classes则指定了分类任务的类别数。

train_data_dir = "./train_test_val/train/"
test_data_dir = "./train_test_val/test/"
val_data_dir = "./train_test_val/val/"
img_height, img_width = 128, 128
batch_size = 32
num_classes = 2
num_epochs = 50

# 数据预处理
train_transforms = transforms.Compose([
    transforms.Resize((img_height, img_width)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

test_transforms = transforms.Compose([
    transforms.Resize((img_height, img_width)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

val_transforms = transforms.Compose([
    transforms.Resize((img_height, img_width)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

train_dataset = datasets.ImageFolder(
    train_data_dir,
    transform=train_transforms
)

test_dataset = datasets.ImageFolder(
    test_data_dir,
    transform=test_transforms
)

val_dataset = datasets.ImageFolder(
    val_data_dir,
    transform=test_transforms
)

train_loader = torch.utils.data.DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True
)

test_loader = torch.utils.data.DataLoader(
    test_dataset,
    batch_size=batch_size,
    shuffle=False
)

val_loader = torch.utils.data.DataLoader(
    val_dataset,
    batch_size=batch_size,
    shuffle=False
)

# 定义模型
class Net(nn.Module):      #卷积神经网络模型

    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3,32,3,padding=1)
        self.conv2 = nn.Conv2d(32,64,3,padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.relu1 = nn.ReLU()

        self.conv3 = nn.Conv2d(64,128,3,padding=1)
        self.conv4 = nn.Conv2d(128,256,3,padding=1)
        self.bn2 = nn.BatchNorm2d(256)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.relu2 = nn.ReLU()

        self.conv5 = nn.Conv2d(256,512,3,padding=1)
        self.conv6 = nn.Conv2d(512,256,3,padding=1)
        self.bn3 = nn.BatchNorm2d(256)
        self.pool3 = nn.MaxPool2d(2, 2)
        self.relu3 = nn.ReLU()

        self.fc1 = nn.Linear(256*16*16, 128)
        self.relu11 = nn.ReLU()
        self.drop1 = nn.Dropout()
        self.fc2 = nn.Linear(128, 64)
        self.relu12 = nn.ReLU()
        self.drop2 = nn.Dropout()
        self.fc3 = nn.Linear(64, num_classes)


    def forward(self,x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.pool1(x)
        x = self.relu1(x)

        x = self.conv3(x)
        x = self.conv4(x)
        x = self.pool2(x)
        x = self.relu2(x)

        x = self.conv5(x)
        x = self.conv6(x)
        x = self.pool3(x)
        x = self.relu3(x)

        x = x.view(-1, 256*16*16)
        x = self.relu11(self.fc1(x))
        x = self.drop1(x)
        x = self.relu12(self.fc2(x))
        x = self.drop2(x)
        x = self.fc3(x)

        return x

model = Net()

# 指定可见的GPU设备
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    torch.cuda.set_device(0)
    model.to(device)
    torch.backends.cudnn.benchmark = True

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)

#创建CSV文件
with open('./txt/mask_50.csv_1', mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['epoch', 'train_loss', 'train_acc', 'val_loss', 'val_acc'])

# 训练模型
for epoch in range(num_epochs):
    running_loss = 0.0
    running_corrects = 0
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        running_corrects += torch.sum(preds == labels.data)

    train_loss = running_loss / len(train_dataset)
    train_acc = running_corrects.double() / len(train_dataset)
    model.eval()
    with torch.no_grad():
        running_loss = 0.0
        running_corrects = 0
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)

        val_loss = running_loss / len(val_dataset)
        val_acc = running_corrects.double() / len(val_dataset)

    print('Epoch [{}/{}], Train Loss: {:.4f}, Train Acc: {:.4f}, Val Loss: {:.4f}, Val Acc: {:.4f}'
          .format(epoch+1, num_epochs, train_loss, train_acc, val_loss, val_acc))
    # 写入CSV文件
    with open('./txt/mask_50_1.csv', mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([epoch+1, train_loss, train_acc.item(), val_loss, val_acc.item()])


# 测试模型
model.eval()
with torch.no_grad():
    running_loss = 0.0
    running_corrects = 0
    for inputs, labels in test_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)
        loss = criterion(outputs, labels)

        running_loss += loss.item() * inputs.size(0)
        running_corrects += torch.sum(preds == labels.data)

    test_loss = running_loss / len(test_dataset)
    test_acc = running_corrects.double() / len(test_dataset)

print('Test accuracy: {:.4f}, Test loss: {:.4f}'.format(test_acc.item(), test_loss))
torch.save(model.state_dict(), './model/mask_50_1.pth')
