import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from collections import OrderedDict

# Загрузка модели
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
net = torchvision.models.alexnet()  # Создайте экземпляр модели
num_classes = 3  # Укажите количество классов в вашем наборе данных
net.classifier[6] = nn.Linear(4096, num_classes)  # Замените последний слой на новый

# Загрузка весов из файла с обновлением ключей
state_dict = torch.load('CnNet.ckpt')
new_state_dict = OrderedDict()

for k, v in state_dict.items():
    name = k
    if name == 'classifier.fc.weight':
        name = 'classifier.6.weight'
    elif name == 'classifier.fc.bias':
        name = 'classifier.6.bias'
    new_state_dict[name] = v

net.load_state_dict(new_state_dict)  # Загрузите веса из файла
net = net.to(device)  # Переместите модель на устройство для вычислений (CPU или GPU)

# Подготовка данных
data_transforms = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Загрузка изображения
your_image = 'test.jpg'  # Здесь должно быть ваше изображение
inputs = data_transforms(your_image).unsqueeze(0).to(device)

# Предсказание
with torch.no_grad():
    outputs = net(inputs)
    _, preds = torch.max(outputs, 1)

# Вывод результата
print('Предсказанный класс: ', preds.item())
