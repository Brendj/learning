import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np

# Проверка доступности GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Преобразования для тренировочных и валидационных данных
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Загрузка тренировочных и валидационных данных
train_data = datasets.ImageFolder('animals/val', transform=transform)
valid_data = datasets.ImageFolder('animals/val', transform=transform)

train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
valid_loader = DataLoader(valid_data, batch_size=32, shuffle=False)

# Определение модели
model = nn.Sequential(
    nn.Conv2d(3, 64, kernel_size=3, padding=1),
    nn.ReLU(),
    nn.MaxPool2d(2),
    nn.Conv2d(64, 128, kernel_size=3, padding=1),
    nn.ReLU(),
    nn.MaxPool2d(2),
    nn.Flatten(),
    nn.Linear(128 * 56 * 56, 256),
    nn.ReLU(),
    nn.Linear(256, 3),
).to(device)  # Перемещение модели на GPU

# Определение функции потерь и оптимизатора
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters())

# Тренировка модели
for epoch in range(2):
    for i, (images, labels) in enumerate(train_loader):
        images = images.to(device)  # Перемещение изображений на GPU
        labels = labels.to(device)  # Перемещение меток на GPU

        outputs = model(images)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Вывод ошибки на каждой сотой итерации
        if i % 1 == 0:
            print(f'Iteration: {i}, Loss: {loss.item()}')

# Сохранение модели
torch.save(model.state_dict(), 'model.pth')
print("Модель успешно сохранена!")

# Валидация модели
model.eval()
with torch.no_grad():
    # Получение 10 случайных изображений из валидационного набора данных
    indices = np.random.choice(range(len(valid_data)), size=10)
    images, labels = zip(*[valid_data[i] for i in indices])

    # Перемещение изображений и меток на GPU
    images = torch.stack(images).to(device)
    labels = torch.tensor(labels).to(device)

    # Получение предсказаний модели
    outputs = model(images)
    _, predicted = torch.max(outputs, 1)

    # Отображение изображений и предсказанных меток
    fig, axs = plt.subplots(2, 5, figsize=(15, 6))
    for i, ax in enumerate(axs.flat):
        ax.imshow(images[i].permute(1, 2, 0).cpu().numpy() * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406]))  # Денормализация изображений
        ax.set_title(f'Predicted: {predicted[i]}')
        ax.axis('off')
    plt.show()
