import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt

if __name__ == '__main__':
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    batch_size = 10

    data_transforms = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    train_dataset = torchvision.datasets.ImageFolder(root='./animals/train',
                                                     transform=data_transforms)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size,
                                               shuffle=True, num_workers=2)
    train_dataset.classes
    class_names = train_dataset.classes

    len(train_dataset.samples)

    test_dataset = torchvision.datasets.ImageFolder(root='./animals/val',
                                                    transform=data_transforms)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size,
                                              shuffle=True, num_workers=2)

    inputs, classes = next(iter(train_loader))
    inputs.shape

    img = torchvision.utils.make_grid(inputs, nrow=5)
    img = img.numpy().transpose((1, 2, 0))
    plt.imshow(img)

    net = torchvision.models.alexnet(pretrained=True)

    for param in net.parameters():
        param.requires_grad = False

    num_classes = 3

    new_classifier = net.classifier[:-1]
    new_classifier.add_module('fc', nn.Linear(4096, num_classes))
    net.classifier = new_classifier

    net = net.to(device)
    correct_predictions = 0
    num_test_samples = len(test_dataset)

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            pred = net(images)
            _, pred_class = torch.max(pred.data, 1)
            correct_predictions += (pred_class == labels).sum().item()

    print('Точность модели: ' + str(100 * correct_predictions / num_test_samples) + '%')
    num_epochs = 2
    lossFn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(net.parameters(), lr=0.01)
    import time

    t = time.time()

    # Создайте список для хранения значений ошибки
    loss_values = []

    num_epochs = 2
    lossFn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(net.parameters(), lr=0.01)

    for epoch in range(num_epochs):
        for i, (images, labels) in enumerate(train_loader):
            images = images.to(device)
            labels = labels.to(device)
            outputs = net(images)
            loss = lossFn(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Сохраните значение ошибки
            loss_values.append(loss.item())

            if i % 100 == 0:
                print('Эпоха ' + str(epoch) + ' из ' + str(num_epochs) + ' Шаг ' +
                      str(i) + ' Ошибка: ', loss.item())

    # Постройте график значений ошибки
    plt.figure(figsize=(10, 5))
    plt.title("Ошибка обучения")
    plt.plot(loss_values, label="Train")
    plt.xlabel("Итерации")
    plt.ylabel("Ошибка")
    plt.legend()
    plt.show()

    print(time.time() - t)
    correct_predictions = 0
    num_test_samples = len(test_dataset)

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            pred = net(images)
            _, pred_class = torch.max(pred.data, 1)
            correct_predictions += (pred_class == labels).sum().item()

    print('Точность модели: ' + str(100 * correct_predictions / num_test_samples) + '%')

    inputs, classes = next(iter(test_loader))
    pred = net(inputs.to(device))
    _, pred_class = torch.max(pred.data, 1)

    for i, j in zip(inputs, pred_class):
        img = i.numpy().transpose((1, 2, 0))
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        img = std * img + mean
        img = np.clip(img, 0, 1)
        plt.imshow(img)
        plt.title(class_names[j])
        plt.pause(2)

    torch.save(net.state_dict(), 'CnNet.ckpt')
