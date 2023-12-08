'''Train CIFAR10 with PyTorch.'''
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms

import os
import argparse

from models.vgg import VGG
from dataset import HeroDataset, HeroTestDataset


# Training
def train(model, trainloader, optimizer, criterion, epoch):
    print('\nEpoch: %d' % epoch)
    model.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

    print('Train Loss: %.2f | Acc: %.2f%% (%d/%d)' % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))


def test(model, testloader, criterion, epoch, best_acc=0):
    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

        print('Test Loss: %.2f | Acc: %.2f%% (%d/%d)' % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))

    # Save checkpoint.
    acc = 100.*correct/total
    if acc > best_acc:
        print('Saving..')
        state = {
            'model': model.state_dict(),
            'acc': acc,
            'epoch': epoch,
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save(state, './checkpoint/ckpt.pth')
        best_acc = acc

    return best_acc

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Training model VGG19')
    parser.add_argument('--lr', default=0.001, type=float, help='learning rate')
    parser.add_argument('--epoch', default=200, type=int, help='num of training epochs')
    parser.add_argument('--wd', default=1e-2, type=float, help='weight decay')
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Data
    print('==> Preparing data..')
    transform_train = transforms.Compose([
        transforms.ToTensor(),
        transforms.RandomChoice([transforms.GaussianBlur(kernel_size=(7, 7), sigma=(0.5, 7.))]),
        # transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        transforms.RandomChoice([transforms.CenterCrop(size=size) for size in (70, 90, 100, 110, 128)]),
        transforms.RandomChoice([transforms.Pad(padding=padding) for padding in (2, 5, 10, 15)]),
        transforms.RandomChoice([transforms.Resize(size=size) for size in (70, 90, 100, 120)]),
        transforms.Resize(size=32)
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        # transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        transforms.Resize(size=32)
    ])

    trainset = HeroDataset(transform=transform_train)
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=32, shuffle=True, num_workers=2)

    testset = HeroTestDataset(trainset.get_class_dict(), transform=transform_test)
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=32, shuffle=False, num_workers=2)

    classes = trainset.get_class_names()

    # Model
    print('==> Building model..')
    model = VGG('VGG19')

    # replace the last layer with classes
    model.classifier = nn.Linear(512, len(classes))

    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.wd)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)

    best_acc = 0
    for epoch in range(args.epoch):
        train(model, trainloader, optimizer, criterion, epoch)
        best_acc = test(model, testloader, criterion, epoch, best_acc)
        scheduler.step()