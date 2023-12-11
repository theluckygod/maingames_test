import torch
import torch.nn as nn
import torchvision.transforms as transforms

from models.vgg import VGG
from dataset import HeroDataset, HeroTestDataset

torch.manual_seed(42)
device = 'cuda'

def test(model, testloader, criterion):
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

            if predicted != targets:
                print(classes[predicted.item()], f"label={classes[targets.item()]}")
    print('Test Loss: %.2f | Acc: %.2f%% (%d/%d)' % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))

if __name__ == "__main__":
    trainset = HeroDataset()
    classes = trainset.get_class_names()

    model = VGG('VGG19')
    # replace the last layer with classes
    model.classifier = nn.Linear(512, len(classes))

    model = model.to(device)

    state_dict = torch.load("checkpoint/ckpt.pth", map_location='cpu')
    model.load_state_dict(state_dict['model'])
    print(f"State dict loaded. Acc: {state_dict['acc']}")

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        # transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        transforms.Resize(size=32)
    ])
    testset = HeroTestDataset(trainset.get_class_dict(), transform=transform_test)
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=1, shuffle=False, num_workers=2)

    criterion = nn.CrossEntropyLoss()

    test(model, testloader, criterion)
