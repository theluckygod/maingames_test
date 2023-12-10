import torch
import torch.nn as nn
import torchvision.transforms as transforms

from models.vgg import VGG
from dataset import HeroDataset, HeroTestDataset

from torch.nn.functional import cosine_similarity

import utils

torch.manual_seed(42)
device = 'cuda'


def test(model, class_embeds, testset, classes: list):
    model.eval()

    correct = 0
    total = len(testset)

    for i in range(len(testset)):
        input, target = testset[i]
        input = input.unsqueeze(0).to(device)
        target = classes[target.item()]
        
        with torch.no_grad():
            outputs = model(input, get_features=True)

        # calculate cosine similarity
        sim = cosine_similarity(outputs, class_embeds)
        predicted = torch.argmax(sim).item()
        predicted = classes[predicted]
        predicted = utils.correct_label(predicted)

        correct += (predicted == target)

    print('Acc: %.2f%% (%d/%d)' % (100.*correct/total, correct, total))


def get_class_embeds(model, dataset):
    model.eval()
    embeds = []
    targets = []
    
    
    for i in range(len(dataset)):
        input, target = dataset[i]

        input = input.unsqueeze(0).to(device)
        with torch.no_grad():
            output = model(input, get_features=True)
        embeds.append(output)
        targets.append(target)

    return torch.cat(embeds, dim=0), targets


if __name__ == "__main__":
    transform_train = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize(size=32)
    ])
    trainset = HeroDataset(transform=transform_train)
    classes = trainset.get_class_names()
    print(classes)

    model = VGG('VGG19')
    # replace the last layer with classes
    model.classifier = nn.Linear(512, len(classes))

    model = model.to(device)

    state_dict = torch.load("checkpoint/ckpt.pth", map_location='cpu')
    model.load_state_dict(state_dict['model'])
    print(f"State dict loaded. Acc: {state_dict['acc']}")

    # get class embeddings
    class_embeds, _ = get_class_embeds(model, trainset)


    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize(size=32)
    ])
    testset = HeroTestDataset(trainset.get_class_dict(), transform=transform_test)

    test(model, class_embeds, testset, classes)
