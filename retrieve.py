import argparse

import torch
import torch.nn as nn
import torchvision.transforms as transforms

from models.vgg import VGG
from dataset import HeroDataset, HeroInferenceDataset

from torch.nn.functional import cosine_similarity

import utils

torch.manual_seed(42)
device = 'cuda'


def predict(model, class_embeds, testset, classes: list, output_path: str):
    model.eval()

    predictions = []
    for i in range(len(testset)):
        input, filename = testset[i]
        input = input.unsqueeze(0).to(device)
        
        with torch.no_grad():
            outputs = model(input, get_features=True)

        # calculate cosine similarity
        sim = cosine_similarity(outputs, class_embeds)
        predicted = torch.argmax(sim).item()
        predicted = classes[predicted]
        predicted = utils.correct_label(predicted)

        predictions.append((filename, predicted))

    with open(output_path, 'w') as f:
        for filename, predicted in predictions:
            f.write(f"{filename}\t{predicted}\n")

    print(f"Predictions saved to {output_path}")

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
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint_path', default='checkpoint/ckpt.pth', type=str, help='checkpoint path')
    parser.add_argument('--data_path', default='/test_images', type=str, help='images path')
    parser.add_argument('--output_path', default='/outputs/test.txt', type=str, help='output path')
    args = parser.parse_args()    

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

    state_dict = torch.load(args.checkpoint_path, map_location='cpu')
    model.load_state_dict(state_dict['model'])
    print(f"State dict loaded. Acc: {state_dict['acc']}")

    # get class embeddings
    class_embeds, _ = get_class_embeds(model, trainset)


    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize(size=32)
    ])
    testset = HeroInferenceDataset(trainset.get_class_dict(),
                              images_path=args.data_path,
                              transform=transform_test)

    predict(model, class_embeds, testset, classes, args.output_path)
