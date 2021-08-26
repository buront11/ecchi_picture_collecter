import glob
import json
import os
from natsort import natsorted

from PIL import Image

import torch
import torch.nn as nn
import torchvision
from torchvision import models, transforms

def preprocess_img(img_paths):
    transform = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ])

    images = {}
    for path in img_paths:
        img = Image.open(path).convert('RGB')
        filename = os.path.splitext(os.path.basename(path))[0]
        img = transform(img)

        images.update({filename:img})

    return images

def eval():
    vgg = models.vgg11(pretrained=True)
    print(vgg)
    vgg.classifier[6] = nn.Linear(in_features=4096, out_features=3)

    model_path = './best_model'
    vgg.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))

    test_image_paths = natsorted(glob.glob('./test_img/*'))
    images = preprocess_img(test_image_paths)

    result_dict = {}

    vgg.eval()

    with torch.no_grad():
        for filename, image in images.items():
            sigmoid = nn.Sigmoid()
            output = vgg(image.unsqueeze(0))
            output = sigmoid(output)
            pred = torch.gt(output, 0.5)
            result_dict.update({filename:[output.tolist(), pred.tolist()]})

    with open('result.json', 'w') as f:
        json.dump(result_dict, f, indent=4)

if __name__=='__main__':
    eval()