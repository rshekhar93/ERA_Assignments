import torch
import numpy as np
from torchvision import datasets
import albumentations as A
from albumentations.pytorch import ToTensorV2

class AlbumentationTransform:
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, img):
        return self.transform(image=np.array(img))['image']

class CIFAR10Albumentation:
    def __init__(self, train=True):
        self.train = train
        self.train_transforms = A.Compose([
            A.HorizontalFlip(p=0.5),
            A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=10, p=0.5),
            A.CoarseDropout(
                max_holes=1, max_height=16, max_width=16,
                min_holes=1, min_height=16, min_width=16,
                fill_value=(0.4914, 0.4822, 0.4465),
                p=0.5
            ),
            A.Normalize(
                mean=[0.4914, 0.4822, 0.4465],
                std=[0.2023, 0.1994, 0.2010]
            ),
            ToTensorV2()
        ])
        
        self.test_transforms = A.Compose([
            A.Normalize(
                mean=[0.4914, 0.4822, 0.4465],
                std=[0.2023, 0.1994, 0.2010]
            ),
            ToTensorV2()
        ])

    def get_dataloader(self, batch_size=128):
        trainset = datasets.CIFAR10(
            root='./data', train=True, download=True,
            transform=AlbumentationTransform(self.train_transforms)
        )
        trainloader = torch.utils.data.DataLoader(
            trainset, batch_size=batch_size, shuffle=True, num_workers=2
        )

        testset = datasets.CIFAR10(
            root='./data', train=False, download=True,
            transform=AlbumentationTransform(self.test_transforms)
        )
        testloader = torch.utils.data.DataLoader(
            testset, batch_size=batch_size, shuffle=False, num_workers=2
        )

        return trainloader, testloader 