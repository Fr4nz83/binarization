import numpy as np
from PIL import Image

import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms


def get_dataloaders(args):
    print(f"Loading Dataset {args.dataset}")

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    if args.dataset == "cifar10":

        train_loader = torch.utils.data.DataLoader(
            datasets.CIFAR10(root='./data', train=True, transform=transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.RandomResizedCrop((args.img_size, args.img_size), scale=(0.05, 1.0)),
                transforms.ToTensor(),
                normalize,
            ]), download=True),
            batch_size=args.train_batch_size, shuffle=True,
            num_workers=args.workers, pin_memory=True)

        val_loader = torch.utils.data.DataLoader(
            datasets.CIFAR10(root='./data', train=False, transform=transforms.Compose([
                transforms.Resize((args.img_size, args.img_size)),
                transforms.ToTensor(),
                normalize,
            ])),
            batch_size=args.eval_batch_size, shuffle=False,
            num_workers=args.workers, pin_memory=True)
    else:
        raise NameError("Wrong Dataset ")

    return train_loader, val_loader
