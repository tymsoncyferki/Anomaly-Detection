import os
import sys

import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision.datasets import PCAM
from torch.utils.data import Dataset

import staintools

sys.path.append("C:/Users/DamnBoii/Python/Anomaly-Detection/AnoGAN")

from fanogan.test_anomaly_detection import test_anomaly_detection
from stainnet.models import load_stain_net

class StainNormalizedDataset(Dataset):

    def __init__(self, input_dataset, stain_net):
        print("Stain normalization in progress...")
        self.normalized_images = []
        self.labels = []
        for i in range(len(input_dataset)):
            self.normalized_images.append(stain_net(input_dataset[i][0]).detach())
            self.labels.append(input_dataset[i][1])
            if i % 2048 == 0 and i > 0:
                print("{} of {} completed".format(i, len(input_dataset)))

    def __len__(self):
        return len(self.normalized_images)

    def __getitem__(self, idx):
        return self.normalized_images[idx], self.labels[idx]


def main(opt):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # device = torch.device("cpu")

    stain_net = load_stain_net("../stainnet/StainNet-Public-centerUni_layer3_ch32.pth")

    pipeline = [transforms.CenterCrop(32),
                transforms.Resize([opt.img_size] * 2),
                transforms.RandomHorizontalFlip()]
    if opt.channels == 1:
        pipeline.append(transforms.Grayscale())
    pipeline.extend([transforms.ToTensor(),
                     transforms.Normalize([0.5] * opt.channels, [0.5] * opt.channels)])

    transform = transforms.Compose(pipeline)
    dataset = PCAM(opt.test_root, split='test', transform=transform, download=opt.force_download)
    dataset = StainNormalizedDataset(dataset, stain_net)
    test_dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

    sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
    from mvtec_ad.model import Generator, Discriminator, Encoder

    generator = Generator(opt)
    discriminator = Discriminator(opt)
    encoder = Encoder(opt)

    test_anomaly_detection(opt, generator, discriminator, encoder,
                           test_dataloader, device)


"""
The code below is:
Copyright (c) 2018 Erik Linder-Norén
Licensed under MIT
(https://github.com/eriklindernoren/PyTorch-GAN/blob/master/LICENSE)
"""

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("test_root", type=str,
                        help="root name of your dataset in test mode")
    parser.add_argument("--force_download", "-f", action="store_true",
                        help="flag of force download")
    parser.add_argument("--latent_dim", type=int, default=100,
                        help="dimensionality of the latent space")
    parser.add_argument("--img_size", type=int, default=64,
                        help="size of each image dimension")
    parser.add_argument("--channels", type=int, default=3,
                        help="number of image channels (If set to 1, convert image to grayscale)")
    opt = parser.parse_args()

    main(opt)
