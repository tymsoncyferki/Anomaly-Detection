import os
import sys

import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision.datasets import PCAM
from torch.utils.data import Subset

import staintools
from stainnet.models import load_stain_net

sys.path.append("C:/Users/tymek/PycharmProjects/anogan")

from fanogan.train_wgangp import train_wgangp


def getSubset(dataset, target, n):
    print(f'Getting label={target} indices...')
    indices_to_keep = [i for i, (_, label) in enumerate(dataset) if label == target]
    indices = indices_to_keep[::n]
    return Subset(dataset, indices)


def main(opt):
    if type(opt.seed) is int:
        torch.manual_seed(opt.seed)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # device = torch.device("cpu")

    stain_normalization = load_stain_net()

    pipeline = [transforms.CenterCrop(32),  # Center crop (tumor counts only in the middle 32x32 area)
                transforms.Resize([opt.img_size]*2),
                transforms.RandomHorizontalFlip()]
    if opt.channels == 1:
        pipeline.append(transforms.Grayscale())
    pipeline.extend([transforms.ToTensor(),
                     transforms.Normalize([0.5] * opt.channels, [0.5] * opt.channels),
                     stain_normalization])

    transform = transforms.Compose(pipeline)
    dataset = PCAM(opt.train_root, split='train', transform=transform, download=opt.force_download)
    normal_dataset = getSubset(dataset, 0, opt.dataset_size)
    train_dataloader = DataLoader(normal_dataset, batch_size=opt.batch_size,
                                  shuffle=False)

    sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
    from mvtec_ad.model import Generator, Discriminator  # May be required to add AnoGAN.mvtec_ad

    generator = Generator(opt)
    discriminator = Discriminator(opt)

    train_wgangp(opt, generator, discriminator, train_dataloader, device)


"""
The code below is:
Copyright (c) 2018 Erik Linder-Norén
Licensed under MIT
(https://github.com/eriklindernoren/PyTorch-GAN/blob/master/LICENSE)
"""


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("train_root", type=str,
                        help="root name of your dataset in train mode")
    parser.add_argument("--force_download", "-f", action="store_true",
                        help="flag of force download")
    parser.add_argument("--n_epochs", type=int, default=300,
                        help="number of epochs of training")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="size of the batches")
    parser.add_argument("--lr", type=float, default=0.0002,
                        help="adam: learning rate")
    parser.add_argument("--b1", type=float, default=0.5,
                        help="adam: decay of first order momentum of gradient")
    parser.add_argument("--b2", type=float, default=0.999,
                        help="adam: decay of first order momentum of gradient")
    parser.add_argument("--latent_dim", type=int, default=100,
                        help="dimensionality of the latent space")
    parser.add_argument("--img_size", type=int, default=64,
                        help="size of each image dimension")
    parser.add_argument("--channels", type=int, default=3,
                        help="number of image channels (If set to 1, convert image to grayscale)")
    parser.add_argument("--n_critic", type=int, default=5,
                        help="number of training steps for "
                             "discriminator per iter")
    parser.add_argument("--sample_interval", type=int, default=400,
                        help="interval betwen image samples")
    parser.add_argument("--seed", type=int, default=None,
                        help="value of a random seed")
    parser.add_argument("--dataset_size", type=int, default=1,
                        help="divides dataset by n")
    opt = parser.parse_args()

    main(opt)
