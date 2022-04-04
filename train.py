import torch
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from CelebADataSet import CelebADataSet
from DSpritesDataSet import DSpritesDataSet
from FERGDataSet import FERGDataSet
from ChairsDataSet import ChairsDataSet
import random
import numpy as np
from tqdm import tqdm
import math
from datetime import datetime
import os
from FlowerDataSet import FlowerDataSet

from model import *

def reconstruction_loss(original_img, reconstructions, distribution):
    batch_size = original_img.size(0)
    if distribution == 'bernoulli':
        recon_loss = F.binary_cross_entropy_with_logits(reconstructions,
                                                        original_img,
                                                        reduction='sum').div(batch_size)
    elif distribution == 'gaussian':
        recon_sig = F.sigmoid(reconstructions)
        recon_loss = F.mse_loss(recon_sig, original_img, reduction='sum').div(batch_size)
        # recon_loss = F.mse_loss(reconstructions * 255,
        #                         original_img * 255,
        #                         reduction="sum") / 255
    else:
        recon_loss = None
    return recon_loss

def kl_divergence_loss(mu, log_var):
    return torch.sum(-0.5 * (1 + log_var - mu ** 2 - log_var.exp()).sum(dim=0))


def main():
    # set random seed
    seed = 0
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    # dataset name:
    # dataset_name = 'CelebA'
    # dataset_name = 'DSprites'
    dataset_name = 'Faces2'

    # TODO Normalize transform
    if dataset_name == 'CelebA' or dataset_name == 'Flower' or dataset_name == "Chairs" or dataset_name == 'FERG':
        transform = transforms.Compose(
                    [transforms.ToTensor(),
                    transforms.Resize((64, 64))
                    ]) # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    elif dataset_name == 'DSprites':
        transform = transforms.Compose([])
    else:
        transform = transforms.Compose([])

    # config:
    num_epochs = 10
    batch_size = 64
    trainset_percentage = 0.8
    betas = [1, 5, 25, 50, 100, 250, 500]

    # CelebA betas done: 1, 5, 25, 50
    # DSprites betas done: 1, 5

    # progress bar
    pb_len = 75

    # Load dataset
    if dataset_name == 'CelebA':
        dataset = CelebADataSet(root_path='datasets/CelebA', transform=transform)
    elif dataset_name == 'Flower':
        dataset = FlowerDataSet(root_path='datasets/flower_data', transform=transform)
    elif dataset_name == 'Chairs':
        dataset = ChairsDataSet(root_path='datasets/rendered_chairs', transform=transform)
    elif dataset_name == 'DSprites':
        dataset = DSpritesDataSet()
    elif dataset_name == 'FERG':
        dataset =  FERGDataSet(transform=transform)
    else:
        dataset = None

    # Train-Test split
    dataset_size = len(dataset)
    trainset_size = int(dataset_size * trainset_percentage)
    trainset, testset = random_split(dataset,
                                    [trainset_size, dataset_size - trainset_size],
                                    generator=torch.Generator().manual_seed(42))

    # Train loader:
    trainloader = DataLoader(trainset, batch_size=batch_size,
                            shuffle=True, num_workers=2)

    # Test data:
    testloader = DataLoader(testset, batch_size=batch_size,
                            shuffle=False, num_workers=2)

    # Set device
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


    for beta in betas:
        print(f'\n#------ Beta={beta} ------#')
        model_dir = f'models/{dataset_name}/beta_{beta}/'

        # Model initialization
        if dataset_name == 'CelebA'or dataset_name == 'Flower' or dataset_name == 'Chairs':
            model = CNNBetaVAE(latent_dim=32, in_channels=3)
        elif dataset_name == 'FERG':
            model = CNNBetaVAE(latent_dim=32, in_channels=3)
        elif dataset_name == 'DSprites':
            model = FCBetaVAE(latent_dim=10, input_dim=4096)
        else:
            model = None
        model = model.to(device)

        # Optimizer
        if dataset_name == 'CelebA' or dataset_name == 'Flower'or dataset_name == 'Chairs':
            optimizer = optim.Adam(model.parameters(), lr=1e-4)
        elif dataset_name == 'FERG':
            optimizer = optim.Adagrad(model.parameters(), lr=1e-2)
        elif dataset_name == 'DSprites':
            optimizer = optim.Adagrad(model.parameters(), lr=1e-2)
        else:
            optimizer = None

        # Loss KL-distribution
        if dataset_name == 'CelebA':
            distribution='gaussian'
        elif dataset_name == 'Chairs':
            distribution='bernoulli'
        elif dataset_name == 'FERG':
            distribution='gaussian'
        elif dataset_name == 'Flower':
            distribution='gaussian'
        elif dataset_name == 'DSprites':
            distribution='bernoulli'
        else:
            distribution = None

        # Training loop
        losses = []
        for epoch in range(num_epochs):  # loop over the dataset multiple times
            t1 = datetime.now()
            model.train()
            num_batches = len(trainloader)
            current_batch = 0
            for i, data in enumerate(trainloader, 0):
                current_batch += 1
                # get the inputs; data is a list of [inputs, labels]
                images = data[0].to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                outputs, mu, log_var = model(images)

                # Loss calculation
                loss = reconstruction_loss(original_img=images, reconstructions=outputs, distribution=distribution) \
                        + beta * kl_divergence_loss(mu, log_var)

                # backward
                loss.backward()

                # optimizer step
                optimizer.step()
                
                # save loss value
                losses.append(loss.item())

                # print progress bar
                t2 = datetime.now()
                time_diff = str(t2 - t1)[2:-4]
                bars=min(math.ceil(current_batch/num_batches*pb_len), pb_len-1)
                print(f'\r[{"="*bars}>{" "*(pb_len-bars-1)}] epoch: {epoch+1} | time: {time_diff} | loss: {loss.item()}', end='')
            print()
            # save model:
            os.makedirs(model_dir, exist_ok=True)
            model_save_path = f'{model_dir}epoch_{epoch+1}.pth'
            torch.save(model.state_dict(), model_save_path)  
            np.save(f'{model_dir}loss_values.npy', losses)

        print('Finished Training')


if __name__ == '__main__':
    main()