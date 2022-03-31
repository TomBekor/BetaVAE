from DSpritesDataSet import DSpritesDataSet
import matplotlib.pyplot as plt
import numpy as np
from DisentanglementDataSet import DisentanglementDataSet
from model import LinearClassifier
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from DSpritesDataSet import DSpritesDataSet
import numpy as np
import math
from datetime import datetime


def main():
    model_path = 'models/DSprites/beta_250/epoch_7.pth'

    dsprites_dataset = DSpritesDataSet()
    disentanglement_dataset = DisentanglementDataSet(dsprites_dataset=dsprites_dataset, dataset_size=1000, L=64, model_path=model_path)

    num_epochs = 1000
    batch_size = 64
    trainset_percentage = 0.8

    # Train-Test split
    dataset_size = len(disentanglement_dataset)
    trainset_size = int(dataset_size * trainset_percentage)
    trainset, testset = random_split(disentanglement_dataset,
                                    [trainset_size, dataset_size - trainset_size],
                                    generator=torch.Generator().manual_seed(42))

    # Train loader:
    trainloader = DataLoader(trainset, batch_size=batch_size,
                            shuffle=True)

    # Test data:
    testloader = DataLoader(testset, batch_size=batch_size,
                            shuffle=False)

    # Set device
    # device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    device = torch.device('cpu')

    model = LinearClassifier(input_dim=10, output_dim=5).to(device)

    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=1e-2)

    # Loss function
    criterion = nn.CrossEntropyLoss()


    # Training loop
    pb_len=30
    losses = []
    epoch_accs = []
    for epoch in range(num_epochs):  # loop over the dataset multiple times
        t1 = datetime.now()
        model.train()
        num_batches = len(trainloader)
        current_batch = 0
        for i, data in enumerate(trainloader, 0):
            current_batch += 1
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data[0].to(device), data[1].to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward
            outputs = model(inputs)

            # Loss calculation
            loss = criterion(outputs, labels)

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

        # calc accuracy
        correct = 0
        total = 0
        model.eval()
        with torch.no_grad():
            for i, data in enumerate(testloader, 0):
                inputs, labels = data[0].to(device), data[1].to(device)
                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        epoch_accs.append(round(100 * correct / total, 2))
    print('Finished Training')
    print(f'Dinsentangle Metric Score: {epoch_accs[-1]}%')


if __name__ == '__main__':
    main()