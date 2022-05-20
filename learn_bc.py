import numpy as np
import torch
from torch.utils.data import Dataset
import network
from botbowl.ai.env import BotBowlEnv, EnvConf
from timm.optim import create_optimizer_v2


class BCDataset(Dataset):
    def __init__(self, file):
        self.data = np.load(file)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def main():
    env = env = BotBowlEnv(EnvConf(size=11, pathfinding=True))
    spat_obs, non_spat_obs, action_mask = env.reset()
    spatial_obs_space = spat_obs.shape
    non_spatial_obs_space = non_spat_obs.shape[0]
    action_space = len(action_mask)
    del env, non_spat_obs, action_mask  # remove from scope to avoid confusion further down
    net = network.MimicBotXNet(spatial_shape=spatial_obs_space, non_spatial_inputs=non_spatial_obs_space,
                               actions=action_space)
    dataset = BCDataset('/data/s3092593/mgai/data.npy')
    trainloader = torch.utils.data.DataLoader(dataset, batch_size=128,
                                              shuffle=False, num_workers=4)

    criterion = torch.nn.MSELoss()
    optimizer = create_optimizer_v2(net, opt='lamb')
    epochs = 5

    for epoch in range(epochs):
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data

            optimizer.zero_grad()

            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if i % 2000 == 1999:  # print every 2000 mini-batches
                print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}')
                running_loss = 0.0

    torch.save(net, '/data/s3092593/mgai/net_bc')

