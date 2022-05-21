import numpy as np
import torch
from torch.utils.data import Dataset
import network
from botbowl.ai.env import BotBowlEnv, EnvConf
import torch.nn.functional as F
from timm.optim import create_optimizer_v2


class BCDataset(Dataset):
    def __init__(self, file, spatial):
        print('Loading...')
        self.data = np.load(file, allow_pickle=True)
        self.env = BotBowlEnv(EnvConf(size=11, pathfinding=True))
        print('Dloader is ready', self.data.shape)
        self.spatial = spatial

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # print(self.data[idx])
        spatial_obs, non_spatial_obs, action_mask, act_id = self.data[idx]
        # if self.spatial:
            # return torch.Tensor(spatial_obs), actions
        # else:
            # return torch.Tensor(non_spatial), actions
        # print(action_mask.shape, spatial_obs.shape, non_spatial_obs.shape, act_id)
        return torch.FloatTensor(spatial_obs),  torch.FloatTensor(non_spatial_obs), \
               torch.LongTensor(action_mask),  torch.LongTensor([act_id])

def main():
    env = env = BotBowlEnv(EnvConf(size=11, pathfinding=True))
    spat_obs, non_spat_obs, action_mask = env.reset()
    spatial_obs_space = spat_obs.shape
    non_spatial_obs_space = non_spat_obs.shape[0]
    action_space = len(action_mask)
    del env, non_spat_obs, action_mask  # remove from scope to avoid confusion further down
    net = network.MimicBotXNet(spatial_shape=spatial_obs_space, non_spatial_inputs=non_spatial_obs_space,
                               actions=action_space)
    net.cuda()
    
    spatial_trainloader = torch.utils.data.DataLoader(BCDataset('/data/s3092593/mgai/data.npy', spatial=True), 
                                                      batch_size=256, shuffle=False, num_workers=4)
    # non_spatial_trainloader = torch.utils.data.DataLoader(BCDataset('/data/s3092593/mgai/data.npy', spatial=False), 
                                                        #   batch_size=256, shuffle=False, num_workers=4)                                  

    criterion = torch.nn.CrossEntropyLoss()
    criterion = criterion.cuda()
    optimizer = create_optimizer_v2(net, opt='lamb')
    epochs = 5

    for epoch in range(epochs):
        running_loss = 0.0
        for i, data in enumerate(spatial_trainloader, 0):
            # print(data)
            spatial_obs, non_spatial_obs, action_mask, act_id = data
            # spatial_inputs, labels = spatial
            # non_spatial_inputs, labels2 = non_spatial
            spatial_obs = spatial_obs.cuda()
            non_spatial_obs = non_spatial_obs.cuda()
            action_mask = action_mask.cuda()
            labels = act_id.cuda().flatten()
            # print(labels.shape)
            # labels = F.one_hot(labels.flatten(), action_space)
            # assert labels == labels2
            # print(spatial_inputs.get_device())
            optimizer.zero_grad()
            # import pdb; pdb.set_trace()
            outputs = net(spatial_obs, non_spatial_obs)[1]
            outputs = F.softmax(outputs, dim=1)
            # outputs = torch.argmax(outputs, dim=1)
            # print(outputs.type(), labels.type())
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if i % 100 == 99:  # print every 100 mini-batches
                print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 100:.3f}')
                running_loss = 0.0

    torch.save(net, '/data/s3092593/mgai/net_bc')

if __name__ == '__main__':
    main()
