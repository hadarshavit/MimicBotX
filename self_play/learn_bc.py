import numpy as np
import torch
from torch.utils.data import Dataset
import network
from botbowl.ai.env import BotBowlEnv, EnvConf
import torch.nn.functional as F
from timm.optim import create_optimizer_v2
from timm.scheduler import create_scheduler
from timm.utils import NativeScaler
from a2c_agent import CNNPolicy
import timm
import os
import generate_bc
from concurrent.futures import ProcessPoolExecutor

# executor = ProcessPoolExecutor(max_workers=1)


class BCDataset(Dataset):
    def __init__(self, train):
        self.executor = ProcessPoolExecutor(max_workers=10)
        self.futures = []

        self.data = []

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # print(self.files[idx])
        # print(self.data[idx])
        spatial_obs, non_spatial_obs, action_mask, act_id = self.data[idx]
        # if self.spatial:
            # return torch.Tensor(spatial_obs), actions
        # else:
            # return torch.Tensor(non_spatial), actions
        # print(action_mask.shape, spatial_obs.shape, non_spatial_obs.shape, act_id)
        return torch.FloatTensor(spatial_obs),  torch.FloatTensor(non_spatial_obs), \
                torch.LongTensor(action_mask),  torch.LongTensor([act_id])
    
    def generate_data(self):
        print('generating')
        for _ in range(20):
            self.futures.append(self.executor.submit(generate_bc.main))
    
    def collect_data(self):
        print('collecting')
        self.data = []
        for f in self.futures:
            self.data += f.result().buffer
        self.futures = []
        # if len(self.data) > 250_000:
        # self.data = self.data[-50_000:]
        print(len(self.data))

def main(activation, block, optimizer, lr, scheduler, epochs, save_id):
    env = env = BotBowlEnv(EnvConf(size=11, pathfinding=True))
    spat_obs, non_spat_obs, action_mask = env.reset()
    spatial_obs_space = spat_obs.shape
    non_spatial_obs_space = non_spat_obs.shape[0]
    action_space = len(action_mask)
    del env, non_spat_obs, action_mask  # remove from scope to avoid confusion further down
    net = network.MimicBotXNet(spatial_shape=spatial_obs_space, non_spatial_inputs=non_spatial_obs_space,
                               actions=action_space, drop_rate=0.0, activation=activation, block=block, drop_path=0.0)
    # net = CNNPolicy(spatial_shape=spatial_obs_space, non_spatial_inputs=non_spatial_obs_space, 
                    # actions=action_space, hidden_nodes = 2048, kernels = [128, 128])
    net.cuda()
    net.to(memory_format=torch.channels_last)

    amp_autocast = torch.cuda.amp.autocast
    loss_scaler = NativeScaler()
    # lr_scheduler, num_epochs = create_scheduler(args, optimizer)
    # start_epoch = 0
    # lr_scheduler.step(start_epoch)
    # data = np.load('/local/s3092593/data.npy', allow_pickle=True)


    # train_dataset = BCDataset(train=True)
    # train_dataset.generate_data()
    # train_dataset.collect_data()


    # train_dataset.generate_data()

    # validation_dataset = BCDataset(train=False)
    # validation_dataset.generate_data()
    # validation_dataset.collect_data()
    # validation_dataset.data = validation_dataset.data[-50_000:]


    # validation_dataset.generate_data()
    # train_loader = torch.utils.data.DataLoader(train_dataset, 
    #                                                   batch_size=512, shuffle=False, num_workers=8)
    # validation_loader = torch.utils.data.DataLoader(validation_dataset, 
    #                                                       batch_size=512, shuffle=False, num_workers=8)     
    

    # criterion = torch.nn.CrossEntropyLoss()
    # criterion = criterion.cuda()

    optimizer = create_optimizer_v2(net.parameters(), optimizer, lr=lr)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=30) if scheduler else None
    best_loss = 10000
    best_epoch = 0
    for epoch in range(epochs):
        running_loss = 0.0
        train_dataset.generate_data()
        for i, data in enumerate(train_loader, 0):
            spatial_obs, non_spatial_obs, action_mask, act_id = data
            spatial_obs = spatial_obs.cuda()
            non_spatial_obs = non_spatial_obs.cuda()
            labels = act_id.cuda().flatten()

            spatial_obs = spatial_obs.contiguous(memory_format=torch.channels_last)
            optimizer.zero_grad()

            with amp_autocast():
                outputs = net(spatial_obs, non_spatial_obs)[1]
                loss = criterion(outputs, labels)
                running_loss += loss
                loss_scaler(loss, optimizer)

            if i % 100 == 0:
                print(f'Train [{epoch + 1}, {i + 1:5d}] loss: {running_loss/(i+1):.3f}')
                if torch.isnan(running_loss):
                    return 10000
        
        print(f'Train [{epoch + 1}, {i + 1:5d}] loss: {running_loss/(i+1):.3f}')
        
        net.eval()
        running_loss = 0
        with torch.no_grad():
            for i, data in enumerate(validation_loader, 0):
                spatial_obs, non_spatial_obs, action_mask, act_id = data

                spatial_obs = spatial_obs.cuda()
                non_spatial_obs = non_spatial_obs.cuda()
                labels = act_id.cuda().flatten()

                spatial_obs = spatial_obs.contiguous(memory_format=torch.channels_last)
                # non_spatial_obs = non_spatial_obs.contiguous(memory_format=torch.channels_last)
                with amp_autocast():
                    outputs = net(spatial_obs, non_spatial_obs)[1]
                    loss = criterion(outputs, labels)
                    running_loss += loss

        print(f'Validate [{epoch + 1}, {i + 1:5d}] loss: {running_loss/(i+1):.3f}')
        if running_loss < best_loss:
            best_loss = running_loss
            best_epoch = epoch
            if torch.isnan(running_loss):
                    return 10000
            torch.save(net, f'/data/s3092593/mgai/net_good{epoch}_{save_id}')
        running_loss = 0
        net.train()

        
        train_dataset.collect_data()

        if scheduler:
            scheduler.step()
    print(f'{best_loss=}, {best_epoch=}')
    train_dataset.executor.shutdown()
    validation_dataset.executor.shutdown()
    del train_dataset
    del validation_dataset
    return best_loss.cpu().item()
    # torch.save(net, '/data/s3092593/mgai/net_bc')

if __name__ == '__main__':
    main()
