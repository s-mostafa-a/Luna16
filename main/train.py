import torch
import numpy as np
import time
import os
from model.net import Net
from model.loss import Loss
from torch.autograd import Variable
import itertools
import pandas as pd
from main.dataset import LunaDataSet
from torch.utils.data import DataLoader
from configs import VAL_PCT, TOTAL_EPOCHS, DEFAULT_LR, OUTPUT_PATH


def get_lr(epoch):
    if epoch <= TOTAL_EPOCHS * 0.5:
        lr = DEFAULT_LR
    elif epoch <= TOTAL_EPOCHS * 0.8:
        lr = 0.1 * DEFAULT_LR
    else:
        lr = 0.01 * DEFAULT_LR
    return lr


def train(data_loader, net, loss, epoch, optimizer, get_lr, save_dir='./models/'):
    start_time = time.time()

    net.train()
    lr = get_lr(epoch)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    metrics = []
    for i, (data, target, coord) in enumerate(data_loader):
        if torch.cuda.is_available():
            data = Variable(data.cuda())
            target = Variable(target.cuda())
            coord = Variable(coord.cuda())
        data = data.float()
        target = target.float()
        coord = coord.float()

        output = net(data, coord)
        loss_output = loss(output, target)
        optimizer.zero_grad()
        loss_output[0].backward()
        optimizer.step()

        loss_output[0] = loss_output[0].item()
        metrics.append(loss_output)
        break
    if epoch % 10 == 0:
        state_dict = net.state_dict()
        for key in state_dict.keys():
            state_dict[key] = state_dict[key].cpu()
        torch.save({
            'epoch': epoch,
            'save_dir': save_dir,
            'state_dict': state_dict}, os.path.join(save_dir, f'''{epoch}.ckpt'''))

    end_time = time.time()

    metrics = np.asarray(metrics, np.float32)
    print(f'''Epoch {epoch} (lr {lr})''')
    print(f'''Train: tpr {100.0 * np.sum(metrics[:, 6]) / np.sum(metrics[:, 7])},
            tnr {100.0 * np.sum(metrics[:, 8]) / np.sum(metrics[:, 9])}, 
            total pos {np.sum(metrics[:, 7])}, total neg {np.sum(metrics[:, 9])}, 
            time {end_time - start_time}''')
    print(f'''loss {np.mean(metrics[:, 0])}, classify loss {np.mean(metrics[:, 1])},
            regress loss {np.mean(metrics[:, 2])}, {np.mean(metrics[:, 3])}, 
            {np.mean(metrics[:, 4])}, {np.mean(metrics[:, 5])}''')


def validate(data_loader, net, loss):
    start_time = time.time()

    net.eval()

    metrics = []
    for i, (data, target, coord) in enumerate(data_loader):
        if torch.cuda.is_available():
            data = Variable(data.cuda())
            target = Variable(target.cuda())
            coord = Variable(coord.cuda())
        data = data.float()
        target = target.float()
        coord = coord.float()

        output = net(data, coord)
        loss_output = loss(output, target, train=False)

        loss_output[0] = loss_output[0].item()
        metrics.append(loss_output)
    end_time = time.time()

    metrics = np.asarray(metrics, np.float32)
    print(f'''time {end_time - start_time}''')
    print(f'''loss {np.mean(metrics[:, 0])}, classify loss {np.mean(metrics[:, 1])},
            regress loss {np.mean(metrics[:, 2])}, {np.mean(metrics[:, 3])}, 
            {np.mean(metrics[:, 4])}, {np.mean(metrics[:, 5])}''')


if __name__ == '__main__':
    neural_net = Net()
    loss_fn = Loss()
    if torch.cuda.is_available():
        neural_net = neural_net.cuda()
        loss_fn = loss_fn.cuda()
    optim = torch.optim.SGD(
        neural_net.parameters(),
        DEFAULT_LR,
        momentum=0.9,
        weight_decay=1e-4)
    meta = pd.read_csv(f'{OUTPUT_PATH}/meta.csv', index_col=0).sample(frac=1).reset_index(drop=True)
    meta_group_by_series = meta.groupby(['seriesuid']).indices
    list_of_groups = [{i: list(meta_group_by_series[i])} for i in meta_group_by_series.keys()]
    np.random.shuffle(list_of_groups)
    val_split = int(VAL_PCT * len(list_of_groups))
    val_indices = list(itertools.chain(*[list(i.values())[0] for i in list_of_groups[:val_split]]))
    train_indices = list(itertools.chain(*[list(i.values())[0] for i in list_of_groups[val_split:]]))
    ltd = LunaDataSet(train_indices, meta)
    lvd = LunaDataSet(val_indices, meta)
    train_loader = DataLoader(ltd, batch_size=1, shuffle=False)
    val_loader = DataLoader(lvd, batch_size=1, shuffle=False)

    save_dir = f'{OUTPUT_PATH}/models/'
    os.makedirs(save_dir, exist_ok=True)
    for ep in range(TOTAL_EPOCHS):
        train(train_loader, neural_net, loss_fn, ep, optim, get_lr, save_dir=save_dir)
        validate(val_loader, neural_net, loss_fn)
