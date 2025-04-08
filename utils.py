import math
import os
import random
import time

from matplotlib import pyplot as plt
from scipy.interpolate import interp1d
import torch
import copy
import numpy as np
from torch import optim
from statistics import mean
from torchvision import datasets, transforms


def FedAvg(local_models, global_model):
    state_dict = global_model.state_dict()
    for key in state_dict.keys():
        local_weights_sum = torch.zeros_like(state_dict[key])
        count_updating_users = 0
        for user_idx in local_models.keys():
            if key in local_models[user_idx]['model'].state_dict():
                local_weights_sum += local_models[user_idx]['model'].state_dict()[key]
                count_updating_users += 1
        if count_updating_users != 0:
            state_dict[key] = (local_weights_sum / len(local_models)).to(state_dict[key].dtype)

    global_model.load_state_dict(state_dict)
    return

def partition_data_min( targets, total_clients, dir_alpha, min_samples=10):
    """Partition data using Dirichlet distribution, ensuring each client has at least min_samples."""    
    num_classes = np.max(targets) + 1
    n_samples = len(targets)
    
    # Check feasibility
    if total_clients * min_samples > n_samples:
        raise ValueError(f"Cannot assign {min_samples} samples to each of {total_clients} clients with only {n_samples} total samples")
    
    client_indices = [[] for _ in range(total_clients)]
    
    # Step 1: Assign minimum samples to each client
    all_indices = np.arange(n_samples)
    np.random.shuffle(all_indices)
    min_assignment = np.array_split(all_indices[:total_clients * min_samples], total_clients)
    for client_idx, indices in enumerate(min_assignment):
        client_indices[client_idx].extend(indices)
    
    # Remaining indices to distribute
    remaining_indices = all_indices[total_clients * min_samples:]
    remaining_targets = targets[remaining_indices]
    
    # Step 2: Distribute remaining samples using Dirichlet
    if len(remaining_indices) > 0:
        for c in range(num_classes):
            idx_c = np.where(remaining_targets == c)[0]
            if len(idx_c) > 0:
                np.random.shuffle(idx_c)
                proportions = np.random.dirichlet(np.repeat(dir_alpha, total_clients))
                split_points = (np.cumsum(proportions) * len(idx_c)).astype(int)[:-1]
                client_splits = np.split(idx_c, split_points)
                for i, split in enumerate(client_splits):
                    client_indices[i].extend(remaining_indices[split])
    
    # Shuffle each client's indices
    for i in range(total_clients):
        np.random.shuffle(client_indices[i])
    
    # Verify minimum samples
    for i, indices in enumerate(client_indices):
        if len(indices) < min_samples:
            raise RuntimeError(f"Client {i} has {len(indices)} samples, less than required {min_samples}")
    
    return client_indices


def federated_setup(global_model, train_data, args):
    # create a dict of dict s (local users), i.e. {'1': {'data':..., 'model':..., 'opt':...}, ...}
    local_models = {}
    if args.dir_alpha > 0:
        #handle noniid data 
        # Split data using Dirichlet distribution
        targets = np.array(train_data.targets if hasattr(train_data, 'targets') else train_data.labels)
        client_indices = partition_data_min( targets = targets, total_clients = args.num_users, dir_alpha = args.dir_alpha)

        for user_idx in range(args.num_users):
            user = {'data': torch.utils.data.DataLoader(
                torch.utils.data.Subset(train_data,
                                        client_indices[user_idx]),
                batch_size=args.train_batch_size, shuffle=True),
                'model': copy.deepcopy(global_model)}
            user['opt'] = optim.SGD(user['model'].parameters(), lr=args.lr,
                                    momentum=args.momentum) if args.optimizer == 'sgd' \
                else optim.Adam(user['model'].parameters(), lr=args.lr)
            user['cid'] = user_idx
            local_models[user_idx] = user
    else:
        indexes = torch.randperm(len(train_data))
        user_data_len = math.floor(len(train_data) / args.num_users) if args.num_samples == None else args.num_samples
        for user_idx in range(args.num_users):
            user = {'data': torch.utils.data.DataLoader(
                torch.utils.data.Subset(train_data,
                                        indexes[user_idx * user_data_len:(user_idx + 1) * user_data_len]),
                batch_size=args.train_batch_size, shuffle=True),
                'model': copy.deepcopy(global_model)}
            user['opt'] = optim.SGD(user['model'].parameters(), lr=args.lr,
                                    momentum=args.momentum) if args.optimizer == 'sgd' \
                else optim.Adam(user['model'].parameters(), lr=args.lr)
            user['cid'] = user_idx
            local_models[user_idx] = user
    
    return local_models


def distribute_model(local_models, global_model):
    for user_idx in local_models.keys():
        local_models[user_idx]['model'].load_state_dict(global_model.state_dict())


class IOStream:
    def __init__(self, path):
        self.f = open(path, 'a')

    def cprint(self, text):
        print(text)
        self.f.write(text + '\n')
        self.f.flush()

    def close(self):
        self.f.close()


def initializations(args):
    #  reproducibility
    torch.backends.cudnn.deterministic = True
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    #  documentation
    if not os.path.exists('checkpoints'):
        os.makedirs('checkpoints')
    if not os.path.exists('checkpoints/' + args.exp_name):
        os.makedirs('checkpoints/' + args.exp_name)
    textio = IOStream('checkpoints/' + args.exp_name + '/run.log')

    best_val_acc = np.inf
    path_best_model = 'checkpoints/' + args.exp_name + '/model.best.t7'

    return textio, best_val_acc, path_best_model


def data(args):
    if args.data == 'mnist':
        train_data = datasets.MNIST('./data', train=True, download=True,
                                    transform=transforms.Compose([
                                        transforms.ToTensor(),
                                        transforms.Normalize((args.norm_mean,), (args.norm_std,))
                                    ]))

        test_loader = torch.utils.data.DataLoader(
            datasets.MNIST('./data', train=False, transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((args.norm_mean,), (args.norm_std,))
            ])),
            batch_size=args.test_batch_size, shuffle=False)
    else:
        train_data = datasets.CIFAR10('./data', train=True, download=True,
                                      transform=transforms.Compose([
                                          transforms.ToTensor(),
                                          transforms.Normalize((args.norm_mean,), (args.norm_std,))
                                      ]))

        test_loader = torch.utils.data.DataLoader(
            datasets.CIFAR10('./data', train=False, transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((args.norm_mean,), (args.norm_std,))
            ])),
            batch_size=args.test_batch_size, shuffle=False)
    return train_data, test_loader


def data_split(data, amount, args, test_loader):
    # split train, validation
    # train_data, val_data = torch.utils.data.random_split(data, [len(data) - amount, amount])
    # val_loader = torch.utils.data.DataLoader(val_data, batch_size=args.test_batch_size, shuffle=False)

    # input, output sizes
    in_channels, dim1, dim2 = data[0][0].shape  # images are dim1 x dim2 pixels
    input = dim1 * dim2 if args.model == 'mlp' or args.model == 'linear' else in_channels
    output = len(data.classes)  # number of classes

    return input, output, data, test_loader


def train_one_epoch(train_loader, model, optimizer,
                    creterion, device, iterations, user_idx):
    total_batches = len(train_loader)
    model.train()
    losses = []
    # if iterations is not None:
    #     local_iteration = 0
    for batch_idx, (data, label) in enumerate(train_loader):
        # print(f"Client : {user_idx} Training batch {batch_idx}/{total_batches}")
        # send to device
        data, label = data.to(device), label.to(device)
        output = model(data)
        loss = creterion(output, label)

        optimizer.zero_grad()
        loss.backward()
        start = time.time()
        optimizer.step()
        nat = (time.time() - start) / 60

        losses.append(loss.item())

        # if iterations is not None:
        #     local_iteration += 1
        #     if local_iteration == iterations:
        #         break
    return mean(losses)


def test(test_loader, model, creterion, device):
    model.eval()
    test_loss = 0
    correct = 0
    for data, label in test_loader:
        data, label = data.to(device), label.to(device)  # send to device

        output = model(data)
        test_loss += creterion(output, label).item()  # sum up batch loss
        pred = output.data.max(1, keepdim=True)[1]  # get the index of the max log-probability
        correct += pred.eq(label.data.view_as(pred)).cpu().sum().item()

    test_loss /= len(test_loader.dataset)
    accuracy = 100. * correct / len(test_loader.dataset)
    return accuracy