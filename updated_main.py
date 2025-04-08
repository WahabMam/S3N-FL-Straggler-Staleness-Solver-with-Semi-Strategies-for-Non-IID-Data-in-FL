import copy
import gc
import random
import sys
import time
import torch
import numpy as np
from tqdm import tqdm
from statistics import mean
from torchinfo import summary
from collections import OrderedDict
from itertools import islice
from numpy.random import randint

from configurations import args_parser
import utils
import models

if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else 'cpu') # Wahab
    start_time = time.time()
    args = args_parser()
    textio, best_val_acc, path_best_model = utils.initializations(args)
    textio.cprint(str(args))

    # data
    train_data, test_loader = utils.data(args)
    input, output, train_data, val_loader = utils.data_split(train_data, len(test_loader.dataset), args)

    # model
    if args.model == 'mlp':
        global_model = models.FC2Layer(input, output)
    elif args.model == 'cnn2':
        global_model = models.CNN2Layer(input, output, args.data)
    elif args.model[:3] == 'VGG':
        if args.data != 'cifar10':
            raise AssertionError('for VGG data must be cifar10')
        global_model = models.VGG(args.model)
    else:
        AssertionError('invalid model')
    textio.cprint(str(summary(global_model, verbose=0)))
    # global_model.to(args.device)
    global_model.to(device=device)

    train_creterion = torch.nn.CrossEntropyLoss(reduction='mean')
    test_creterion = torch.nn.CrossEntropyLoss(reduction='sum')

    # learning curve
    train_loss_list = []
    val_acc_list = []

    #  inference
    if args.eval:
        global_model.load_state_dict(torch.load(path_best_model))
        test_acc = utils.test(test_loader, global_model, test_creterion, device) # --> Wahab
        textio.cprint(f'eval test_acc: {test_acc:.0f}%')
        gc.collect()
        sys.exit()

    local_models = utils.federated_setup(global_model, train_data, args)

    # stragglers
    num_of_layers = global_model.state_dict().keys().__len__()
    if args.stragglers is not None:
        stragglers_idx = random.sample(range(args.num_users), round(args.stragglers_percent * args.num_users))
    else:
        stragglers_idx = []
    
    num_stragglers = len(stragglers_idx)  # Number of straggler clients WM
    # local_rounds = 5  # Number of local rounds for stragglers in the ring WM
    print(f"*********** number of straggler are:{num_stragglers} ***********************")

    for global_epoch in tqdm(range(0, args.global_epochs)):
        print(f"Global Epochs---------------------> {global_epoch}")
        round_start_time = time.time()  # Start time for the current round ----> Added by WM
        utils.distribute_model(local_models, global_model)
        users_loss = [] 
        # Handle Powerful Clients (Synchronous Training)
        for user_idx in range(args.num_users):
            print(f"********************* all clients, user_idx: {user_idx}")
            if (args.stragglers == 'drop') & (user_idx in stragglers_idx):
                user_new_state_dict = copy.deepcopy(global_model).state_dict()
                user_new_state_dict.update({})  # Empty update for dropped stragglers
                local_models[user_idx]['model'].load_state_dict(user_new_state_dict)
                continue

            user_loss = []
            for local_epoch in range(0, args.local_epochs):
                user = local_models[user_idx]
                # print(f"******************* user ----------------> {user}")
                # print(f"------------------------> user index: {user_idx}")
                print(f"----------------->  Training user {user_idx} for {local_epoch} local epochs")

                # train_loss = utils.train_one_epoch(user['data'], user['model'], user['opt'],
                #                                    train_creterion, args.device, args.local_iterations)
                
                # Train the model for the current client (non-straggler client) #WM

                train_loss = utils.train_one_epoch(user['data'], user['model'], user['opt'],
                                                   train_creterion, device, args.local_iterations) # Wahab
                user_loss.append(train_loss)
            #----> here we need to call FedAvg() for doing aggregation for powerfull cleints.
            if user_idx not in stragglers_idx:
                utils.FedAvg(local_models, global_model)

            if (args.stragglers == 'salf') & (user_idx in stragglers_idx):
                # need to add local epochs for stragglers
                # need to create ring topology to, in which the model will be aggregated and running amongst the nodes for specified local rounds.
                # after finishing the local rounds, one node will send trined model to the server.

                num_local_rounds = args.local_Rounds_topology  # Number of rounds stragglers train locally
                print(f"the topology train for {num_local_rounds} Rounds")
                straggler_idx_in_ring = stragglers_idx.index(user_idx)
                # print("************** stragglers_idx is : {stragglers_idx} and straggler_idx_in_ring  is : {straggler_idx_in_ring}")
  
                next_straggler_idx = stragglers_idx[(straggler_idx_in_ring + 1) % len(stragglers_idx)] # % is for the ring purpose, Modulus (%) ensures that when you reach the last straggler, you loop back to the first one (i.e., it forms a ring).
                # print(f"*next straggler index: {next_straggler_idx}")
                for local_round in range(num_local_rounds):
                    train_loss = utils.train_one_epoch(user['data'], user['model'], user['opt'], train_creterion, device, args.local_iterations)

                    current_state = user['model'].state_dict()
                    next_state = local_models[next_straggler_idx]['model'].state_dict()

                    for key in current_state:
                        next_state[key] = (current_state[key] + next_state[key]) / 2
                    local_models[next_straggler_idx]['model'].load_state_dict(next_state)


                # user_new_state_dict = copy.deepcopy(global_model).state_dict()
                # if args.up_to_layer is not None:
                #     up_to_layer = num_of_layers - args.up_to_layer  # last-to-first layers updated
                # else:
                #     up_to_layer = np.random.randint(1, num_of_layers + 1)  # random last-to-first layers updated
                #     # print(f" ***************** totla number of layers: {num_of_layers} ************** ")
                #     # print(f"***************** Randomly skipping {len(str(up_to_layer))} **********************")

                # user_updated_layers = OrderedDict(islice(reversed(user['model'].state_dict().items()), up_to_layer))
                # user_new_state_dict.update(user_updated_layers)
                # user['model'].load_state_dict(user_new_state_dict)
            try:
                users_loss.append(mean(user_loss))
            except:
                continue
        try:
            train_loss = mean(users_loss)
        except:
            train_loss = 0
        utils.FedAvg(local_models, global_model)

        val_acc = utils.test(val_loader, global_model, test_creterion, device)
        train_loss_list.append(train_loss)
        val_acc_list.append(val_acc)

        gc.collect()

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(global_model.state_dict(), path_best_model)

        textio.cprint(f'epoch: {global_epoch} | train_loss: {train_loss:.2f} | val_acc: {val_acc:.0f}%')

    np.save(f'checkpoints/{args.exp_name}/train_loss_list.npy', train_loss_list)
    np.save(f'checkpoints/{args.exp_name}/val_acc_list.npy', val_acc_list)


    elapsed_min = (time.time() - start_time) / 60
    textio.cprint(f'total execution time: {elapsed_min:.0f} min')
