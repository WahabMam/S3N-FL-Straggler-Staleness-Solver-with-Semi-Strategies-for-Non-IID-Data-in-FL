import copy
import gc
import random
import sys
import time
import torch
import numpy as np
from tqdm import tqdm
from statistics import mean
# from torchinfo import summary
from collections import OrderedDict
from itertools import islice
from numpy.random import randint
from collections import OrderedDict, deque
import numpy as np 
import csv
import os
from datetime import datetime
import time

from configurations import args_parser
import utils
import models
import threading
from concurrent.futures import ThreadPoolExecutor
import concurrent.futures
import threading
# Create a shared event for signaling
stop_event = threading.Event()
import shutil



MODELS_DIR = './models_3'
SYNC = 10
if os.path.exists(MODELS_DIR):
    shutil.rmtree(MODELS_DIR,ignore_errors=True)



def average_models(model1, model2):
    """ Averages two PyTorch model state_dicts """
    averaged_state_dict = {}
    for key in model1.keys():
        averaged_state_dict[key] = (model1[key] + model2[key]) / 2  # Element-wise averaging
    return averaged_state_dict


def  train_centralized(fast_models,stragglers_idx, global_model,fast=True):
    global_model_fast = copy.deepcopy(global_model)
    textio.cprint(f'Training centralised fast models')
    best_val_acc = np.inf
    for global_epoch in range(0, args.global_epochs):
        if stop_event.is_set():
            return 
        textio.cprint(f"++++++++++++ [Centralized ] Round {global_epoch}/{args.global_epochs}")
        utils.distribute_model(fast_models, global_model_fast)
        users_loss = []

        for user_idx in fast_models.keys():
            if not fast:
                time.sleep(random.uniform(0.02, 0.1)) 
            if (args.stragglers == 'drop') & (user_idx in stragglers_idx):
                user_new_state_dict = copy.deepcopy(global_model_fast).state_dict()
                user_new_state_dict.update({})
                fast_models[user_idx]['model'].load_state_dict(user_new_state_dict)
                continue

            user_loss = []
            for local_epoch in range(0, args.local_epochs):
                
                user = fast_models[user_idx]
                train_loss = utils.train_one_epoch(user['data'], user['model'], user['opt'],
                                                   train_creterion, args.device, args.local_iterations,user_idx)
                user_loss.append(train_loss)

        
            try:
                users_loss.append(mean(user_loss))
            except:
                continue
        try:
            train_loss = mean(users_loss)
        except:
            train_loss = 0
        utils.FedAvg(fast_models, global_model_fast)

        #save the model
        if fast:
            save_path = os.path.join(MODELS_DIR,'cfl')
            if not os.path.exists(save_path):
                os.makedirs(save_path)
        else:
            save_path = os.path.join(MODELS_DIR,'dfl')
            if not os.path.exists(save_path):
                os.makedirs(save_path)
        torch.save(global_model_fast.state_dict(), os.path.join(save_path,f"model_round_{global_epoch}.pt"))

        

        if (global_epoch+1) % SYNC == 0:
            #synchronise the models after every SYNC rounds with the latest avelaible dfl model
            all_dfl_models = os.listdir(os.path.join(MODELS_DIR,'dfl'))
            all_dfl_models = sorted(all_dfl_models, key=lambda x: int(x.split('_')[2].replace('.pt', '')))
            latest_dfl_model = all_dfl_models[-1]
            latest_dfl_model = os.path.join(os.path.join(MODELS_DIR,'dfl',latest_dfl_model))
            global_model_fast.load_state_dict(average_models(global_model_fast.state_dict(),torch.load(latest_dfl_model)))
            textio.cprint(f"Updated current global model round {global_epoch} with  Dfl slow model : {latest_dfl_model}")
        
        val_acc = utils.test(val_loader, global_model_fast, test_creterion, args.device)
        train_loss_list.append(train_loss)
        val_acc_list.append(val_acc)
        

        gc.collect()

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(global_model_fast.state_dict(), path_best_model)

        textio.cprint(f'Fast clients => epoch: {global_epoch} | train_loss: {train_loss:.2f} | val_acc: {val_acc:.0f}%')
        print(f"****************-------> Fast Clients dfl=> epoch: {global_epoch} | train_loss: {train_loss:.2f} | val_acc: {val_acc:.0f}% ")

        if not fast:
            writer2.writerow([global_epoch, np.round(train_loss, 2), val_acc])
        else:
            writer.writerow([global_epoch, np.round(train_loss, 2), val_acc])

    val_acc_final_fast = utils.test(val_loader, global_model_fast, test_creterion, args.device)

    # writer.writerow([global_epoch, np.round(train_loss, 2), val_acc])

    textio.cprint(f'Fast Clients final acc => {val_acc_final_fast}')
    np.save(f'checkpoints/{args.exp_name}/train_loss_list_cen.npy', train_loss_list)
    np.save(f'checkpoints/{args.exp_name}/val_acc_list_cen.npy', val_acc_list)



def  train_decentralised(slow_models,stragglers_idx, global_model):
    global_model_slow = copy.deepcopy(global_model)
    textio.cprint(f'Training decentralised slow models')
    best_val_acc = np.inf
    client_ids = list(slow_models.keys())  # Get client IDs
    client_ring = deque(client_ids)  # Create a ring structure
    # print(f'********************************** Clietn_Ring: {client_ring}')

    for global_epoch in range(0, args.global_epochs):
        if stop_event.is_set():
            return 
        textio.cprint(f"++++++++++++ [DFL slow ] Round {global_epoch}/{args.global_epochs}")
        
        users_loss = []
        local_rounds_count = 0  # Initialize local rounds counter for the global round

        for user_idx in slow_models.keys():
            time.sleep(random.uniform(0.05, 0.3)) 
            
            if (args.stragglers == 'drop') & (user_idx in stragglers_idx):
                user_new_state_dict = copy.deepcopy(global_model).state_dict()
                user_new_state_dict.update({})
                fast_models[user_idx]['model'].load_state_dict(user_new_state_dict)
                continue

            user_loss = []
            for local_epoch in range(0, args.local_epochs):
                user = slow_models[user_idx]
                train_loss = utils.train_one_epoch(user['data'], user['model'], user['opt'],
                                                train_creterion, args.device, args.local_iterations)
                user_loss.append(train_loss)
                local_rounds_count += 1  # Increment local rounds counter for each local epoch

            try:
                users_loss.append(mean(user_loss))
            except:
                continue

        try:
            train_loss = mean(users_loss)
        except:
            train_loss = 0

        # Print the number of local rounds after processing the clients in the ring
        textio.cprint(f"Total Local Rounds in Global Round {global_epoch}: {local_rounds_count}")

        # **Decentralized Ring Communication Step with Averaging**
        client_ring.rotate(-1)  # Shift clients in the ring
        for i, user_idx in enumerate(client_ids):
            next_user_idx = client_ring[i]  # Get the next client in the ring
            # Get current and next client's models
            current_model_state = slow_models[user_idx]['model'].state_dict()
            next_model_state = slow_models[next_user_idx]['model'].state_dict()

            # Compute averaged model
            averaged_state_dict = average_models(current_model_state, next_model_state)

            # Update next user's model with the averaged model
            slow_models[next_user_idx]['model'].load_state_dict(averaged_state_dict)

        

        # up-date slow global model
        global_model_slow.load_state_dict(average_models(global_model_slow.state_dict(), slow_models[client_ids[0]]['model'].state_dict()))

        # Save the model
        save_path = os.path.join(MODELS_DIR, 'dfl')
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        torch.save(global_model_slow.state_dict(), os.path.join(save_path, f"model_round_{global_epoch}.pt"))

        if (global_epoch + 1) % SYNC == 0:
            # Synchronize the models after every 10 rounds with the latest available cfl model
            all_cfl_models = os.listdir(os.path.join(MODELS_DIR, 'cfl'))
            all_cfl_models = sorted(all_cfl_models, key=lambda x: int(x.split('_')[2].replace('.pt', '')))
            latest_cfl_model = all_cfl_models[-1]
            latest_cfl_model = os.path.join(os.path.join(MODELS_DIR, 'cfl', latest_cfl_model))
            global_model_slow.load_state_dict(average_models(global_model_slow.state_dict(), torch.load(latest_cfl_model)))
            textio.cprint(f"Updated current global model round {global_epoch} with cfl fast model: {latest_cfl_model}")

        gc.collect()

        val_acc = utils.test(val_loader, global_model_slow, test_creterion, args.device)
        train_loss_list.append(train_loss)
        val_acc_list.append(val_acc)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(global_model_slow.state_dict(), path_best_model)

        textio.cprint(f'Slow Clients dfl=> epoch: {global_epoch} | train_loss: {train_loss:.2f} | val_acc: {val_acc:.0f}%')
        writer2.writerow([global_epoch, np.round(train_loss, 2), val_acc])
        print(f"**********************-------> Slow Clients dfl=> epoch: {global_epoch} | train_loss: {train_loss:.2f} | val_acc: {val_acc:.0f}% ")

    # print(f"**********************-------> Slow Clients dfl=> epoch: {global_epoch} | train_loss: {train_loss:.2f} | val_acc: {val_acc:.0f}% ")
    val_acc_final_slow = utils.test(val_loader, global_model_slow, test_creterion, args.device)
    textio.cprint(f'Slow Clients final acc => {val_acc_final_slow}')
    np.save(f'checkpoints/{args.exp_name}/train_loss_list_dfl.npy', train_loss_list)
    np.save(f'checkpoints/{args.exp_name}/val_acc_list_dfl.npy', val_acc_list)

    # writer.writerow([global_epoch, train_loss, val_acc])
# file.close()


if __name__ == '__main__':
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
    elif args.model == 'LeNet':
        global_model = models.LeNet5(input, output)
    else:
        AssertionError('invalid model')
    # textio.cprint(str(summary(global_model, verbose=0)))
    global_model.to(args.device)

    train_creterion = torch.nn.CrossEntropyLoss(reduction='mean')
    test_creterion = torch.nn.CrossEntropyLoss(reduction='sum')
   

    # learning curve
    train_loss_list = []
    val_acc_list = []

    current_time = datetime.now().strftime("%Y-%m-%d_%I-%M-%p")  # Example: 2025-02-24_05-45-PM
    filename = f"MAK_log_{current_time}.csv"

    save_dir_1 = f"MAK3_logs/FIRST_Completed/{args.stragglers_percent}_strg_{SYNC}sync"
    os.makedirs(save_dir_1, exist_ok=True)
    # os.makedirs('MAK_logs/10_Rounds_0.5', exist_ok=True)
    # os.makedirs("experiment_logs/after_20_rounds", exist_ok=True)
    file_path = os.path.join(save_dir_1, filename)
    file = open(file_path, mode= 'w', newline='')
    writer = csv.writer(file)
    writer.writerow(['Round','Train Loss', 'Accuracy'])


    filename = f"MAK_log_{current_time}.csv"
    save_dir2 = f"MAK3_logs/FIRST_Exception/{args.stragglers_percent}_strg_{SYNC}sync"
    os.makedirs(save_dir2, exist_ok=True)
    # os.makedirs('MAK_logs/10_Rounds_0.5', exist_ok=True)
    # os.makedirs("experiment_logs/after_20_rounds", exist_ok=True)
    file_path = os.path.join(save_dir2, filename)

    file = open(file_path, mode= 'w', newline='')
    writer2 = csv.writer(file)
    writer2.writerow(['Round','Train Loss', 'Accuracy'])

    #  inference
    if args.eval:
        global_model.load_state_dict(torch.load(path_best_model))
        test_acc = utils.test(test_loader, global_model, test_creterion, args.device)
        textio.cprint(f'eval test_acc: {test_acc:.0f}%')
        gc.collect()
        sys.exit()

    local_models = utils.federated_setup(global_model, train_data, args)


    # stragglers
    ncum_of_layers = global_model.state_dict().keys().__len__()
    if args.stragglers is not None:
        stragglers_idx = random.sample(range(args.num_users), round(args.stragglers_percent * args.num_users))
    else:
        stragglers_idx = []

    #filter models based on slow and fast
    slow_models = {user_idx: user for user_idx, user in local_models.items() if user_idx in stragglers_idx}
    fast_models = {user_idx: user for user_idx, user in local_models.items() if user_idx not in stragglers_idx}


    textio.cprint(f"Slow client ids : {slow_models.keys()}")
    textio.cprint(f"Fast Client ids : {fast_models.keys()}")
   
    # Main execution
    stop_event.clear()  # Reset the event at the start
    with ThreadPoolExecutor(max_workers=2) as executor:
        f1 = executor.submit(train_centralized, slow_models, stragglers_idx, global_model,False)
        f2 = executor.submit(train_centralized, fast_models, stragglers_idx, global_model)

        futures = [f1, f2]
        
        try:
            # Wait for either completion or first exception
            done, not_done = concurrent.futures.wait(
                futures, 
                return_when=concurrent.futures.FIRST_EXCEPTION
            )
            # Set stop signal for the remaining process
            stop_event.set()

            for future in not_done:
                future.cancel()

            # Check for exceptions and results
            for future in done:
                # This will raise any exception that occurred
                future.result()
                
        except Exception as e:
            textio.cprint(f"Error occurred: {e}")
            # Signal all threads to stop
            stop_event.set()
            
            # Cancel any not-done futures
            for future in not_done:
                future.cancel()
            
            # Wait a moment for threads to clean up
            executor.shutdown(wait=True)
            raise  # Re-raise the error
        

    elapsed_min = (time.time() - start_time) / 60
    textio.cprint(f'total execution time: {elapsed_min:.0f} min')



