import utils
import torch
import models
import copy
import numpy as np
from collections import OrderedDict
from itertools import islice
# from torchinfo import summary
from configurations import args_parser
args = args_parser()
textio, best_val_acc, path_best_model = utils.initializations(args)
device = torch.device("cuda" if torch.cuda.is_available() else 'cpu') # Wahab


textio.cprint(str(args))

 # data
train_data, test_loader = utils.data(args)
input, output, train_data, val_loader = utils.data_split(train_data, len(test_loader.dataset), args,test_loader)

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


for name, param in global_model.state_dict().items():
    print(f"Layer name: {name}, Shape: {param.shape}")

# global_model.to(args.device)
global_model.to(device=device)
num_of_layers = global_model.state_dict().keys().__len__()

print(f"Total leyaers : {num_of_layers}")

model1 = copy.deepcopy(global_model)

user_new_state_dict = copy.deepcopy(global_model).state_dict()
if args.up_to_layer is not None:
    up_to_layer = num_of_layers - args.up_to_layer  # last-to-first layers updated
else:
    up_to_layer = np.random.randint(1, num_of_layers + 1)  # random last-to-first layers updated
print(f"Upto layers : {up_to_layer}")

user_updated_layers = OrderedDict(islice(reversed(model1.state_dict().items()), up_to_layer))
print(f"Total layers of the model are: {num_of_layers} and model will update: {up_to_layer}")
print(" Udated layers will  be: ++++++++++++++++++++++++++++")
for key in user_updated_layers:
    print(key)

print(f"*******************************************************************************************************************")
for name, param in global_model.named_parameters():
  print(f'Layer: {name}, shape: {param.shape}, Required Grad: {param.requires_grad}')

total_paramters = sum(p.numel() for p in global_model.parameters())
print(f'Total paramters are: {total_paramters}')


print(f"*******************************************************************************************************************")

trainable_paramters = sum(p.numel() for p in global_model.parameters() if p.requires_grad)
non_trainalbe_praramters = total_paramters - trainable_paramters
print(f"Trainable paramters: {trainable_paramters}")
print(f'Non-trainable paramters: {non_trainalbe_praramters}')