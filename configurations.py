import argparse


def args_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument('--exp_name', type=str, default='exp',
                        help="the name of the current experiment")
    parser.add_argument('--stragglers', type=str, default='salf',
                        choices=['salf', 'drop', None],
                        help="whether the FL is stragglers aware")
    parser.add_argument('--stragglers_percent', type=float, default=0.9,
                        help="the percent of percent out of the edge users")
    parser.add_argument('--up_to_layer', type=int, default=1,
                        help="if 'None' - choose randomly, else - update until (num_layers - up_to_layer)"
                             "example: up_to_layer=1 results with an update up to one before the first layer")

    parser.add_argument('--data', type=str, default='mnist',
                        choices=['mnist', 'cifar10'],
                        help="dataset to use (mnist or cifar)")
    parser.add_argument('--num_samples', type=int, default=None,
                        help="number of samples per user; if 'None' - uniformly distribute all data among all users)")

    parser.add_argument('--model', type=str, default='mlp',
                        choices=['mlp', 'cnn2'],
                        help="model to use (cnn, mlp)")
    parser.add_argument('--global_epochs', type=int, default=100,
                        help="number of global epochs")
    parser.add_argument('--device', type=str, default='cuda:1',
                        choices=['cuda:0', 'cuda:1', 'cpu'],
                        help="device to use (gpu or cpu)")

    parser.add_argument('--num_users', type=int, default=30,
                        help="number of users participating in the federated learning")

    parser.add_argument('--lr', type=float, default=0.1,
                        help="learning rate")

    parser.add_argument('--train_batch_size', type=int, default=16,
                        help="trainset batch size")
    parser.add_argument('--local_iterations', type=int, default=1,
                        help="number of local iterations instead of local epoch")
    parser.add_argument('--norm_mean', type=float, default=0.5,
                        help="normalize the data to norm_mean")
    parser.add_argument('--norm_std', type=float, default=0.5,
                        help="normalize the data to norm_std")
    parser.add_argument('--test_batch_size', type=int, default=1000,
                        help="testset batch size")
    parser.add_argument('--local_epochs', type=int, default=1,
                        help="number of local epochs")
    parser.add_argument('--optimizer', type=str, default='sgd',
                        choices=['sgd', 'adam'],
                        help="optimizer to use (sgd or adam)")
    parser.add_argument('--momentum', type=float, default=0.5,
                        help="momentum")
    parser.add_argument('--seed', type=float, default=1234,
                        help="manual seed for reproducibility")
    parser.add_argument('--eval', action='store_true',
                        help="weather to perform inference of training")
    args = parser.parse_args()
    return args
