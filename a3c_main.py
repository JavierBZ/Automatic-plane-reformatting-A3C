import os
os.environ["OMP_NUM_THREADS"] = "1"
import argparse
import logging
import torch
import torch.multiprocessing as mp
from a3c_models import A3C_MLP, A3C_CONV, A3C_CONV_new,A3C_CONV_2
from a3c_train import train
from a3c_test import test
from shared_optim import SharedRMSprop, SharedAdam
import time


parser = argparse.ArgumentParser(description='A3C')
parser.add_argument(
    '--lr',
    type=float,
    default=0.0000025,
    metavar='LR',
    help='learning rate (default: 0.00001)')
parser.add_argument(
    '--gamma',
    type=float,
    default=0.99,
    metavar='G',
    help='discount factor for rewards (default: 0.99)')
parser.add_argument(
    '--tau',
    type=float,
    default=1.00,
    metavar='T',
    help='parameter for GAE (default: 1.00)')
parser.add_argument(
    '--seed',
    type=int,
    default=1,
    metavar='S',
    help='random seed (default: 1)')
parser.add_argument(
    '--workers',
    type=int,
    default=32,
    #default=24,
    metavar='W',
    help='how many training processes to use (default: 32)')
parser.add_argument(
    '--num-steps',
    type=int,
    default=10,
    metavar='NS',
    help='number of forward steps in A3C (default: 50)')
parser.add_argument(
    '--max-episode-length',
    type=int,
    default=100,
    metavar='M',
    help='maximum length of an episode (default: 100)')

parser.add_argument(
    '--shared-optimizer',
    action='store_false',
    help='use an optimizer without shared statistics.')
parser.add_argument(
    '--load',
    action='store_true',
    help='load a trained model')
parser.add_argument(
    '--save-max',
    action='store_false',
    help='Save model on every test run high score matched or bested')
parser.add_argument(
    '--optimizer',
    default='Adam',
    metavar='OPT',
    help='shares optimizer choice of Adam or RMSprop')
parser.add_argument(
    '--load-model-dir',
    #default='home1/jebisbal/a3c_models/',
    default='load_a3c/',
    metavar='LMD',
    help='folder to load trained models from')
parser.add_argument(
    '--save-model-dir',
    #default='home1/jebisbal/a3c_models/',
    default='a3c_models/',
    metavar='SMD',
    help='folder to save trained models')
parser.add_argument(
    '--log-dir',
    default='logs/',
    metavar='LG',
    help='folder to save logs')
parser.add_argument(
    '--model',
    #default='MLP',
    default='CONV',
    metavar='M',
    help='Model type to use')
parser.add_argument(
    '--stack-frames',
    type=int,
    default=1,
    metavar='SF',
    help='Choose number of observations to stack')
parser.add_argument(
    '--gpu-ids',
    type=int,
    default=-1,
    nargs='+',
    help='GPUs to use [-1 CPU only] (default: -1)')
parser.add_argument(
    '--amsgrad',
    action='store_false',
    help='Adam optimizer amsgrad parameter')
parser.add_argument(
    '--gdrive',
    action='store_true',
    help='Use Google Colab (Drive)')

parser.add_argument(
    '--MAC',
    action='store_true',
    help='Use MAC')

parser.add_argument(
    '--group',
    default='2022_04_06-RPA-PC2Dorto_Adam+3layers_256-f',
    metavar='grp',
    help='group for wandb')

parser.add_argument(
    '--env',
    default='MedicalPlayer-RPA-PC2Dorto_Adam+3layers_256-f',
    #default='MedicalPlayer-LPA-PCvel_f',                                      #CAMBIAR ACAAAA
    metavar='ENV',
    help='environment to train on (default: MedicalPlayer-cont)')

parser.add_argument(
    '--folder',
    type=int,
    default=0,
    nargs='+',
    help='folder for cross-validation')

# Based on
# https://github.com/pytorch/examples/tree/master/mnist_hogwild
# Training settings
# Implemented multiprocessing using locks but was not beneficial. Hogwild
# training was far superior

if __name__ == '__main__':

    args = parser.parse_args()
    # logging.basicConfig(level=logging.INFO)
    # logging.info(os.environ["CUDA_VISIBLE_DEVICES"])


    if args.gdrive:
        args.log_dir = "gdrive/My Drive/11_semestre/Automatic_reformatting/Results_RL/A3C/logs/"
        args.save_model_dir = "gdrive/My Drive/11_semestre/Automatic_reformatting/Results_RL/A3C/models/"
        args.load_model_dir = "gdrive/My Drive/11_semestre/Automatic_reformatting/Results_RL/A3C/models/"
    if args.MAC:
        args.cluster = False
        args.log_dir = "/Users/javierbisbal/Documents/GoogleDrive_notsync/Automatic_reformatting/a3c_results/"
        args.save_model_dir = "/Users/javierbisbal/Documents/GoogleDrive_notsync/Automatic_reformatting/a3c_models/"
        args.load_model_dir = "/Users/javierbisbal/Documents/GoogleDrive_notsync/Automatic_reformatting/a3c_models/"
    else:
        args.cluster = True

    args.model = 'CONV'
    args.screen_dims = (3,70,70)
    #args.screen_dims = (9,224,224)
    args.scale_d = 65
    args.scale_d2 = 10
    #args.angle_step = 5
    #args.dist_step = 5
    args.angle_step = 3
    args.dist_step = 3
    args.angle_step2 = 5
    args.dist_step2 = 2
    args.second_model = False
    args.max_episode_length2 = 20
    #args.max_episode_length = args.max_episode_length + args.max_episode_length2
    args.n_actions = 4
    args.num_steps2 = 2
    args.contrast = 'all'
    args.lstm = True
    args.frame_history = 1


    args.Plane = 'RPA'


    args.mid_slice = 0
    args.velocities = False
    args.vel_interp = False
    args.PC_vel  = False
    args.TwoDim = True
    args.orto_2D = True

    args.only_move = False

    # args.velocities = True
    # args.vel_interp = True
    # args.PC_vel = True
    # args.orto_2D = False
    # args.TwoDim = False
    # args.mid_slice = args.screen_dims[0]//2


    args.folder = args.folder[0]
    args.env = args.env + str(args.folder)
    args.group = args.group + str(args.folder)

    args.optimizer = 'Adam'
    args.out_feats = 256
    args.layers = 3

    torch.manual_seed(args.seed)
    if args.gpu_ids == -1:
        args.gpu_ids = [-1]
    else:
        torch.cuda.manual_seed(args.seed)
        mp.set_start_method('spawn')


    if args.model == 'MLP':
        num_actions = args.n_actions
        num_channels= args.frame_history
        shared_model = A3C_MLP(num_channels, num_actions)
    if args.model == 'CONV':
        num_actions = args.n_actions
        if (args.velocities and not args.vel_interp) or args.TwoDim:
            num_channels= args.frame_history * 3
        elif args.PC_vel:
            num_channels =args.frame_history * 2
        else:
            num_channels= args.frame_history
        if args.lstm:
            shared_model = A3C_CONV(num_channels, num_actions,out_feats=args.out_feats,layers=args.layers)
        else:
            shared_model = A3C_CONV_new(num_channels, num_actions)
    if args.model == 'CONV_2':
        num_actions = args.n_actions
        if (args.velocities and not args.vel_interp) or args.TwoDim:
            num_channels= args.frame_history * 3
        elif args.PC_vel:
            num_channels =args.frame_history * 2
        else:
            num_channels= args.frame_history
        if args.lstm:
            shared_model = A3C_CONV_2(num_channels, num_actions)
        else:
            shared_model = A3C_CONV_new(num_channels, num_actions)

    shared_model_2 = None
    if args.second_model:
        if args.model == 'MLP':
            num_actions = args.n_actions
            if args.velocities and not args.vel_interp:
                num_channels= args.frame_history * 3
            else:
                num_channels= args.frame_history
            shared_model_2 = A3C_MLP(num_channels, num_actions)
        if args.model == 'CONV':
            num_actions = args.n_actions
            if args.velocities and not args.vel_interp:
                num_channels= args.frame_history * 3
            else:
                num_channels= args.frame_history
            if args.lstm:
                shared_model_2 = A3C_CONV(num_channels, num_actions)
            else:
                shared_model_2 = A3C_CONV_new(num_channels, num_actions)


    if args.load:
        saved_state = torch.load('{0}{1}.dat'.format(
            args.load_model_dir, args.env), map_location=lambda storage, loc: storage)
        shared_model.load_state_dict(saved_state)
        if args.second_model:
            shared_model2.load_state_dict(saved_state)


    shared_model.share_memory()
    if args.second_model:
        shared_model2.share_memory()


    if args.shared_optimizer:
        if args.optimizer == 'RMSprop':
            optimizer = SharedRMSprop(shared_model.parameters(), lr=args.lr)
            if args.second_model:
                optimizer2 = SharedRMSprop(shared_model2.parameters(), lr=args.lr)
        if args.optimizer == 'Adam':
            optimizer = SharedAdam(
                shared_model.parameters(), lr=args.lr, amsgrad=args.amsgrad)
            if args.second_model:
                optimizer2 = SharedAdam(
                    shared_model.parameters(), lr=args.lr, amsgrad=args.amsgrad)
        optimizer.share_memory()
        if args.second_model:
            optimizer2.share_memory()
    else:
        optimizer = None

    processes = []
    if args.second_model:
        p = mp.Process(target=test, args=(args, shared_model,args.group,'validation'))
    else:
        p = mp.Process(target=test, args=(args, shared_model,args.group,'validation'))
    p.start()
    processes.append(p)
    time.sleep(0.1)
    for rank in range(0, args.workers):
        if args.second_model:
            p = mp.Process(target=train, args=(
                rank, args, shared_model, optimizer,args.group,'train'))
        else:
            if rank == 0:
                p = mp.Process(target=train, args=(
                    rank, args, shared_model, optimizer,args.group,'train',True))
            else:
                p = mp.Process(target=train, args=(
                    rank, args, shared_model, optimizer,args.group,'train',False))
        p.start()
        processes.append(p)
        time.sleep(0.1)
    for p in processes:
        time.sleep(0.1)
        p.join()
