import argparse
import easydict
import sys
import os
import time
import yaml
import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as data
import torchvision.transforms as transforms
from tensorboardX import SummaryWriter
import faiss
import os
import sys
import errno
import random
import copy
import numpy as np
from collections import defaultdict
import torch
from torch.utils.data.sampler import Sampler
from utils import Logger, set_seed, GenIdx, IdentitySampler, SemiIdentitySampler_randomIR, SemiIdentitySampler_pseudoIR
from model.network import BaseResNet


def to_torch(ndarray):
    if type(ndarray).__module__ == 'numpy':
        return torch.from_numpy(ndarray)
    elif not torch.is_tensor(ndarray):
        raise ValueError("Cannot convert {} to torch tensor"
                         .format(type(ndarray)))
    return ndarray

def extract_cnn_feature(model, inputs):
    inputs = to_torch(inputs).cuda()
    outputs = model(inputs)
    outputs = outputs.data.cpu()
    return outputs

def extract_features(model, data_loader, print_freq=50):
    model.eval()

    with torch.no_grad():
        for i, (imgs, fnames, pids, _, _) in enumerate(data_loader):
            outputs = extract_cnn_feature(model, imgs)

    features=[]
    return features

def generate_cluster_labels(args, main_net, trainloader, tIndex, n_class, print_freq=50):
    main_net.train()
    cluster_pseudo_label=[]
    return cluster_pseudo_label

def set_seed(seed, cuda=True):
    random.seed(seed)
    np.random.RandomState(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if cuda:
        torch.cuda.manual_seed(seed)

def main_worker(args, args_main):
    ## set start epoch and end epoch
    start_epoch = args.start_epoch
    end_epoch = args.end_epoch

    ## set gpu id and seed id
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    torch.backends.cudnn.benchmark = True  # accelerate the running speed of convolution network
    device = "cuda" if torch.cuda.is_available() else "cpu"
    set_seed(args.seed, cuda=torch.cuda.is_available())
    if not os.path.isdir(args.dataset + "_" + args.setting + "_" + args.file_name):
        os.makedirs(args.dataset + "_" + args.setting + "_" + args.file_name)
    file_name = args.dataset + "_" + args.setting + "_" + args.file_name

    if args.dataset == "sysu":
        data_path = args.dataset_path + "SYSU-MM01/"
        log_path = os.path.join(file_name, args.dataset + "_" + args.log_path)
        vis_log_path = os.path.join(file_name, args.dataset + "_" + args.vis_log_path)
        model_path = os.path.join(file_name, args.dataset + "_" + args.model_path)
        test_mode = [1, 2]
    if not os.path.isdir(log_path):
        os.makedirs(log_path)
    if not os.path.isdir(vis_log_path):
        os.makedirs(vis_log_path)
    if not os.path.isdir(model_path):
        os.makedirs(model_path)

    sys.stdout = Logger(os.path.join(log_path, "log_os_mix_robust_0619_2.txt"))
    main_net1 = BaseResNet(pool_dim=args.pool_dim, class_num=395, per_add_iters=args.per_add_iters, arch=args.arch)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="OTLA-ReID for training")
    parser.add_argument("--config", default="config/baseline.yaml", help="config file")
    parser.add_argument("--resume", action="store_true", help="resume from checkpoint")
    parser.add_argument("--resume_path", default="", help="checkpoint path")

    args_main = parser.parse_args()
    args = yaml.load(open(args_main.config), Loader=yaml.FullLoader)
    args = easydict.EasyDict(args)
    main_worker(args, args_main)


