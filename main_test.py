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
from utils import Logger, set_seed, GenIdx
from data_loader import TestData
from data_manager import process_query_sysu, process_gallery_sysu, process_test_regdb
from model.network import BaseResNet
from engine import tester


def check_file(filename):
    if not os.path.isdir(filename):
        os.makedirs(filename)


def set_file_path(args):
    data_path = args.dataset_path + ("SYSU-MM01/" if args.dataset == "sysu" else "RegDB/")
    if not os.path.exists(data_path):
        raise RuntimeError("'{}' is not available".format(data_path))
    file_name = os.path.join(args.log_path, args.dataset + "_" + args.setting + "_" + args.file_name)
    log_path = os.path.join(file_name, args.dataset + "_" + args.log_path)
    vis_log_path = os.path.join(file_name, args.dataset + "_" + args.vis_log_path)
    model_path = os.path.join(file_name, args.dataset + "_" + args.model_path)
    # check_file
    check_file(file_name)
    check_file(log_path)
    check_file(vis_log_path)
    check_file(model_path)

    return data_path, log_path, vis_log_path, model_path


def create_test_loader(args, data_path, transform_test=None):
    # create the test set
    gall_cam = []
    query_cam = []
    if args.dataset == "sysu":
        query_img, query_label, query_cam = process_query_sysu(data_path, mode=args.mode)
        gall_img, gall_label, gall_cam = process_gallery_sysu(data_path, mode=args.mode)
    elif args.dataset == "regdb":
        query_img, query_label = process_test_regdb(data_path, trial=args.trial, modality=args.mode.split("to")[0])
        gall_img, gall_label = process_test_regdb(data_path, trial=args.trial, modality=args.mode.split("to")[1])
    gallset = TestData(gall_img, gall_label, transform_test=transform_test, img_size=(args.img_w, args.img_h))
    queryset = TestData(query_img, query_label, transform_test=transform_test, img_size=(args.img_w, args.img_h))

    # create the testing data loader
    gall_loader = data.DataLoader(gallset, batch_size=args.test_batch_size, shuffle=False, num_workers=args.workers)
    query_loader = data.DataLoader(queryset, batch_size=args.test_batch_size, shuffle=False, num_workers=args.workers)

    return gall_loader, gall_label, gall_cam, query_loader, query_label, query_cam


def print_dataset_statistics(args, query_label, gall_label, end):
    print("Dataset {} Statistics:".format(args.dataset))
    print("  ----------------------------")
    print("  subset   | # ids | # images")
    print("  ----------------------------")
    print("  query    | {:5d} | {:8d}".format(len(np.unique(query_label)), len(query_label)))
    print("  gallery  | {:5d} | {:8d}".format(len(np.unique(gall_label)), len(gall_label)))
    print("  ----------------------------")
    print("Data loading time:\t {:.3f}".format(time.time() - end))


def main_worker(args, args_main):
    # set start epoch
    start_epoch = args.start_epoch

    # set gpu id and seed id
    if torch.cuda.is_available():
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
        torch.backends.cudnn.benchmark = True  # accelerate the running speed of convolution network
    device = "cuda" if torch.cuda.is_available() else "cpu"
    set_seed(args.seed, cuda=torch.cuda.is_available())

    # set log file
    if args.dataset == "sysu":
        data_path, log_path, _, _ = set_file_path(args)
        test_mode = [1, 2]
    elif args.dataset == "regdb":
        data_path, log_path, _, _ = set_file_path(args)
        if args.mode == "thermaltovisible":
            test_mode = [1, 2]
        elif args.mode == "visibletothermal":
            test_mode = [2, 1]

    sys.stdout = Logger(os.path.join(log_path, "log_test.txt"))

    # load data
    print("==========\nargs_main:{}\n==========".format(args_main))
    print("==========\nargs:{}\n==========".format(args))
    print("==> Loading data...")

    # transform
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    transform_test = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((args.img_h, args.img_w)),
        transforms.ToTensor(),
        normalize,
    ])

    end = time.time()
    # testing loader
    gall_loader, gall_label, gall_cam, query_loader, query_label, query_cam = create_test_loader(args, data_path,
                                                                                                 transform_test)
    # print dataset info
    print_dataset_statistics(args, query_label, gall_label, end)

    # define n_class
    if args.dataset == "sysu":
        n_class = 395  # initial value
    elif args.dataset == "regdb":
        n_class = 206  # initial value
    else:
        n_class = 1000  # initial value

    # resume checkpoints
    if args_main.resume:
        resume_path = args_main.resume_path
        if os.path.exists(resume_path):
            checkpoint1 = torch.load(resume_path)
            if "main_net1" in checkpoint1.keys():
                n_class = checkpoint1["main_net1"]["classifier.weight"].size(0)
            start_epoch = checkpoint1["epoch"]
            print("==> Loading checkpoint {} (epoch {}, number of classes {})".format(resume_path, start_epoch,
                                                                                      n_class))
        else:
            print("==> No checkpoint is found at {} (epoch {}, number of classes {})".format(resume_path,
                                                                                             start_epoch, n_class))
    else:
        print("==> No checkpoint is loaded (epoch {}, number of classes {})".format(start_epoch, n_class))

    # build model
    main_net = BaseResNet(pool_dim=args.pool_dim, class_num=n_class, per_add_iters=args.per_add_iters, arch=args.arch)
    if args_main.resume and os.path.isfile(resume_path):
        if "main_net1" in checkpoint1.keys():
            main_net.load_state_dict(checkpoint1["main_net1"])
    main_net.to(device)

    # start testing
    if args.dataset == "sysu":
        print("Testing Epoch: {}, Testing mode: {}".format(start_epoch, args.mode))
    elif args.dataset == "regdb":
        print("Testing Epoch: {}, Testing mode: {}, Trial: {}".format(start_epoch, args.mode, args.trial))

    end = time.time()
    if args.dataset == "sysu":
        cmc, mAP, mINP = tester(args, start_epoch, main_net, main_net, test_mode, gall_label, gall_loader, query_label,
                                query_loader, feat_dim=args.pool_dim, query_cam=query_cam, gall_cam=gall_cam)
    elif args.dataset == "regdb":
        cmc, mAP, mINP = tester(args, start_epoch, main_net, main_net, test_mode, gall_label, gall_loader, query_label,
                                query_loader, feat_dim=args.pool_dim)
    print("Testing time per epoch: {:.3f}".format(time.time() - end))

    print("Performance: Rank-1: {:.2%} | Rank-5: {:.2%} | Rank-10: {:.2%}| Rank-20: {:.2%}| mAP: {:.2%}| " \
          "mINP: {:.2%}".format(cmc[0], cmc[4], cmc[9], cmc[19], mAP, mINP))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="OTLA-ReID for testing")
    parser.add_argument("--config", default="config/regdb.yaml", help="config file")
    parser.add_argument("--resume", action="store_true", help="resume from checkpoint")
    parser.add_argument("--resume_path", default="", help="checkpoint path")

    args_main = parser.parse_args()
    args = yaml.load(open(args_main.config), Loader=yaml.FullLoader)
    args = easydict.EasyDict(args)

    main_worker(args, args_main)
