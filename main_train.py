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

from utils import Logger, set_seed, GenIdx, IdentitySampler, SemiIdentitySampler_randomIR, SemiIdentitySampler_pseudoIR, \
    AllSampler
from data_loader import SYSUData, SYSUData_E, RegDBData, TestData
from data_manager import process_query_sysu, process_gallery_sysu, process_test_regdb
from model.network import BaseResNet
from loss import TripletLoss, PredictionAlignmentLoss, RobustTripletLoss_final
from optimizer import select_optimizer, adjust_learning_rate
from engine import trainer, tester, evaler, cluster
from otla_sk import cpu_sk_ir_trainloader
from torch.utils.data.sampler import Sampler
from IPython import embed


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
    # check file
    check_file(file_name)
    check_file(log_path)
    check_file(vis_log_path)
    check_file(model_path)

    return data_path, log_path, vis_log_path, model_path


def print_dataset_statistics(args, trainset, query_label, gall_label, end):
    n_rgb = len(np.unique(trainset.train_color_label))  # number of visible ids
    n_ir = len(np.unique(trainset.train_thermal_label))  # number of infrared ids
    n_query = len(np.unique(query_label))  # number of query ids
    n_gall = len(np.unique(gall_label))  # number of gallery ids
    print("Dataset {} Statistics:".format(args.dataset))
    print("  ----------------------------")
    print("  subset   | # ids | # images")
    print("  ----------------------------")
    print("  visible  | {:5d} | {:8d}".format(n_rgb, len(trainset.train_color_label)))
    print("  thermal  | {:5d} | {:8d}".format(n_ir, len(trainset.train_thermal_label)))
    print("  ----------------------------")
    print("  query    | {:5d} | {:8d}".format(n_query, len(query_label)))
    print("  gallery  | {:5d} | {:8d}".format(n_gall, len(gall_label)))
    print("  ----------------------------")
    print("Data loading time:\t {:.3f}".format(time.time() - end))


def compute_cluster_label_acc(cluster_thermal_label, GT, train_thermal_pseudo_label):
    uni_cluster = np.unique(cluster_thermal_label)  # 去除重复的元素，由小到大排列
    for i in range(max(uni_cluster) + 1):
        index = np.where(cluster_thermal_label == i)
        cluster2OT = train_thermal_pseudo_label[index]
        unique_values, counts = np.unique(cluster2OT, return_counts=True)
        # 找到出现次数最多的值和对应的次数
        max_count_index = np.argmax(counts)
        most_common_value = unique_values[max_count_index]
        cluster_thermal_label[index] = most_common_value
    outlines = np.where(cluster_thermal_label == -1)
    for outline in outlines:
        cluster_thermal_label[outline] = 0
    err_x = np.sum(cluster_thermal_label[:] == GT[:])
    print("err_x: {}".format(err_x))
    cluster_missrate = err_x.astype(float) / (cluster_thermal_label.shape[0])

    return cluster_missrate


def create_dataset(args, data_path, transform_train_rgb=None, transform_train_ir=None, transform_test=None):
    if args.dataset == "sysu":
        # training set
        trainset = SYSUData(args, data_path, transform_train_rgb=transform_train_rgb,
                            transform_train_ir=transform_train_ir)
        # evaluating set
        evaltrainset = SYSUData(args, data_path, transform_train_rgb=transform_test,
                                transform_train_ir=transform_test, trainset=False)
    elif args.dataset == "regdb":
        # training set
        trainset = RegDBData(args, data_path, transform_train_rgb=transform_train_rgb,
                             transform_train_ir=transform_train_ir)
        # evaluating set
        evaltrainset = RegDBData(args, data_path, transform_train_rgb=transform_test,
                                 transform_train_ir=transform_test, trainset=False)
    # generate the idx of each person identity
    color_pos, thermal_pos = GenIdx(trainset.train_color_label, trainset.train_thermal_label)

    return trainset, evaltrainset, color_pos, thermal_pos


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


def create_train_eval_loader(args, dataset, sampler, drop_last=True):
    dataset.cIndex = sampler.index1  # color index
    dataset.tIndex = sampler.index2  # thermal index
    dataloader = data.DataLoader(dataset, batch_size=args.train_batch_size * args.num_pos, sampler=sampler,
                                 num_workers=args.workers, drop_last=drop_last)
    return dataloader


def define_criterion(args):
    criterion = []
    criterion_id = nn.CrossEntropyLoss()  # id loss
    criterion.append(criterion_id)
    criterion_tri = TripletLoss(margin=args.margin)  # triplet loss
    criterion.append(criterion_tri)
    criterion_dis = nn.BCELoss()
    criterion.append(criterion_dis)
    criterion_pa = PredictionAlignmentLoss(lambda_vr=args.lambda_vr,
                                           lambda_rv=args.lambda_rv)  # prediction alignment loss
    criterion.append(criterion_pa)
    RobustTripletLoss = RobustTripletLoss_final(batch_size=args.train_batch_size * args.num_pos, margin=args.margin)
    criterion.append(RobustTripletLoss)

    return criterion


def compute_label_pred_acc(ir_op, ir_mp, ir_real, train_thermal_pseudo_label, trainset):
    predict_per_epoch_op = (ir_op.eq(ir_real).sum().item()) / ir_real.size(0)
    predict_per_epoch_mp = (ir_mp.eq(ir_real).sum().item()) / ir_real.size(0)
    predict_per_epoch_all = (train_thermal_pseudo_label == trainset.train_thermal_label).sum() / len(
        trainset.train_thermal_label)

    return predict_per_epoch_op, predict_per_epoch_mp, predict_per_epoch_all


def compute_otla_label_acc(evaltrainset, GT, thermal_pseudo_label1, thermal_pseudo_label2):
    evaltrainset.train_thermal_label = thermal_pseudo_label1
    err = np.sum(thermal_pseudo_label1[:] == GT[:])
    OTLA_missrate = err.astype(float) / thermal_pseudo_label2.shape[0]

    return OTLA_missrate


def compute_prob(args, eval_set, pseudo_label, main_net):
    eval_set.train_thermal_label = pseudo_label
    eval_sampler = AllSampler(args.dataset, eval_set.train_color_label, eval_set.train_thermal_label)
    eval_loader = create_train_eval_loader(args, eval_set, eval_sampler)
    n_ir = len(eval_set.train_thermal_label)
    prob_I = evaler(args, main_net, eval_loader, n_ir)
    return prob_I


def main_worker(args, args_main):
    # set start epoch and end epoch
    start_epoch = args.start_epoch
    end_epoch = args.end_epoch

    # set gpu id and seed id
    if torch.cuda.is_available():
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
        torch.backends.cudnn.benchmark = True  # accelerate the running speed of convolution network
    device = "cuda" if torch.cuda.is_available() else "cpu"
    set_seed(args.seed, cuda=torch.cuda.is_available())

    # set log file
    if args.dataset == "sysu":
        data_path, log_path, vis_log_path, model_path = set_file_path(args)
        test_mode = [1, 2]
    elif args.dataset == "regdb":
        data_path, log_path, vis_log_path, model_path = set_file_path(args)
        if args.mode == "thermaltovisible":
            test_mode = [1, 2]
        elif args.mode == "visibletothermal":
            test_mode = [2, 1]

    sys.stdout = Logger(os.path.join(log_path, args.train_os_log))
    test_os_log = open(os.path.join(log_path, args.test_os_log), "w")
    # tensorboard
    writer = SummaryWriter(vis_log_path)

    # load data
    print("==========\nargs_main:{}\n==========".format(args_main))
    print("==========\nargs:{}\n==========".format(args))
    print("==> Loading data...")

    # set transform
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    transform_train_rgb = transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomGrayscale(p=0.5),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
        transforms.RandomErasing(p=0.5),
    ])
    transform_train_ir = transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
        transforms.RandomErasing(p=0.5),
    ])
    transform_test = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((args.img_h, args.img_w)),
        transforms.ToTensor(),
        normalize,
    ])

    end = time.time()
    # training set, evaluating set and idx of each person identity
    trainset1, evaltrainset1, color_pos1, thermal_pos1 = create_dataset(args, data_path, transform_train_rgb,
                                                                        transform_train_ir, transform_test)
    trainset2, evaltrainset2, color_pos2, thermal_pos2 = create_dataset(args, data_path, transform_train_rgb,
                                                                        transform_train_ir, transform_test)
    # testing data loader
    gall_loader, gall_label, gall_cam, query_loader, query_label, query_cam = create_test_loader(args, data_path,
                                                                                                 transform_test)
    # print dataset info
    print_dataset_statistics(args, trainset1, query_label, gall_label, end)

    # build model
    n_classes = len(np.unique(trainset1.train_color_label))  # number of classes
    main_net1 = BaseResNet(pool_dim=args.pool_dim, class_num=n_classes, per_add_iters=args.per_add_iters,
                           arch=args.arch)
    main_net2 = BaseResNet(pool_dim=args.pool_dim, class_num=n_classes, per_add_iters=args.per_add_iters,
                           arch=args.arch)
    main_net1.to(device)
    main_net2.to(device)

    # resume checkpoints
    if args_main.resume:
        resume_path1 = args_main.resume_path1
        resume_path2 = args_main.resume_path2
        if os.path.exists(resume_path1) and os.path.exists(resume_path2):
            checkpoint1 = torch.load(resume_path1)
            checkpoint2 = torch.load(resume_path2)
            if "epoch" in checkpoint1.keys():
                start_epoch = checkpoint1["epoch"]
            main_net1.load_state_dict(checkpoint1["main_net1"])
            main_net2.load_state_dict(checkpoint2["main_net2"])
            print("==> Loading checkpoint {} (epoch {})".format(resume_path1, start_epoch))
            print("==> Loading checkpoint {} (epoch {})".format(resume_path2, start_epoch))
        else:
            print("==> No checkpoint is found at {} or {}".format(resume_path1, resume_path2))
    print("Start epoch: {}, end epoch: {}".format(start_epoch, end_epoch))

    # define loss functions
    criterion = define_criterion(args)

    # set optimizer
    optimizer1 = select_optimizer(args, main_net1)
    optimizer2 = select_optimizer(args, main_net2)

    # start training and testing
    best_acc = 0
    train_thermal_pseudo_label1 = np.random.randint(0, n_classes, len(trainset1.train_thermal_label))
    train_thermal_pseudo_label2 = np.random.randint(0, n_classes, len(trainset2.train_thermal_label))

    for epoch in range(start_epoch, end_epoch - start_epoch):
        end = time.time()
        print("==> Preparing data loader...")
        if args.setting == "unsupervised" or args.setting == "semi-supervised":
            if epoch == 0:
                sampler1 = SemiIdentitySampler_randomIR(trainset1.train_color_label, train_thermal_pseudo_label1,
                                                        color_pos1, args.num_pos, args.train_batch_size,
                                                        args.dataset_num_size)
                sampler2 = SemiIdentitySampler_randomIR(trainset2.train_color_label, train_thermal_pseudo_label2,
                                                        color_pos2, args.num_pos, args.train_batch_size,
                                                        args.dataset_num_size)
            else:
                sampler1 = SemiIdentitySampler_pseudoIR(trainset1.train_color_label, train_thermal_pseudo_label2,
                                                        color_pos1, args.num_pos, args.train_batch_size,
                                                        args.dataset_num_size)
                sampler2 = SemiIdentitySampler_pseudoIR(trainset2.train_color_label, train_thermal_pseudo_label1,
                                                        color_pos2, args.num_pos, args.train_batch_size,
                                                        args.dataset_num_size)

        # create training data loader
        trainloader1 = create_train_eval_loader(args, trainset1, sampler1)
        trainloader2 = create_train_eval_loader(args, trainset2, sampler2)

        ir_pseudo_label_op1, ir_pseudo_label_mp1, ir_real_label1, unique_tIndex_idx1 = cpu_sk_ir_trainloader(args,
                                                                                                             main_net1,
                                                                                                             trainloader1,
                                                                                                             sampler1.index2,
                                                                                                             n_classes)
        ir_pseudo_label_op2, ir_pseudo_label_mp2, ir_real_label2, unique_tIndex_idx2 = cpu_sk_ir_trainloader(args,
                                                                                                             main_net2,
                                                                                                             trainloader2,
                                                                                                             sampler2.index2,
                                                                                                             n_classes)

        train_thermal_pseudo_label1[unique_tIndex_idx1] = ir_pseudo_label_op1.numpy()
        train_thermal_pseudo_label2[unique_tIndex_idx2] = ir_pseudo_label_op2.numpy()
        # label prediction accuracy
        print("1Total number of IR per trainloader: {}, Unique number of IR " \
              "per trainloader: {}".format(len(sampler1.index2), len(unique_tIndex_idx1)))
        predict_op1, predict_mp1, predict_all1 = compute_label_pred_acc(ir_pseudo_label_op1, ir_pseudo_label_mp1,
                                                                        ir_real_label1, train_thermal_pseudo_label1,
                                                                        trainset1)
        print("1Label prediction accuracy, Op: {:.2f}%, Mp: {:.2f}%, All: {:.2f}%".format(predict_op1 * 100,
                                                                                          predict_mp1 * 100,
                                                                                          predict_all1 * 100))
        print("2Total number of IR per trainloader: {}, Unique number of IR " \
              "per trainloader: {}".format(len(sampler2.index2), len(unique_tIndex_idx2)))
        predict_op2, predict_mp2, predict_all2 = compute_label_pred_acc(ir_pseudo_label_op2, ir_pseudo_label_mp2,
                                                                        ir_real_label2, train_thermal_pseudo_label2,
                                                                        trainset2)
        print("2Label prediction accuracy, Op: {:.2f}%, Mp: {:.2f}%, All: {:.2f}%".format(predict_op2 * 100,
                                                                                          predict_mp2 * 100,
                                                                                          predict_all2 * 100))

        # compute OTLA label accuracy
        GT = np.load(args.GT_path)
        OTLA_missrate1 = compute_otla_label_acc(evaltrainset1, GT, train_thermal_pseudo_label1,
                                                train_thermal_pseudo_label2)
        print("OTLA_label_acc1:{}".format(OTLA_missrate1))
        OTLA_missrate2 = compute_otla_label_acc(evaltrainset2, GT, train_thermal_pseudo_label2,
                                                train_thermal_pseudo_label1)
        print("OTLA_label_acc2:{}".format(OTLA_missrate2))

        # compute cluster label accuracy
        cluster_thermal_label1 = np.load(args.cluster_thermal_label_path)
        cluster_thermal_label2 = cluster_thermal_label1.copy()
        cluster_missrate1 = compute_cluster_label_acc(cluster_thermal_label1, GT, train_thermal_pseudo_label1)
        print("Cluster_label_acc1:{}".format(cluster_missrate1))
        cluster_missrate2 = compute_cluster_label_acc(cluster_thermal_label2, GT, train_thermal_pseudo_label2)
        print("Cluster_label_acc2:{}".format(cluster_missrate2))

        trainset1.train_thermal_label = train_thermal_pseudo_label2
        trainset2.train_thermal_label = train_thermal_pseudo_label1

        # confidence generating
        print("==> Start confidence generating...")
        prob_I1 = compute_prob(args, evaltrainset1, train_thermal_pseudo_label2, main_net1)
        prob_I2 = compute_prob(args, evaltrainset2, train_thermal_pseudo_label1, main_net2)
        print("==> Finish confidence generating...")

        cluster_prob_I1 = compute_prob(args, evaltrainset1, cluster_thermal_label1, main_net1)
        t_index = np.where(prob_I1 < cluster_prob_I1)
        if t_index is not None:
            print("==> Start label hybrid...")
            train_thermal_pseudo_label1[t_index] = cluster_thermal_label1[t_index]
            # Net2
            cluster_prob_I2 = compute_prob(args, evaltrainset2, cluster_thermal_label2, main_net2)
            t_index = np.where(prob_I2 < cluster_prob_I2)
            if t_index is not None:
                train_thermal_pseudo_label2[t_index] = cluster_thermal_label2[t_index]
            err_x = np.sum(train_thermal_pseudo_label2[:] == GT[:])
            Hybird_missrate = err_x.astype(float) / (train_thermal_pseudo_label1.shape[0])
            print("Hybird_label_acc:{}".format(Hybird_missrate))
            print("==> Finish label hybrid...")

            trainset1.train_thermal_label = train_thermal_pseudo_label1
            trainset2.train_thermal_label = train_thermal_pseudo_label2

        # training
        print("==> Start training...")
        trainer(args, epoch, main_net1, adjust_learning_rate, optimizer1, trainloader1, criterion, prob_I1,
                writer=writer)
        trainer(args, epoch, main_net2, adjust_learning_rate, optimizer2, trainloader2, criterion, prob_I2,
                writer=writer)
        print("Training time per epoch: {:.3f}".format(time.time() - end))

        if epoch % args.eval_epoch == 0:
            if args.dataset == "sysu":
                print("Testing Epoch: {}, Testing mode: {}".format(epoch, args.mode))
                print("Testing Epoch: {}, Testing mode: {}".format(epoch, args.mode), file=test_os_log)
            elif args.dataset == "regdb":
                print("Testing Epoch: {}, Testing mode: {}, Trial: {}".format(epoch, args.mode, args.trial))
                print("Testing Epoch: {}, Testing mode: {}, Trial: {}".format(epoch, args.mode, args.trial),
                      file=test_os_log)

            # start testing
            end = time.time()
            if args.dataset == "sysu":
                cmc, mAP, mINP = tester(args, epoch, main_net1, main_net1, test_mode, gall_label, gall_loader,
                                        query_label, query_loader, feat_dim=args.pool_dim, query_cam=query_cam,
                                        gall_cam=gall_cam, writer=writer)
            elif args.dataset == "regdb":
                cmc, mAP, mINP = tester(args, epoch, main_net1, main_net1, test_mode, gall_label, gall_loader,
                                        query_label,
                                        query_loader, feat_dim=args.pool_dim, writer=writer)
            print("Testing time per epoch: {:.3f}".format(time.time() - end))

            # save model
            if cmc[0] > best_acc:
                best_acc = cmc[0]
                best_epoch = epoch
                best_mAP = mAP
                best_mINP = mINP
                state1 = {
                    "main_net1": main_net1.state_dict(),
                    "cmc": cmc,
                    "mAP": mAP,
                    "mINP": mINP,
                    "epoch": epoch,
                    "n_class": n_classes,
                }
                state2 = {
                    "main_net2": main_net2.state_dict(),
                    "cmc": cmc,
                    "mAP": mAP,
                    "mINP": mINP,
                    "epoch": epoch,
                    "n_class": n_classes,
                }
                torch.save(state1, os.path.join(model_path, "best_checkpoint1.pth"))
                torch.save(state2, os.path.join(model_path, "best_checkpoint2.pth"))
            print("Performance: Rank-1: {:.2%} | Rank-5: {:.2%} | Rank-10: {:.2%}| Rank-20: {:.2%}| mAP: {:.2%}| " \
                  "mINP: {:.2%}".format(cmc[0], cmc[4], cmc[9], cmc[19], mAP, mINP))
            print("Performance: Rank-1: {:.2%} | Rank-5: {:.2%} | Rank-10: {:.2%}| Rank-20: {:.2%}| mAP: {:.2%}| " \
                  "mINP: {:.2%}".format(cmc[0], cmc[4], cmc[9], cmc[19], mAP, mINP), file=test_os_log)
            print("Best Epoch [{}], Rank-1: {:.2%} |  mAP: {:.2%}| mINP: {:.2%}".format(best_epoch, best_acc,
                                                                                        best_mAP, best_mINP))
            print('Best Epoch [{}], Rank-1: {:.2%} |  mAP: {:.2%}| mINP: {:.2%}'.format(best_epoch, best_acc, best_mAP,
                                                                                        best_mINP), file=test_os_log)

            test_os_log.flush()

        torch.cuda.empty_cache()  # nvidia-smi memory release


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="OTLA-ReID for training")
    parser.add_argument("--config", default="config/config_sysu.yaml", help="config file")
    parser.add_argument("--resume", action="store_true", help="resume from checkpoint")
    parser.add_argument("--resume_path1", default="", help="checkpoint path1")
    parser.add_argument("--resume_path2", default="", help="checkpoint path2")

    args_main = parser.parse_args()
    args = yaml.load(open(args_main.config), Loader=yaml.FullLoader)
    args = easydict.EasyDict(args)

    main_worker(args, args_main)
