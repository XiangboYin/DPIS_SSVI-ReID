import time
import numpy as np
import torch
from torch.autograd import Variable
from utils import AverageMeter
from eval_metrics import eval_regdb, eval_sysu
import torch.nn as nn
import faiss
from sklearn.cluster import DBSCAN
from torch.nn import functional as F
from IPython import embed

criterion_CE = nn.CrossEntropyLoss(reduction='none')


def trainer(args, epoch, main_net, adjust_learning_rate, optimizer, trainloader, criterion, prob_I, writer=None,
            print_freq=50):
    current_lr = adjust_learning_rate(args, optimizer, epoch)

    total_loss = AverageMeter()
    id_loss_rgb = AverageMeter()
    id_loss_ir = AverageMeter()
    tri_loss_rgb = AverageMeter()
    tri_loss_ir = AverageMeter()
    dis_loss = AverageMeter()
    pa_loss = AverageMeter()
    robust_tri_loss = AverageMeter()

    batch_time = AverageMeter()

    correct_tri_rgb = 0
    correct_tri_ir = 0
    pre_rgb = 0  # it is meaningful only in the case of semi supervised setting
    pre_ir = 0  # it is meaningful only in the case of semi supervised setting
    pre_rgb_ir = 0  # it is meaningful only in the case of semi supervised setting, whether labels of selected samples per batch are equal
    num_rgb = 0
    num_ir = 0

    main_net.train()  # switch to train mode
    end = time.time()

    for batch_id, (input_rgb0, input_rgb1, input_ir, label_rgb, label_ir, index_V, index_I) in enumerate(trainloader):
        # label_ir is only used to calculate the prediction accuracy of pseudo infrared labels on semi-supervised setting
        # label_ir is meaningless on unsupervised setting
        # for supervised setting, we change "label_rgb" of "loss_id_ir" and "loss_tri_ir" into "label_ir"
        input_rgb = torch.cat((input_rgb0, input_rgb1,), 0)
        label_rgb = torch.cat((label_rgb, label_rgb), 0)
        labels = torch.cat((label_rgb, label_ir), 0)
        labels = labels.cuda()
        label_rgb = label_rgb.cuda()
        label_ir = label_ir.cuda()
        input_rgb = input_rgb.cuda()
        input_ir = input_ir.cuda()

        feat, output_cls, output_dis = main_net(input_rgb, input_ir, modal=0, train_set=True)
        loss_id_rgb = criterion[0](output_cls[:input_rgb.size(0)], label_rgb)
        loss_tri_rgb, correct_tri_batch_rgb = criterion[1](feat[:input_rgb.size(0)], label_rgb)
        prob_I_batch = prob_I[index_I]
        prob_I_batch = torch.tensor(prob_I_batch)
        prob_V_batch = torch.ones_like(prob_I_batch)
        prob = torch.cat((prob_V_batch, prob_V_batch, prob_I_batch), 0)
        prob.cuda()
        device = torch.device("cuda")  # 选择第一个可用的 GPU 设备
        prob_I_batch = prob_I_batch.to(device)  # 将张量移动到 GPU
        loss_id_ir = criterion_CE(output_cls[input_rgb.size(0):], label_ir)
        loss_id_ir = loss_id_ir * prob_I_batch
        loss_id_ir = loss_id_ir.sum() / 32
        if args.setting == "semi-supervised" or args.setting == "unsupervised":
            loss_tri_ir, correct_tri_batch_ir = criterion[1](feat[input_rgb.size(0):], label_ir)
            loss_tri, batch_acc, cnt = criterion[4](feat, output_cls, labels, labels, prob, threshold=0.5)

        elif args.setting == "supervised":
            loss_tri_ir, correct_tri_batch_ir = criterion[1](feat[input_rgb.size(0):], label_ir)
            loss_tri, batch_acc, cnt = criterion[4](feat, output_cls, labels, labels, prob, threshold=0.5)

        dis_label = torch.cat((torch.ones(input_rgb.size(0)), torch.zeros(input_ir.size(0))), dim=0).cuda()
        loss_dis = criterion[2](output_dis.view(-1), dis_label)

        loss_pa, sim_rgbtoir, sim_irtorgb = criterion[3](output_cls[:input_rgb.size(0)], output_cls[input_rgb.size(0):])

        loss = loss_id_rgb + loss_tri_rgb + 0.1 * loss_id_ir + 0.5 * loss_tri_ir + loss_dis + loss_pa + loss_tri

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        correct_tri_rgb += correct_tri_batch_rgb
        correct_tri_ir += correct_tri_batch_ir
        _, pre_label = output_cls.max(1)
        pre_batch_rgb = (pre_label[:input_rgb.size(0)].eq(label_rgb).sum().item())
        pre_batch_ir = (pre_label[input_rgb.size(0):].eq(label_ir).sum().item())
        pre_batch_rgb_ir = (label_rgb[:32].eq(label_ir).sum().item())
        pre_rgb += pre_batch_rgb
        pre_ir += pre_batch_ir
        pre_rgb_ir += pre_batch_rgb_ir
        num_rgb += input_rgb.size(0)
        num_ir += input_ir.size(0)
        # assert num_rgb == num_ir

        total_loss.update(loss.item(), input_rgb.size(0) + input_ir.size(0))
        id_loss_rgb.update(loss_id_rgb.item(), input_rgb.size(0))
        id_loss_ir.update(loss_id_ir.item(), input_ir.size(0))
        tri_loss_rgb.update(loss_tri_rgb, input_rgb.size(0))
        tri_loss_ir.update(loss_tri_ir, input_ir.size(0))
        dis_loss.update(loss_dis, input_rgb.size(0) + input_ir.size(0))
        pa_loss.update(loss_pa.item(), input_rgb.size(0) + input_ir.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if batch_id % print_freq == 0:
            print("Epoch: [{}][{}/{}] "
                  "Time: {batch_time.val:.3f} ({batch_time.avg:.3f}) "
                  "Lr: {:.6f} "
                  "Coeff: {:.3f} "
                  "Total_Loss: {total_loss.val:.4f}({total_loss.avg:.4f}) "
                  "ID_Loss_RGB: {id_loss_rgb.val:.4f}({id_loss_rgb.avg:.4f}) "
                  "ID_Loss_IR: {id_loss_ir.val:.4f}({id_loss_ir.avg:.4f}) "
                  "Tri_Loss_RGB: {tri_loss_rgb.val:.4f}({tri_loss_rgb.avg:.4f}) "
                  "Tri_Loss_IR: {tri_loss_ir.val:.4f}({tri_loss_ir.avg:.4f}) "
                  "Dis_Loss: {dis_loss.val:.4f}({dis_loss.avg:.4f}) "
                  "Pa_Loss: {pa_loss.val:.4f}({pa_loss.avg:.4f}) "
                  "Tri_RGB_Acc: {:.2f}% "
                  "Tri_IR_Acc: {:.2f}% "
                  "Pre_RGB_Acc: {:.2f}% "
                  "Pre_IR_Acc: {:.2f}% "
                  "Pre_RGB_IR_Acc: {:.2f}% ".format(epoch, batch_id, len(trainloader), current_lr,
                                                    main_net.adnet.coeff,
                                                    100. * correct_tri_rgb / num_rgb,
                                                    100. * correct_tri_ir / num_ir,
                                                    100. * pre_rgb / num_rgb,
                                                    100. * pre_ir / num_ir,
                                                    100. * pre_rgb_ir / num_rgb,
                                                    batch_time=batch_time,
                                                    total_loss=total_loss,
                                                    id_loss_rgb=id_loss_rgb,
                                                    id_loss_ir=id_loss_ir,
                                                    tri_loss_rgb=tri_loss_rgb,
                                                    tri_loss_ir=tri_loss_ir,
                                                    dis_loss=dis_loss,
                                                    pa_loss=pa_loss))

    if writer is not None:
        writer.add_scalar("Lr", current_lr, epoch)
        writer.add_scalar("Coeff", main_net.adnet.coeff, epoch)
        writer.add_scalar("Total_Loss", total_loss.avg, epoch)
        writer.add_scalar("ID_Loss_RGB", id_loss_rgb.avg, epoch)
        writer.add_scalar("ID_Loss_IR", id_loss_ir.avg, epoch)
        writer.add_scalar("Tri_Loss_RGB", tri_loss_rgb.avg, epoch)
        writer.add_scalar("Tri_Loss_IR", tri_loss_ir.avg, epoch)
        writer.add_scalar("Dis_Loss", dis_loss.avg, epoch)
        writer.add_scalar("Pa_Loss", pa_loss.avg, epoch)
        writer.add_scalar("Tri_RGB_Acc", 100. * correct_tri_rgb / num_rgb, epoch)
        writer.add_scalar("Tri_IR_Acc", 100. * correct_tri_ir / num_ir, epoch)
        writer.add_scalar("Pre_RGB_Acc", 100. * pre_rgb / num_rgb, epoch)
        writer.add_scalar("Pre_IR_Acc", 100. * pre_ir / num_ir, epoch)


def tester(args, epoch, main_net1, main_net2, test_mode, gall_label, gall_loader, query_label, query_loader,
           feat_dim=2048, query_cam=None, gall_cam=None, writer=None):
    # switch to evaluation mode
    main_net1.eval()
    main_net2.eval()

    print("Extracting Gallery Feature...")
    ngall = len(gall_label)
    start = time.time()
    ptr = 0
    gall_feat = np.zeros((ngall, feat_dim))
    with torch.no_grad():
        for batch_idx, (input, label) in enumerate(gall_loader):
            batch_num = input.size(0)
            input = Variable(input.cuda())
            feat1 = main_net1(input, input, modal=test_mode[0])
            feat2 = main_net2(input, input, modal=test_mode[0])
            feat = (feat1 + feat2) / 2
            gall_feat[ptr:ptr + batch_num, :] = feat.detach().cpu().numpy()
            ptr = ptr + batch_num
    print("Extracting Time:\t {:.3f}".format(time.time() - start))

    print("Extracting Query Feature...")
    nquery = len(query_label)
    start = time.time()
    ptr = 0
    query_feat = np.zeros((nquery, feat_dim))
    with torch.no_grad():
        for batch_idx, (input, label) in enumerate(query_loader):
            batch_num = input.size(0)
            input = Variable(input.cuda())
            feat1 = main_net1(input, input, modal=test_mode[1])
            feat2 = main_net2(input, input, modal=test_mode[1])
            feat = (feat1 + feat2) / 2
            query_feat[ptr:ptr + batch_num, :] = feat.detach().cpu().numpy()
            ptr = ptr + batch_num
    print("Extracting Time:\t {:.3f}".format(time.time() - start))

    start = time.time()
    # compute the similarity
    distmat = -np.matmul(query_feat, np.transpose(gall_feat))
    # evaluation
    if args.dataset == "sysu":
        cmc, mAP, mINP = eval_sysu(distmat, query_label, gall_label, query_cam, gall_cam)
    elif args.dataset == "regdb":
        cmc, mAP, mINP = eval_regdb(distmat, query_label, gall_label)
    print("Evaluation Time:\t {:.3f}".format(time.time() - start))

    if writer is not None:
        writer.add_scalar("Rank1", cmc[0], epoch)
        writer.add_scalar("mAP", mAP, epoch)
        writer.add_scalar("mINP", mINP, epoch)

    return cmc, mAP, mINP


def tester_full(args, epoch, main_net, test_mode, gall_label, gall_loader, query_label, query_loader, feat_dim=2048,
                query_cam=None, gall_cam=None, writer=None):
    # switch to evaluation mode
    main_net.eval()

    print("Extracting Gallery Feature...")
    ngall = len(gall_label)
    start = time.time()
    ptr = 0
    gall_feat = np.zeros((ngall, feat_dim))
    with torch.no_grad():
        for batch_idx, (input, label) in enumerate(gall_loader):
            batch_num = input.size(0)
            input = Variable(input.cuda())
            feat = main_net(input, input, modal=test_mode[0])
            gall_feat[ptr:ptr + batch_num, :] = feat.detach().cpu().numpy()
            ptr = ptr + batch_num
    print("Extracting Time:\t {:.3f}".format(time.time() - start))

    print("Extracting Query Feature...")
    nquery = len(query_label)
    start = time.time()
    ptr = 0
    query_feat = np.zeros((nquery, feat_dim))
    with torch.no_grad():
        for batch_idx, (input, label) in enumerate(query_loader):
            batch_num = input.size(0)
            input = Variable(input.cuda())
            feat = main_net(input, input, modal=test_mode[1])
            query_feat[ptr:ptr + batch_num, :] = feat.detach().cpu().numpy()
            ptr = ptr + batch_num
    print("Extracting Time:\t {:.3f}".format(time.time() - start))

    start = time.time()
    # compute the similarity
    distmat = -np.matmul(query_feat, np.transpose(gall_feat))
    # evaluation
    if args.dataset == "sysu":
        cmc, mAP, mINP = eval_sysu(distmat, query_label, gall_label, query_cam, gall_cam)
    elif args.dataset == "regdb":
        cmc, mAP, mINP = eval_regdb(distmat, query_label, gall_label)
    print("Evaluation Time:\t {:.3f}".format(time.time() - start))

    if writer is not None:
        writer.add_scalar("Rank1", cmc[0], epoch)
        writer.add_scalar("mAP", mAP, epoch)
        writer.add_scalar("mINP", mINP, epoch)

    return cmc, mAP, mINP


def evaler(args, main_net, eval_loader, n_thermal_label):
    losses_I = -1. * torch.ones(n_thermal_label)
    main_net.train()
    with torch.no_grad():
        for batch_idx, (input1, input2, label1, label2, index_V, index_I) in enumerate(eval_loader):
            input1 = input1.cuda()
            input2 = input2.cuda()
            label1 = label1.cuda()
            label2 = label2.cuda()

            # index_V = np.concatenate((index_V, index_V), 0)
            labels = torch.cat((label1, label2), 0)
            labels = labels.long()
            # _, out0, = main_net(input1, input2)
            feat, output_cls, output_dis = main_net(input1, input2, modal=0, train_set=True)
            # print(output_cls.shape)

            # print("batch:{}, output:{}, labels:{}".format(batch_idx, output_cls.shape, labels.shape))
            # print("batch:{}, output:{}, labels:{}".format(batch_idx, output_cls.max(), labels.max()))
            # print("batch:{}, output:{}, labels:{}".format(batch_idx, output_cls.min(), labels.min()))

            loss = criterion_CE(output_cls, labels)
            loss1 = loss[0:32]
            loss2 = loss[32:64]

            # for n1 in range(input2.size(0)):
            #     losses_V_aug1[index_V[n1]] = loss1[n1]
            # try:
            for n2 in range(input2.size(0)):
                losses_I[index_I[n2]] = loss2[n2]
            # except:
            #     continue

    losses_I_slt = (losses_I - losses_I.min()) / (losses_I.max() - losses_I.min())
    input_loss_I = losses_I_slt.reshape(-1, 1)
    from sklearn.mixture import GaussianMixture
    gmm_I = GaussianMixture(n_components=2, max_iter=100, tol=1e-2, reg_covar=5e-4)
    gmm_I.fit(input_loss_I)
    prob_I = gmm_I.predict_proba(input_loss_I)
    prob_I = prob_I[:, gmm_I.means_.argmin()]
    return prob_I


def cluster(args, main_net, eval_loader, n_thermal_label):
    main_net.eval()
    batch_time = AverageMeter()
    data_time = AverageMeter()

    labels = []
    end = time.time()
    features = [[torch.ones(2048)] for _ in range(n_thermal_label)]
    with torch.no_grad():
        for batch_idx, (input1, input2, label1, label2, index_V, index_I) in enumerate(eval_loader):
            data_time.update(time.time() - end)
            outputs = extract_cnn_feature(main_net, input2)
            batch_time.update(time.time() - end)
            end = time.time()
            labels.extend(label2)

            for n2 in range(input2.size(0)):
                features[index_I[n2]] = outputs[n2]
            if (batch_idx + 1) % 50 == 0:
                print('Extract Features: [{}/{}]\t'
                      'Time {:.3f} ({:.3f})\t'
                      'Data {:.3f} ({:.3f})\t'
                      .format(batch_idx + 1, len(eval_loader),
                              batch_time.val, batch_time.avg,
                              data_time.val, data_time.avg))
    cf = torch.stack(list(features))
    labels_t = torch.stack(list(labels))
    rerank_dist = compute_jaccard_distance(cf, k1=30, k2=6)
    # eps = args.eps
    eps = 0.5
    print('eps in cluster: {:.3f}'.format(eps))
    print('Clustering and labeling...')
    # DBSCAN
    cluster = DBSCAN(eps=eps, min_samples=4, metric='precomputed', n_jobs=-1)
    cluster_labels = cluster.fit_predict(rerank_dist)
    return cluster_labels


def extract_cnn_feature(model, inputs):
    inputs = to_torch(inputs).cuda()
    outputs = model(inputs, inputs, 2)
    outputs = outputs.data.cpu()
    return outputs


def to_torch(ndarray):
    if type(ndarray).__module__ == 'numpy':
        return torch.from_numpy(ndarray)
    elif not torch.is_tensor(ndarray):
        raise ValueError("Cannot convert {} to torch tensor"
                         .format(type(ndarray)))
    return ndarray


def compute_jaccard_distance(target_features, k1=20, k2=6, print_flag=True, search_option=0, use_float16=False):
    end = time.time()
    if print_flag:
        print('Computing jaccard distance...')

    ngpus = faiss.get_num_gpus()
    N = target_features.size(0)
    mat_type = np.float16 if use_float16 else np.float32

    if (search_option == 0):
        # GPU + PyTorch CUDA Tensors (1)
        res = faiss.StandardGpuResources()
        res.setDefaultNullStreamAllDevices()
        _, initial_rank = search_raw_array_pytorch(res, target_features, target_features, k1)
        initial_rank = initial_rank.cpu().numpy()
    elif (search_option == 1):
        # GPU + PyTorch CUDA Tensors (2)
        res = faiss.StandardGpuResources()
        index = faiss.GpuIndexFlatL2(res, target_features.size(-1))
        index.add(target_features.cpu().numpy())
        _, initial_rank = search_index_pytorch(index, target_features, k1)
        res.syncDefaultStreamCurrentDevice()
        initial_rank = initial_rank.cpu().numpy()
    elif (search_option == 2):
        # GPU
        index = index_init_gpu(ngpus, target_features.size(-1))
        index.add(target_features.cpu().numpy())
        _, initial_rank = index.search(target_features.cpu().numpy(), k1)
    else:
        # CPU
        index = index_init_cpu(target_features.size(-1))
        index.add(target_features.cpu().numpy())
        _, initial_rank = index.search(target_features.cpu().numpy(), k1)

    nn_k1 = []
    nn_k1_half = []
    for i in range(N):
        nn_k1.append(k_reciprocal_neigh(initial_rank, i, k1))
        nn_k1_half.append(k_reciprocal_neigh(initial_rank, i, int(np.around(k1 / 2))))

    V = np.zeros((N, N), dtype=mat_type)
    for i in range(N):
        k_reciprocal_index = nn_k1[i]
        k_reciprocal_expansion_index = k_reciprocal_index
        for candidate in k_reciprocal_index:
            candidate_k_reciprocal_index = nn_k1_half[candidate]
            if (len(np.intersect1d(candidate_k_reciprocal_index, k_reciprocal_index)) > 2 / 3 * len(
                    candidate_k_reciprocal_index)):
                k_reciprocal_expansion_index = np.append(k_reciprocal_expansion_index, candidate_k_reciprocal_index)

        k_reciprocal_expansion_index = np.unique(k_reciprocal_expansion_index)  ## element-wise unique
        dist = 2 - 2 * torch.mm(target_features[i].unsqueeze(0).contiguous(),
                                target_features[k_reciprocal_expansion_index].t())
        if use_float16:
            V[i, k_reciprocal_expansion_index] = F.softmax(-dist, dim=1).view(-1).cpu().numpy().astype(mat_type)
        else:
            V[i, k_reciprocal_expansion_index] = F.softmax(-dist, dim=1).view(-1).cpu().numpy()

    del nn_k1, nn_k1_half

    if k2 != 1:
        V_qe = np.zeros_like(V, dtype=mat_type)
        for i in range(N):
            V_qe[i, :] = np.mean(V[initial_rank[i, :k2], :], axis=0)
        V = V_qe
        del V_qe

    del initial_rank

    invIndex = []
    for i in range(N):
        invIndex.append(np.where(V[:, i] != 0)[0])  # len(invIndex)=all_num

    jaccard_dist = np.zeros((N, N), dtype=mat_type)
    for i in range(N):
        temp_min = np.zeros((1, N), dtype=mat_type)
        # temp_max = np.zeros((1,N), dtype=mat_type)
        indNonZero = np.where(V[i, :] != 0)[0]
        indImages = []
        indImages = [invIndex[ind] for ind in indNonZero]
        for j in range(len(indNonZero)):
            temp_min[0, indImages[j]] = temp_min[0, indImages[j]] + np.minimum(V[i, indNonZero[j]],
                                                                               V[indImages[j], indNonZero[j]])
            # temp_max[0,indImages[j]] = temp_max[0,indImages[j]]+np.maximum(V[i,indNonZero[j]],V[indImages[j],indNonZero[j]])

        jaccard_dist[i] = 1 - temp_min / (2 - temp_min)
        # jaccard_dist[i] = 1-temp_min/(temp_max+1e-6)

    del invIndex, V

    pos_bool = (jaccard_dist < 0)
    jaccard_dist[pos_bool] = 0.0
    if print_flag:
        print("Jaccard distance computing time cost: {}".format(time.time() - end))

    return jaccard_dist


def k_reciprocal_neigh(initial_rank, i, k1):
    forward_k_neigh_index = initial_rank[i, :k1 + 1]
    backward_k_neigh_index = initial_rank[forward_k_neigh_index, :k1 + 1]
    fi = np.where(backward_k_neigh_index == i)[0]
    return forward_k_neigh_index[fi]


def search_raw_array_pytorch(res, xb, xq, k, D=None, I=None,
                             metric=faiss.METRIC_L2):
    assert xb.device == xq.device

    nq, d = xq.size()
    if xq.is_contiguous():
        xq_row_major = True
    elif xq.t().is_contiguous():
        xq = xq.t()  # I initially wrote xq:t(), Lua is still haunting me :-)
        xq_row_major = False
    else:
        raise TypeError('matrix should be row or column-major')

    xq_ptr = swig_ptr_from_FloatTensor(xq)

    nb, d2 = xb.size()
    assert d2 == d
    if xb.is_contiguous():
        xb_row_major = True
    elif xb.t().is_contiguous():
        xb = xb.t()
        xb_row_major = False
    else:
        raise TypeError('matrix should be row or column-major')
    xb_ptr = swig_ptr_from_FloatTensor(xb)

    if D is None:
        D = torch.empty(nq, k, device=xb.device, dtype=torch.float32)
    else:
        assert D.shape == (nq, k)
        assert D.device == xb.device

    if I is None:
        I = torch.empty(nq, k, device=xb.device, dtype=torch.int64)
    else:
        assert I.shape == (nq, k)
        assert I.device == xb.device

    D_ptr = swig_ptr_from_FloatTensor(D)
    I_ptr = swig_ptr_from_LongTensor(I)

    faiss.bruteForceKnn(res, metric,
                        xb_ptr, xb_row_major, nb,
                        xq_ptr, xq_row_major, nq,
                        d, k, D_ptr, I_ptr)

    return D, I


def swig_ptr_from_FloatTensor(x):
    assert x.is_contiguous()
    assert x.dtype == torch.float32
    return faiss.cast_integer_to_float_ptr(
        x.storage().data_ptr() + x.storage_offset() * 4)


def search_index_pytorch(index, x, k, D=None, I=None):
    """call the search function of an index with pytorch tensor I/O (CPU
    and GPU supported)"""
    assert x.is_contiguous()
    n, d = x.size()
    assert d == index.d

    if D is None:
        D = torch.empty((n, k), dtype=torch.float32, device=x.device)
    else:
        assert D.size() == (n, k)

    if I is None:
        I = torch.empty((n, k), dtype=torch.int64, device=x.device)
    else:
        assert I.size() == (n, k)
    torch.cuda.synchronize()
    xptr = swig_ptr_from_FloatTensor(x)
    Iptr = swig_ptr_from_LongTensor(I)
    Dptr = swig_ptr_from_FloatTensor(D)
    index.search_c(n, xptr,
                   k, Dptr, Iptr)
    torch.cuda.synchronize()
    return D, I


def swig_ptr_from_LongTensor(x):
    assert x.is_contiguous()
    assert x.dtype == torch.int64, 'dtype=%s' % x.dtype
    return faiss.cast_integer_to_long_ptr(
        x.storage().data_ptr() + x.storage_offset() * 8)


def index_init_gpu(ngpus, feat_dim):
    flat_config = []
    for i in range(ngpus):
        cfg = faiss.GpuIndexFlatConfig()
        cfg.useFloat16 = False
        cfg.device = i
        flat_config.append(cfg)

    res = [faiss.StandardGpuResources() for i in range(ngpus)]
    indexes = [faiss.GpuIndexFlatL2(res[i], feat_dim, flat_config[i]) for i in range(ngpus)]
    index = faiss.IndexShards(feat_dim)
    for sub_index in indexes:
        index.add_shard(sub_index)
    index.reset()
    return index


def index_init_cpu(feat_dim):
    return faiss.IndexFlatL2(feat_dim)
