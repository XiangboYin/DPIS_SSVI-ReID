import torch
import torch.nn as nn
import numpy as np

def normalize(x, axis=-1):
    """
    Normalizing to unit length along the specified dimension.
    """
    x = 1. * x / (torch.norm(x, 2, axis, keepdim=True).expand_as(x) + 1e-12)
    return x


class TripletLoss(nn.Module):
    """
    Triplet loss with hard positive/negative mining.
    Reference: Hermans et al. In Defense of the Triplet Loss for Person Re-Identification. arXiv:1703.07737.
    Code imported from https://github.com/Cysu/open-reid/blob/master/reid/loss/triplet.py.
    Args:
    - margin (float): margin for triplet.
    - inputs: feature matrix with shape (batch_size, feat_dim).
    - targets: ground truth labels with shape (num_classes).
    """
    def __init__(self, margin=0.3):
        super(TripletLoss, self).__init__()
        self.margin = margin
        self.ranking_loss = nn.MarginRankingLoss(margin=margin)

    def forward(self, inputs, targets):
        n = inputs.size(0)

        # Compute pairwise distance, replace by the official when merged
        dist = torch.pow(inputs, 2).sum(dim=1, keepdim=True).expand(n, n)
        dist = dist + dist.t()
        dist.addmm_(1, -2, inputs, inputs.t())
        dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability

        # For each anchor, find the hardest positive and negative
        mask = targets.expand(n, n).eq(targets.expand(n, n).t())
        dist_ap, dist_an = [], []
        for i in range(n):
            dist_ap.append(dist[i][mask[i]].max().unsqueeze(0))
            dist_an.append(dist[i][mask[i] == 0].min().unsqueeze(0))
        dist_ap = torch.cat(dist_ap)
        dist_an = torch.cat(dist_an)

        # Compute ranking hinge loss
        y = torch.ones_like(dist_an)
        loss = self.ranking_loss(dist_an, dist_ap, y)

        # compute accuracy
        correct = torch.ge(dist_an, dist_ap).sum().item()  # torch.eq: greater than or equal to >=

        return loss, correct


class PredictionAlignmentLoss(nn.Module):
    """
    Proposed loss for Prediction Alignment Learning (PAL).
    """
    def __init__(self, lambda_vr=0.1, lambda_rv=0.5):
        super(PredictionAlignmentLoss, self).__init__()
        self.lambda_vr = lambda_vr
        self.lambda_rv = lambda_rv

    def forward(self, x_rgb, x_ir):
        sim_rgbtoir = torch.mm(normalize(x_rgb), normalize(x_ir).t())
        sim_irtorgb = torch.mm(normalize(x_ir), normalize(x_rgb).t())
        sim_irtoir = torch.mm(normalize(x_ir), normalize(x_ir).t())

        sim_rgbtoir = nn.Softmax(1)(sim_rgbtoir)
        sim_irtorgb = nn.Softmax(1)(sim_irtorgb)
        sim_irtoir = nn.Softmax(1)(sim_irtoir)

        KL_criterion = nn.KLDivLoss(reduction="batchmean")

        x_rgbtoir = torch.mm(sim_rgbtoir, x_ir)
        x_irtorgb = torch.mm(sim_irtorgb, x_rgb)
        x_irtoir = torch.mm(sim_irtoir, x_ir)

        x_rgb_s = nn.Softmax(1)(x_rgb)
        x_rgbtoir_ls = nn.LogSoftmax(1)(x_rgbtoir)
        x_irtorgb_s = nn.Softmax(1)(x_irtorgb)
        x_irtoir_ls = nn.LogSoftmax(1)(x_irtoir)

        loss_rgbtoir = KL_criterion(x_rgbtoir_ls, x_rgb_s)
        loss_irtorgb = KL_criterion(x_irtoir_ls, x_irtorgb_s)

        loss = self.lambda_vr * loss_rgbtoir + self.lambda_rv * loss_irtorgb

        return loss, sim_rgbtoir, sim_irtorgb

class RobustTripletLoss_final(nn.Module):
    def __init__(self, batch_size, margin):
        super(RobustTripletLoss_final, self).__init__()
        self.batch_size = batch_size
        self.margin = margin
        # self.T=1#V4
        # self.T=0.1#V6
        self.T = 0.1  # V4


    def forward(self, inputs, prediction, targets, true_targets, prob, threshold):

        n = inputs.size(0)
        # Compute pairwise distance, replace by the official when merged
        dist = torch.pow(inputs, 2).sum(dim=1, keepdim=True).expand(n, n)
        dist = dist + dist.t()
        dist.addmm_(1, -2, inputs, inputs.t())
        dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability

        # For each anchor, find the positive and negative
        is_pos = targets.expand(n, n).eq(targets.expand(n, n).t())
        is_neg = targets.expand(n, n).ne(targets.expand(n, n).t())
        is_confident = (prob >= threshold)
        dist_ap, dist_an = [], []
        cnt, loss = 0, 0
        tnt=0
        loss_inverse = False
        K = 20

        for i in range(n):
            # print(i)
            if is_confident[i]:
            # if 0:

                pos_idx = (torch.nonzero(is_pos[i].long())).squeeze(1)
                neg_idx = (torch.nonzero(is_neg[i].long())).squeeze(1)

                random_pos_index = int(np.random.choice(pos_idx.cpu().numpy(), 1))
                endwhile=0
                while random_pos_index == i:
                    endwhile+=1
                    random_pos_index = int(np.random.choice(pos_idx.cpu().numpy(), 1))
                    if endwhile>10:
                        break

                rank_neg_index = dist[i][neg_idx].argsort()
                hard_neg_index = rank_neg_index[0]
                hard_neg_index = neg_idx[hard_neg_index]

                dist_ap.append(dist[i][random_pos_index].unsqueeze(0))
                dist_an.append(dist[i][hard_neg_index].unsqueeze(0))

                if prob[random_pos_index] >= threshold and prob[hard_neg_index] >= threshold:
                    # TP-TN
                    pass

                elif prob[random_pos_index] >= threshold and prob[hard_neg_index] < threshold:
                    is_FN = (torch.argmax(prediction[hard_neg_index]) == targets[i])
                    # TP-FN
                    if is_FN:
                        tmp = rank_neg_index[1]
                        hard_neg_index_new = neg_idx[tmp]
                        j = 1
                        loop_cnt = 0
                        while prob[hard_neg_index_new] < threshold:
                            j += 1
                            tmp = rank_neg_index[j]
                            hard_neg_index_new = neg_idx[tmp]
                            loop_cnt += 1
                            if loop_cnt >= 10:
                                # print("------------warning, break the death loop---------------")
                                break
                        dist_ap[cnt] = (dist[i][random_pos_index].unsqueeze(0) +
                                        dist[i][hard_neg_index].unsqueeze(0)) / 2
                        dist_an[cnt] = dist[i][hard_neg_index_new].unsqueeze(0)
                    # TP-TN
                    else:
                        pass

                elif prob[random_pos_index] < threshold and prob[hard_neg_index] >= threshold:
                    # FP-TN
                    random_pos_index_new = int(np.random.choice(pos_idx.cpu().numpy(), 1))
                    loop_cnt = 0
                    while random_pos_index_new == i or prob[random_pos_index_new] < threshold:
                        random_pos_index_new = int(np.random.choice(pos_idx.cpu().numpy(), 1))
                        loop_cnt += 1
                        if loop_cnt >= 5:
                            # print("------------warning, break the death loop---------------")
                            break
                    dist_an[cnt] = (dist[i][random_pos_index].unsqueeze(0)
                                    + dist[i][hard_neg_index].unsqueeze(0)) / 2
                    dist_ap[cnt] = dist[i][random_pos_index_new].unsqueeze(0)

                elif prob[random_pos_index] < threshold and prob[hard_neg_index] < threshold:
                    is_FN = (torch.argmax(prediction[hard_neg_index]) == targets[i])
                    # FP-FN
                    if is_FN:
                        loss_inverse = True
                    # FP-TN
                    else:
                        random_pos_index_new = int(np.random.choice(pos_idx.cpu().numpy(), 1))
                        loop_cnt = 0
                        while random_pos_index_new == i or prob[random_pos_index_new] < threshold:
                            random_pos_index_new = int(np.random.choice(pos_idx.cpu().numpy(), 1))
                            loop_cnt += 1
                            if loop_cnt >= 5:
                                # print("------------warning, break the death loop---------------")
                                break
                        dist_an[cnt] = (dist[i][random_pos_index].unsqueeze(0)
                                        + dist[i][hard_neg_index].unsqueeze(0)) / 2
                        dist_ap[cnt] = dist[i][random_pos_index_new].unsqueeze(0)

                if loss_inverse:
                    loss += torch.clamp(dist_an[cnt] - dist_ap[cnt] + self.margin, 0)
                else:
                    # try:
                    loss += torch.clamp(dist_ap[cnt] - dist_an[cnt] + self.margin, 0)
                    # except:
                    #     continue
                cnt += 1
                tnt+=1
                loss_inverse = False
            # elif epoch<=0:
            #     continue
            else:
                cln=0.01
                # continue
                if i<=31:
                    ap_dis=dist[i][i+32]
                    # ap_dis=dist[i][i]

                    V_nagetive=dist[i][0:32]
                    I_nagetive = dist[i][64:]
                    V_an_dis, AV_indices = torch.sort(V_nagetive, dim=0, descending=True)
                    I_an_dis, AI_indices = torch.sort(I_nagetive, dim=0, descending=True)
                    V_an_dis = dist[i][AV_indices[0:K]]
                    I_an_dis = dist[i][AI_indices[0:K]]
                    V_an_dis=torch.sum(V_an_dis, dim=0)/K
                    I_an_dis=torch.sum(I_an_dis, dim=0)/K

                    loss = (torch.clamp(ap_dis - (I_an_dis)+ self.margin, 0)+torch.clamp(ap_dis - (V_an_dis)+ self.margin, 0))/2


                elif i>=32 and i<64:
                    # continue
                    ap_dis = dist[i][i - 32]
                    # ap_dis=dist[i][i]
                    V_nagetive = dist[i][32:64]
                    I_nagetive = dist[i][64:]
                    V_an_dis, AV_indices = torch.sort(V_nagetive, dim=0, descending=True)
                    I_an_dis, AI_indices = torch.sort(I_nagetive, dim=0, descending=True)

                    V_an_dis = dist[i][AV_indices[0:K]]
                    I_an_dis = dist[i][AI_indices[0:K]]
                    V_an_dis = torch.sum(V_an_dis, dim=0)/K
                    I_an_dis = torch.sum(I_an_dis, dim=0)/K

                    # loss += torch.clamp(ap_dis - (V_an_dis+ I_an_dis)+ self.margin, 0)
                    loss = (torch.clamp(ap_dis - (I_an_dis)+ self.margin, 0)+torch.clamp(ap_dis - (V_an_dis)+ self.margin, 0))/2


                else:
                    # continue
                    ap_dis = dist[i][i]
                    V_nagetive = dist[i][64:]
                    an_dis, AV_indices = torch.sort(V_nagetive, dim=0, descending=True)
                    an_dis = dist[i][AV_indices[0:K]]
                    an_dis = torch.sum(an_dis, dim=0)/K

                    loss += torch.clamp(ap_dis - an_dis + self.margin, 0)
                    # continue
                    # Anchor = inputs[i]
                    # Positive = Anchor
                    # V_nagetive = inputs[:64]
                    # I_nagetive = torch.cat((inputs[64:i], inputs[i:96]), 0)
                    # mat_sim_AV = torch.matmul(Anchor, V_nagetive.transpose(0, 1))
                    # mat_sim_AI = torch.matmul(Anchor, I_nagetive.transpose(0, 1))
                    # sorted_AV_distance, AV_indices = torch.sort(mat_sim_AV, dim=0, descending=False)
                    # sorted_AI_distance, AI_indices = torch.sort(mat_sim_AI, dim=0, descending=False)
                    # # Hard_AV_distance = sorted_AV_distance[0:5]
                    # # Hard_AI_distance = sorted_AI_distance[0:5]
                    # Hard_AV_distance = sorted_AV_distance[0:10]
                    # Hard_AI_distance = sorted_AI_distance[0:10]#6
                    # Positive_distance = torch.matmul(Anchor, Positive)
                    # loss_AV = -torch.log((torch.exp(Positive_distance / self.T)) / (
                    #         torch.exp(Positive_distance / self.T) + torch.sum(torch.exp(Hard_AV_distance / self.T),
                    #                                                           dim=0)))
                    # loss_AI = -torch.log((torch.exp(Positive_distance / self.T)) / (
                    #         torch.exp(Positive_distance / self.T) + torch.sum(torch.exp(Hard_AI_distance / self.T),
                    #                                                           dim=0)))
                    # loss += (loss_AV+loss_AI)*cln
                    # tnt += 1
                # loss=loss
                # print(loss)
                tnt += 1
                loss=loss.reshape(1,-1)
                    # cnt += 1
                    # print('loss_AV:', loss_AV)
                    # print('loss_AI:', loss_AI)
        # compute accuracy
        if cnt == 0:
            return torch.Tensor([0.]).to(inputs.device), 0, cnt
        else:
            dist_ap = torch.cat(dist_ap)
            dist_an = torch.cat(dist_an)
            correct = torch.ge(dist_an, dist_ap).sum().item()
            return loss / tnt, correct, cnt