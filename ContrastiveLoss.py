import torch
import torch.nn.functional as F
from hyptorch.pmath import dist_matrix

class HyperLoss(torch.nn.Module):
    def __init__(self):
        super(HyperLoss, self).__init__()
        self.loss = 0
    
    def forward(self, train_poincare_points, train_poincare_points_label):
        self.loss = 0
        for i in range(len(train_poincare_points)):
            A_i = 0
            P_i = 0
            p_sum = 0
            z_i = train_poincare_points[i]
            for a in range(len(train_poincare_points)):
                if i == a: continue
                A_i += torch.exp(-dist_matrix(z_i, train_poincare_points[a]))
            for p in range(len(train_poincare_points)):
                if i == p: continue
                if train_poincare_points_label[i] != train_poincare_points_label[p]: continue
                p_sum += torch.log(torch.exp(-dist_matrix(z_i, train_poincare_points[p])) / A_i)
                P_i += 1
            self.loss += -p_sum / P_i
        return self.loss

class ContrastiveLoss(torch.nn.Module):
    """
    Contrastive loss function.
    Based on: http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
    """

    def __init__(self, margin=100.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, output1, output2, label):
        euclidean_distance = F.pairwise_distance(output1, output2, keepdim = True)
        loss_contrastive = torch.mean((1-label) * torch.pow(euclidean_distance, 2) +
                                      (label) * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2))

        return loss_contrastive

class KLContrastiveLoss(torch.nn.Module):
    def __init__(self, margin=200.0):
        super(KLContrastiveLoss, self).__init__()
        self.margin = margin
    
    def forward(self, output1, output2, label):
        jsd_loss = JSD()
        alpha = 0.01
        KLD1 = self.KLD(output1[1], output1[2])
        KLD2 = self.KLD(output2[1], output2[2])
        jsd = jsd_loss(output1[0], output2[0])
        loss_contrastive = torch.mean((1-label) * torch.pow(jsd, 2) +
                                      (label) * torch.pow(torch.clamp(self.margin - jsd, min=0.0), 2))
        return loss_contrastive + KLD1 + KLD2

    def KLD(self, mu, logvar):
        return -0.5 * torch.sum(1+logvar - mu.pow(2) - logvar.exp())


class JSD(torch.nn.Module):
    def __init__(self):
        super(JSD, self).__init__()
        self.kl = torch.nn.KLDivLoss(reduction='batchmean', log_target=True)

    def forward(self, p: torch.tensor, q: torch.tensor):
        p, q = p.view(-1, p.size(-1)), q.view(-1, q.size(-1))
        m = torch.nan_to_num((0.5 * (p + q)).log())
        return 0.5 * (self.kl(m, torch.nan_to_num(p.log())) + self.kl(m, torch.nan_to_num(q.log())))