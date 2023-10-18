from random import shuffle
import torch
import torch.optim as optim
from dataset import BrainDataset, BrainUMAP, load_volume, BrainDatasetHyper, BrainSupportDataset
from ContrastiveLoss import HyperLoss
from torch.optim.lr_scheduler import ExponentialLR
from model import SiameseNetwork
from tqdm import tqdm
import numpy as np

from sklearn.metrics import confusion_matrix, accuracy_score
import copy
import matplotlib.pyplot as plt
import os
import random

from hyptorch.nn import ToPoincare
from hyptorch.pmath import poincare_mean, dist_matrix
import umap
import seaborn as sns

SEED = 42
def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic   = True
    torch.backends.cudnn.benchmark = True


def main():
    seed_everything(SEED)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dataset = BrainDatasetHyper(k_shot=5)
    test_dataset = BrainDataset(k_shot=5,training=False)
    support_dataset = BrainSupportDataset(test_dataset.test_support_patient_path, len(dataset.train_Subtype))
    UMAP_dataset = BrainUMAP()


    train_dataloader = torch.utils.data.DataLoader(dataset=dataset, batch_size = 1)
    test_dataloader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size = 1)
    support_dataloader = torch.utils.data.DataLoader(dataset=support_dataset, batch_size = 1)
    UMAP_dataloader = torch.utils.data.DataLoader(dataset=UMAP_dataset, batch_size = 1)

    dim = 256
    model = SiameseNetwork(z=dim)

    model.to(device)

    criterion = HyperLoss()
    e2p = ToPoincare(1, False, True, dim)
    margin = 50
    optimizer = optim.Adam(model.parameters(), lr=0.001, betas=(0.9,0.999))
    scheduler = ExponentialLR(optimizer, gamma=0.98)

    epochs = 100
    best_acc = 0
    best_acc_model = None

    for epoch in range(epochs):
        model.train()
        train_loss = []
        train_poincare_points = []
        train_poincare_points_label = []
        for data in tqdm(train_dataloader):
            vol1, label = data
            vol1 = vol1.to(device)

            output = model.forward_once(vol1)
            output = output.cpu()
            output = e2p(output)
            train_poincare_points.append(output)
            train_poincare_points_label.append(label)

        for data in tqdm(support_dataloader):
            vol1, label = data
            vol1 = vol1.to(device)

            output = model.forward_once(vol1)
            output = output.cpu()

            output = e2p(output)
            train_poincare_points.append(output)
            train_poincare_points_label.append(label)
        
        optimizer.zero_grad()
        loss = criterion(train_poincare_points, train_poincare_points_label)

        loss.backward()
        optimizer.step()
        train_loss.append(loss.item())
        epoch_loss = np.mean(train_loss)
        print("[{}/{}], Train Loss: {}".format(epoch, epochs, epoch_loss / len(train_poincare_points)))

        scheduler.step()
        model.eval()

        poincare_mean_list = []
        with torch.no_grad():
            y_true = []
            y_pred = []

            for patient_type in test_dataset.test_Subtype:
                supports = []
                for support in test_dataset.test_support_patient_path[patient_type]:
                    vol1 = load_volume(patient_type, support)

                    vol1 = np.expand_dims(vol1, axis=0)
                    vol1 = np.expand_dims(vol1, axis=0)

                    vol1 = torch.from_numpy(vol1)
                    vol1 = vol1.to(device)

                    output = model.forward_once(vol1)
                    output = output.detach().cpu()

                    output = e2p(output)
                    supports.append(output[0].tolist())
                supports_poincare_mean = poincare_mean(torch.Tensor(np.array(supports)), dim=0, c= e2p.c)
                poincare_mean_list.append(supports_poincare_mean)

            for i, data in tqdm(enumerate(test_dataloader)):
                path = test_dataset.test_query_patient_path[i]
                vol2, label = data
                distance = []
                vol2 = np.expand_dims(vol2, axis=0)
                vol2 = torch.from_numpy(vol2)
                vol2 = vol2.to(device)
                
                output = model.forward_once(vol2)
                output = output.detach().cpu()
                output = e2p(output)

                for poincare_mean_point in poincare_mean_list:
                    poincare_mean_point = poincare_mean_point.unsqueeze(dim=0)
                    d = (dist_matrix(output, poincare_mean_point, c=e2p.c) / 1)
                    distance.append(d[0].item())
                print(path, distance, np.argmin(distance))
                y_true.append(label.item())
                y_pred.append(np.argmin(distance))
            cm = confusion_matrix(y_true, y_pred)
            epoch_acc = accuracy_score(y_true, y_pred)

            print(cm, epoch_acc)
            
            if best_acc <= epoch_acc:
                best_acc = epoch_acc
                best_acc_model = copy.deepcopy(model.state_dict())
                torch.save(best_acc_model, './weights/best_model.pt')
                print('model saved. best accuracy:', best_acc)
            torch.save(best_acc_model, f'./weights/model_{epoch}.pt')

        with torch.no_grad():
            embedding = []
            targets = []
            querys = []
            paths = []
            for data in tqdm(UMAP_dataloader):
                vol, label, isquery, path = data
                vol = vol.to(device)

                output = model.forward_once(vol)
                output = output.detach().cpu()
                output = e2p(output)

                embedding.append(output[0].tolist())
                targets.append(label[0])
                querys.append(isquery.item())
                paths.append(path)
            for i, poincare_mean_point in enumerate(poincare_mean_list):
                embedding.append(poincare_mean_point.tolist())
                targets.append(i+5)
                querys.append(False)
                paths.append('poincare_mean_point')
            
            querys = np.array(querys)
            targets = np.array(targets)
            embedding = np.array(embedding)
            sns.set(style='white', rc={'figure.figsize':(10,10)})

            hyperbolic_mapper = umap.UMAP(output_metric='hyperboloid',
                                        random_state=42).fit(embedding)
            x = hyperbolic_mapper.embedding_[:, 0]
            y = hyperbolic_mapper.embedding_[:, 1]
            z = np.sqrt(1 + np.sum(hyperbolic_mapper.embedding_**2, axis=1))

            disk_x = x / (1 + z)
            disk_y = y / (1 + z)
            cdict = {0:'yellow', 1: 'red', 2: 'blue', 3: 'green', 4:'orange'}
            plt.clf()
            fig = plt.figure()
            ax = fig.add_subplot(111)
            for c, label in enumerate(UMAP_dataset.train_Subtype):
                idx = np.where(targets == label)
                ax.scatter(disk_x[idx], disk_y[idx], s=30, c=cdict[c], label=label, alpha=0.5)

            idx = np.where(querys == True)
            ax.scatter(disk_x[idx], disk_y[idx],s=100,linewidth=1,facecolors="none",edgecolors="k")

            ax.scatter(disk_x[-2], disk_y[-2], s=50, marker='*', linewidth=1, c=cdict[3], edgecolors="k")
            ax.scatter(disk_x[-1], disk_y[-1], s=50, marker='*', linewidth=1, c=cdict[4], edgecolors="k")

            boundary = plt.Circle((0,0), 1, fc='none', ec='k')
            ax.legend()
            ax.add_artist(boundary)
            plt.xlim([-1, 1])      # X축의 범위: [xmin, xmax]
            plt.ylim([-1, 1])

            plt.savefig(f'./temp/{epoch}_debug.png')
            for i in range(len(disk_x)):
                ax.text(disk_x[i], disk_y[i], i, fontsize='xx-small')
            with open('./num2patient_hyper.txt', 'w') as f:
                for i in range(len(disk_x)):
                    f.write(f'{i}, {paths[i][0][:-7]}\n')
            plt.savefig(f'./temp/{epoch}.png')

main()