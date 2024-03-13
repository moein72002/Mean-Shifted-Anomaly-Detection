import torch
from sklearn.metrics import roc_auc_score
import torch.optim as optim
import argparse
import utils
from tqdm import tqdm
import torch.nn.functional as F
import os
import random
from torchvision import transforms
import matplotlib.pyplot as plt


def contrastive_loss(out_1, out_2):
    out_1 = F.normalize(out_1, dim=-1)
    out_2 = F.normalize(out_2, dim=-1)
    bs = out_1.size(0)
    temp = 0.25
    # [2*B, D]
    out = torch.cat([out_1, out_2], dim=0)
    # [2*B, 2*B]
    sim_matrix = torch.exp(torch.mm(out, out.t().contiguous()) / temp)
    mask = (torch.ones_like(sim_matrix) - torch.eye(2 * bs, device=sim_matrix.device)).bool()
    # [2B, 2B-1]
    sim_matrix = sim_matrix.masked_select(mask).view(2 * bs, -1)

    # compute loss
    pos_sim = torch.exp(torch.sum(out_1 * out_2, dim=-1) / temp)
    # [2*B]
    pos_sim = torch.cat([pos_sim, pos_sim], dim=0)
    loss = (- torch.log(pos_sim / sim_matrix.sum(dim=-1))).mean()
    return loss


def train_model(model, train_loader, test_loader, train_loader_1, device, args):
    model.eval()
    auc, feature_space = get_score(model, device, train_loader, test_loader)
    print('Epoch: {}, AUROC is: {}'.format(0, auc))
    optimizer = optim.SGD(model.parameters(), lr=args.lr, weight_decay=0.00005)
    center = torch.FloatTensor(feature_space).mean(dim=0)
    if args.angular:
        center = F.normalize(center, dim=-1)
    center = center.to(device)
    for epoch in range(args.epochs):
        running_loss = run_epoch(model, train_loader_1, optimizer, center, device, args.angular)
        print('Epoch: {}, Loss: {}'.format(epoch + 1, running_loss))
        auc, _ = get_score(model, device, train_loader, test_loader)
        print('Epoch: {}, AUROC is: {}'.format(epoch + 1, auc))

    return auc


def run_epoch(model, train_loader, optimizer, center, device, is_angular):
    total_loss, total_num = 0.0, 0
    for ((img1, img2), _) in tqdm(train_loader, desc='Train...'):

        img1, img2 = img1.to(device), img2.to(device)

        optimizer.zero_grad()

        out_1 = model(img1)
        out_2 = model(img2)
        out_1 = out_1 - center
        out_2 = out_2 - center

        loss = contrastive_loss(out_1, out_2)

        if is_angular:
            loss += ((out_1 ** 2).sum(dim=1).mean() + (out_2 ** 2).sum(dim=1).mean())

        loss.backward()

        optimizer.step()

        total_num += img1.size(0)
        total_loss += loss.item() * img1.size(0)

    return total_loss / (total_num)


def get_score(model, device, train_loader, test_loader):
    train_feature_space = []
    with torch.no_grad():
        for (imgs, _) in tqdm(train_loader, desc='Train set feature extracting'):
            imgs = imgs.to(device)
            features = model(imgs)
            train_feature_space.append(features)
        train_feature_space = torch.cat(train_feature_space, dim=0).contiguous().cpu().numpy()
    test_feature_space = []
    test_labels = []
    with torch.no_grad():
        for (imgs, labels) in tqdm(test_loader, desc='Test set feature extracting'):
            imgs = imgs.to(device)
            features = model(imgs)
            test_feature_space.append(features)
            test_labels.append(labels)
        test_feature_space = torch.cat(test_feature_space, dim=0).contiguous().cpu().numpy()
        test_labels = torch.cat(test_labels, dim=0).cpu().numpy()

    distances = utils.knn_score(train_feature_space, test_feature_space)

    auc = roc_auc_score(test_labels, distances)

    return auc, train_feature_space

def main(args):
    print('Dataset: {}, Normal Label: {}, LR: {}'.format(args.dataset, args.label, args.lr))
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)
    model = utils.Model(args.backbone)
    model = model.to(device)

    if args.dataset == 'mvtec':
        # List all files and directories in the directory
        all_entries = sorted(os.listdir(directory_path))
        # Filter out the directories
        all_categories = [entry for entry in all_entries if os.path.isdir(os.path.join("/kaggle/input/mvtec-ad/", entry))]
        # all_categories = sorted(os.listdir("/kaggle/input/mvtec-ad/"))
        auc_sum = 0.0
        for category in all_categories:
            print(f"category: {category}")
            train_loader, test_loader, train_loader_1 = utils.get_mvtec_loaders(category=category, shrink_factor=args.shrink_factor, batch_size=args.batch_size, backbone=args.backbone)
            auc = train_model(model, train_loader, test_loader, train_loader_1, device, args)
            print(f"auc_{category}: {auc}")
            auc_sum += auc
        auc_avg = auc_sum / len(all_categories)
        print(f"auc_avg: {auc_avg}")
    else:
        train_loader, test_loader, train_loader_1 = utils.get_loaders(dataset=args.dataset, label_class=args.label, batch_size=args.batch_size, backbone=args.backbone)
        train_model(model, train_loader, test_loader, train_loader_1, device, args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--dataset', default='mvtec')
    parser.add_argument('--epochs', default=20, type=int, metavar='epochs', help='number of epochs')
    parser.add_argument('--label', default=0, type=int, help='The normal class')
    parser.add_argument('--lr', type=float, default=1e-5, help='The initial learning rate.')
    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--backbone', default=152, type=int, help='ResNet 18/152')
    parser.add_argument('--angular', action='store_true', help='Train with angular center loss')
    parser.add_argument('--shrink_factor', type=float, default=1.0)
    args = parser.parse_args()
    main(args)
