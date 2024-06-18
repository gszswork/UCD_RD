import torch 
import sys, argparse
from dataset import UCDDataset
from torch_geometric.data import DataLoader
from process import get_dataset
from model import *
import numpy as np
from sklearn import metrics
import wandb
import time


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
torch.manual_seed(0)
np.random.seed(0)
# Args
parser = argparse.ArgumentParser()
parser.add_argument('--lr', type=float, default=0.000001, help='learning rate')
parser.add_argument('--epochs', type=int, default=100, help='number of training epochs')
try:
    args = parser.parse_args()
except:
    parser.print_help()
    sys.exit(0)


def train_ucd_rd(model, dataname, args):
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    

    for epoch in range(args.epochs):

        model.train()
        in_dataset, out_dataset = get_dataset(dataname)
        in_data_loader = DataLoader(in_dataset, batch_size=64, shuffle=True)
        out_data_loader = DataLoader(out_dataset, batch_size=64, shuffle=True)
        avg_loss = []
        backprop_num = 0
        for out_batch in out_data_loader:
            for in_batch in in_data_loader:
                in_batch.to(device)
                out_batch.to(device)
                backprop_num += 1
                if in_batch.x.shape != out_batch.x.shape:
                    continue
                in_pred, loss, loss_things = model(in_batch, out_batch)
                avg_loss.append(loss.item())
                _, cur_pred = in_pred.max(dim=-1)
                correct = cur_pred.eq(in_batch.y).sum().item()
                in_train_acc = correct / len(in_batch.y)
                
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
            # If visualise the loss, need to calcualte the mean_loss 
            print('Epoch: ', epoch, 'Batch_num',backprop_num , 'Loss: ', np.mean(avg_loss), 'Train Acc.:', in_train_acc)

        model.eval()
        target_pred, target_true = np.array([]), np.array([])
        for out_batch in out_data_loader:
            out_batch.to(device)
            out_labels = model.predict(out_batch)
            _, pred  = out_labels.max(dim=-1)

            target_pred = np.append(target_pred, pred.detach().cpu().numpy())
            target_true = np.append(target_true, out_batch.y.cpu().numpy())

        print(metrics.classification_report(target_pred, target_true, digits=4))

if __name__ == '__main__':
    print('Training the model on device: ', device)
    parser = argparse.ArgumentParser()
    datasetname = 'Twitter'

    model = Net(input_dim=768, d_model=768, nhead=8, num_layers=3, gama=[1,0,0])  # input_dim param useless. 

    train_ucd_rd(model, datasetname, args)

