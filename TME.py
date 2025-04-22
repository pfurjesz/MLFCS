import pandas as pd
import numpy as np
import datetime

from torch.nn import Softmax

from utils import read_txn_data, preprocess_txn_data, create_lob_dataset, merge_txn_and_lob
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset, DataLoader
from torch import nn

def recalc(df:pd.DataFrame,train_size=0.7)->pd.DataFrame:
    df['time'] = df['datetime'].dt.time
    split = train_test_split(df, train_size=train_size, shuffle=False)
    a = split[0].shape[0]
    train = df.iloc[:a].copy()

    train['mean_volume'] = train.groupby('time')['total_volume'].transform('mean')
    train['deseasoned_total_volume'] = train['total_volume'] / train['mean_volume']
    train['log_deseasoned_total_volume'] = np.log(train['deseasoned_total_volume'])

    rest = df.iloc[a:].copy()

    trange = pd.date_range("00:00:05", "23:59:05", freq='1min').time
    for t in trange:
        if (t in rest['time'].values) and (t in train['time'].values):
            rest.loc[rest.index[rest['time'] == t], 'mean_volume'] = \
            train.loc[train.index[train['time'] == t], 'mean_volume'].iat[0]
        # elif (t in rest['time'].values) and not (t in train['time'].values):
        #     rest.loc[rest.index[rest['time'] == t], 'mean_volume'] = 0
    rest['deseasoned_total_volume'] = rest['total_volume'] / rest['mean_volume']
    rest['log_deseasoned_total_volume'] = np.log(rest['deseasoned_total_volume'])

    return pd.concat([train,rest])

class CustomDataset(Dataset):
    def __init__(self, dataframe: pd.DataFrame, h: int):
        self.trx = torch.tensor(dataframe[['buy_volume', 'sell_volume', 'buy_txn', 'sell_txn', 'volume_imbalance', 'txn_imbalance']].to_numpy())
        self.lob = torch.tensor(dataframe[['ask_volume', 'bid_volume', 'spread', 'lob_volume_imbalance', 'ask_slope_1', 'ask_slope_5', 'ask_slope_10', 'bid_slope_5', 'bid_slope_10', 'spread']].to_numpy())
        self.y = torch.tensor(dataframe['log_deseasoned_total_volume'].to_numpy())
        self.h = h

    def __len__(self):
        return len(self.y)-self.h

    def __getitem__(self, idx):
        trx = self.trx[idx:idx+self.h].unfold(0,self.h,1).squeeze()
        lob = self.lob[idx:idx+self.h].unfold(0,self.h,1).squeeze()
        label = self.y[idx+self.h]

        return trx, lob, label
        # return torch.cat((trx, lob),dim=0), label

class latent_dist(nn.Module):
    def __init__(self,h):
        super().__init__()
        self.h=h
        self.R_trx = nn.Linear(self.h, 1,bias=False)
        self.L_trx = nn.Linear(6, 1,bias=True)
        self.R_lob = nn.Linear(self.h, 1,bias=False)
        self.L_lob = nn.Linear(10, 1,bias=True)
        self.soft=nn.Softmax(dim=1)

    def forward(self, trx, lob):
        trx = self.R_trx(trx)
        trx = self.L_trx(torch.permute(trx,(0,2,1)))
        lob = self.R_lob(lob)
        lob = self.L_lob(torch.permute(lob,(0,2,1)))

        return self.soft(torch.cat((trx,lob),dim=1)).squeeze(dim=2)
        # return torch.cat((trx,lob),dim=1).squeeze(dim=2) #return logits

class log_norm(nn.Module):
    def __init__(self,h):
        super().__init__()
        self.h=h
        self.meanR_trx = nn.Linear(self.h, 1,bias=False)
        self.meanL_trx = nn.Linear(6, 1,bias=True)
        self.meanR_lob = nn.Linear(self.h, 1,bias=False)
        self.meanL_lob = nn.Linear(10, 1,bias=True)

        self.varR_trx = nn.Linear(self.h, 1,bias=False)
        self.varL_trx = nn.Linear(6, 1,bias=True)
        self.varR_lob = nn.Linear(self.h, 1,bias=False)
        self.varL_lob = nn.Linear(10, 1,bias=True)


    def forward(self, trx, lob):
        trx_mean = self.meanR_trx(trx)
        trx_mean = self.meanL_trx(torch.permute(trx_mean,(0,2,1)))
        lob_mean = self.meanR_lob(lob)
        lob_mean = self.meanL_lob(torch.permute(lob_mean,(0,2,1)))

        trx_var = self.varR_trx(trx)
        trx_var = self.varL_trx(torch.permute(trx_var,(0,2,1)))
        lob_var = self.varR_lob(lob)
        lob_var = self.varL_lob(torch.permute(lob_var,(0,2,1)))

        return torch.cat((trx_mean,lob_mean),dim=1).squeeze(dim=2), torch.exp(torch.cat((trx_var,lob_var),dim=1).squeeze(dim=2))

class TME(nn.Module):
    def __init__(self,h):
        super().__init__()
        self.log_norm=log_norm(h)
        self.latent_dist=latent_dist(h)

    def forward(self,trx,lob):
        mean, var = self.log_norm(trx,lob)
        prob = self.latent_dist(trx,lob)
        return mean,var,prob

def TME_loss(pred,target,eps=1e-6):
    mean,var,prob = pred[0],pred[1],pred[2]
    target = target.unsqueeze(dim=1)
    eps = torch.tensor(eps)
    p1 = torch.exp(-torch.square(target - mean)/(2*var))/(torch.exp(target)*torch.sqrt(var)) #dropped constants
    return -torch.log(torch.maximum(torch.diag(torch.matmul(p1,torch.permute(prob,(1,0)))),eps)).sum()

def train_loop(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    # Set the model to training mode - important for batch normalization and dropout layers
    # Unnecessary in this situation but added for best practices
    model.train()
    for batch, (trx,lob, y) in enumerate(dataloader):
        # Compute prediction and loss
        pred = model(trx,lob)
        loss = loss_fn(pred, y)
        # print(f"batch number {batch},loss: {loss.item()}")

        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * batch_size + len(trx)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

def test_loop(dataloader, model, loss_fn):
    # Set the model to evaluation mode - important for batch normalization and dropout layers
    # Unnecessary in this situation but added for best practices
    model.eval()
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss = 0

    # Evaluating the model with torch.no_grad() ensures that no gradients are computed during test mode
    # also serves to reduce unnecessary gradient computations and memory usage for tensors with requires_grad=True
    with torch.no_grad():
        for trx,lob, y in dataloader:
            pred = model(trx,lob)
            test_loss += loss_fn(pred, y).item()

    print(f"Test Error: \n Test Loss: {test_loss:>8f} \n")

if __name__ == "__main__":
    trx_df = read_txn_data(use_load=False)
    lob_df = create_lob_dataset(use_load=False)

    trx_df = preprocess_txn_data(trx_df, freq='1min', fill_missing_ts=False)

    df_merged = merge_txn_and_lob(trx_df, lob_df)
    #split data in train,val, test
    _, test_df = train_test_split(df_merged, train_size=0.8, shuffle=False)
    train_df, val_df = train_test_split(_, train_size=7 / 8, shuffle=False)

    #hyperparams
    h = 3 # h is window size
    learning_rate = 1e-3
    batch_size = 64
    epochs = 20
    Lambda = 1 #L2 regularisation coefficient

    #standardize features
    train_data = CustomDataset((train_df.iloc[:, 1:] - train_df.iloc[:, 1:].mean()) / train_df.iloc[:, 1:].std(), h)
    val_data = CustomDataset((val_df.iloc[:, 1:] - train_df.iloc[:, 1:].mean()) / train_df.iloc[:, 1:].std(), h)
    test_data = CustomDataset((test_df.iloc[:, 1:] - train_df.iloc[:, 1:].mean()) / train_df.iloc[:, 1:].std(), h)

    train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=False)
    val_dataloader = DataLoader(val_data, batch_size=batch_size, shuffle=False)
    test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

    model = TME(h).double()

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate,weight_decay=Lambda)

    for t in range(epochs):
        print(f"Epoch {t + 1}\n-------------------------------")
        train_loop(train_dataloader, model, TME_loss, optimizer)
        test_loop(test_dataloader, model, TME_loss)
    print("Done!")