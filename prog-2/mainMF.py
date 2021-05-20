import pandas as pd
import numpy as np
import torch
from torch import nn as nn
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm
from model import MF

class Recommender:
    def __init__(self, train_file='train.csv', output_file='output.csv', hidden_size = 32):
        self.train_file = train_file
        self.output_file = output_file
        self.hidden_size = hidden_size
        self.criterion = nn.BCEWithLogitsLoss(reduction='sum')
        self.load_data(train_file)
        self.model = MF(self.user_num, self.item_num, self.hidden_size)
        if torch.cuda.is_available():
            self.model.cuda()
        self.opt = torch.optim.SGD(self.model.parameters(), lr = 0.01)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.opt, step_size=10, gamma=0.35)
        

    def load_data(self, train_file):
        df = pd.read_csv(train_file)
        user_item_list = []
        self.max_item_index = 0
        self.positive_data = np.empty((0,2), int)

        for index, row in df.iterrows():
            items = np.array(row['ItemId'].split(), dtype=int)
            new_data = np.vstack((index*np.ones(items.shape[0]), items))
            self.positive_data = np.vstack((self.positive_data, new_data.T))
            user_item_list.append(items)
            self.max_item_index = max(np.max(items), self.max_item_index)
        user_item_matrix = np.zeros((len(df), self.max_item_index+1), dtype=int)
        for index, items in enumerate(user_item_list):
            user_item_matrix[index, items] = 1

        self.user_item_matrix = user_item_matrix
        self.user_num = user_item_matrix.shape[0]
        self.item_num = user_item_matrix.shape[1]
        self.positive_num = user_item_matrix.sum()
        self.negative_num = self.user_num*self.item_num - self.positive_num
        print("Positive samples:", self.positive_num)
        print("Negative samples:", self.negative_num)


    def get_train_data(self, ratio=0.5):
        user_ids = np.random.choice(self.user_num, self.positive_num)
        item_ids = np.random.choice(self.item_num, self.positive_num)
        negative_data = np.vstack((user_ids, item_ids)).T
        
        indexes = np.where(self.user_item_matrix[user_ids, item_ids] == 1)[0]
        negative_data = np.delete(negative_data, indexes, axis = 0)

        train_x = np.vstack((self.positive_data, negative_data))
        train_y = np.hstack((np.ones(self.positive_num), np.zeros(negative_data.shape[0])))

        return train_x, train_y
    
    def get_data_loader(self, batch_size, val_ratio=0.2):
        train_x, train_y = self.get_train_data()
        length = train_x.shape[0]
        tensor_x = torch.LongTensor(train_x)
        tensor_y = torch.tensor(train_y)
        if torch.cuda.is_available():
            tensor_x = tensor_x.cuda()
            tensor_y = tensor_y.cuda()

        dataset = TensorDataset(tensor_x, tensor_y)
        train_size = int(length*(1-val_ratio))
        val_size = length - train_size
        train_data, val_data = torch.utils.data.random_split(dataset, [train_size, val_size])
        train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)

        return train_loader, val_loader

    def train(self, batch_size=2048, epochs=40):
        for epoch in tqdm(range(epochs)):
            train_loader, val_loader = self.get_data_loader(batch_size)
            print("Epoch:", epoch+1)
            train_loss = []
            self.model.train()
            for batch in train_loader:
                # batch[0]: x
                # batch[1]: y
                inputs = batch[0]
                targets = batch[1]
                logits = self.model(inputs)
                loss = self.criterion(logits, targets)
                self.opt.zero_grad()
                loss.backward()
                self.opt.step()
                train_loss.append(loss.item())
            print("Train Loss:", np.mean(train_loss))

            self.model.eval()
            val_loss = []
            val_acc = []
            for batch in val_loader:
                inputs = batch[0]
                targets = batch[1]
                logits = self.model(inputs)
                loss = self.criterion(logits, targets)
                preds = logits > 0
                acc = (preds == targets).float().mean()
                val_loss.append(loss.item())
                val_acc.append(acc.item())
            print("Valid Loss:", np.mean(val_loss))
            print("Valid Acc:", np.mean(val_acc))
            self.scheduler.step()

    def predict(self):
        user_emb = self.model.user_emb.weight.cpu().detach().numpy()
        item_emb = self.model.item_emb.weight.cpu().detach().numpy().transpose()
        ratings = np.matmul(user_emb, item_emb)
        min_rating = ratings.min()
        indexes = self.user_item_matrix.astype(bool)
        ratings[indexes] = min_rating
        result = ratings.argsort(axis=1)[:,::-1]
        print(result.shape)

        with open(f"{self.output_file}", 'w') as f:
            f.write("UserId,ItemId\n")
            for i in range(self.user_num):
                f.write(f"{i},{' '.join(map(str, result[i][:50]))}\n")

        print(f"{self.output_file} saved!")






if __name__ == '__main__':
    recommender = Recommender()
    recommender.train()
    recommender.predict()
