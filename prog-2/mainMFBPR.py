import comet_ml
import pandas as pd
import numpy as np
import torch
# torch.manual_seed(0)
from torch import nn as nn
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm
from model import MFBPR
from utils import calc_ap
import argparse
import math

parser = argparse.ArgumentParser()

parser.add_argument("--lr", type=float, default=0.0001, help="learning rate")
parser.add_argument("--epochs", type=int, default=40, help="training epochs")
parser.add_argument("--batch_size", type=int, default=64, help="batch size")
parser.add_argument("--weight_decay", type=float, default=0, help="L2 regularization")
parser.add_argument("--do_train", action="store_true", help="do training")
parser.add_argument("--do_test", action="store_true", help="do testing")
parser.add_argument("--model", type=str, default=None, help="path to load model")

args = parser.parse_args()

class Recommender:
    def __init__(self,
        train_file='train.csv',
        output_file='output.csv',
        n_factor = 128,
        lr = 0.1,
        weight_decay = 0.01,
        batch_size = 64,
        epochs = 60
    ):
        if torch.cuda.is_available():
            self.device = 'cuda'
        else:
            self.device = 'cpu'
        self.train_file = train_file
        self.output_file = output_file
        self.n_factor = n_factor
        self.batch_size = batch_size
        self.epochs = epochs
        self.lr = lr
        self.weight_decay = weight_decay
        self.criterion = nn.BCEWithLogitsLoss(reduction = 'sum')
        self.load_data(train_file)
        self.model = MFBPR(self.user_num, self.item_num, self.n_factor).to(self.device)
        self.opt = torch.optim.SGD(self.model.parameters(), lr = lr, weight_decay=weight_decay)
        # self.scheduler = torch.optim.lr_scheduler.StepLR(self.opt, step_size=10, gamma=0.1)
        
    def load_data(self, train_file):
        df = pd.read_csv(train_file)
        self.pos_items_list = []
        self.neg_items_list = []
        max_item_index = 0

        for index, row in df.iterrows():
            items = np.array(row['ItemId'].split(), dtype=int)
            self.pos_items_list.append(items)
            max_item_index = max(np.max(items), max_item_index)
        for pos_items in self.pos_items_list:
            neg_items = np.delete(np.arange(max_item_index+1), pos_items)
            self.neg_items_list.append(neg_items)

        self.user_num = len(self.pos_items_list)
        self.item_num = max_item_index + 1

        self.mask = np.zeros((self.user_num, self.item_num), dtype=bool)
        for i, items in enumerate(self.pos_items_list):
            self.mask[i, items] = True
    
    def get_data_loader(self, batch_size, ratio=0.11):
        # Split train valid set
        self.pos_items_list_train = []
        self.pos_items_list_valid = []
        self.train_mask = np.zeros((self.user_num, self.item_num), dtype=bool)
        self.valid_mask = np.zeros((self.user_num, self.item_num), dtype=bool)
        for i, pos_items in enumerate(self.pos_items_list):
            length = pos_items.shape[0]
            p = np.arange(length)
            np.random.shuffle(p)
            self.pos_items_list_train.append(pos_items[p[int(length*ratio):]])
            self.pos_items_list_valid.append(pos_items[p[:int(length*ratio)]])

        for i, items in enumerate(self.pos_items_list_train):
            self.train_mask[i, items] = True
        for i, items in enumerate(self.pos_items_list_valid):
            self.valid_mask[i, items] = True

        # Build train dataloader
        train_x = np.empty((0,2), dtype=int)
        for i, pos_items in enumerate(self.pos_items_list_train):
            new_data = np.vstack((i*np.ones(pos_items.shape[0]), pos_items)).T
            train_x = np.vstack((train_x, new_data))

        self.num_iteration = math.ceil(train_x.shape[0] / batch_size)

        tensor_x = torch.LongTensor(train_x).to(self.device)
        tensor_y = torch.ones(train_x.shape[0], dtype=torch.float).to(self.device)

        dataset = TensorDataset(tensor_x, tensor_y)
        train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

        return train_loader

    def bpr_data_preprocess(self, x):
        # (batch_size, 2) -> (batch_size, 3)
        neg_items = torch.zeros((x.shape[0], 1), dtype=torch.long).to(self.device)
        for i, pair in enumerate(x):
            neg_items[i][0] = np.random.choice(self.neg_items_list[pair[0]])
            # neg_items[i][0] = self.neg_items_list[pair[0]][0]

        return torch.cat((x,neg_items), 1)

    def train(self):
        experiment = comet_ml.Experiment(
            api_key='hxGwRITemrFy7uvrecyt5tkIW',
            project_name="IR"
        )
        users_tensor = torch.tensor(np.arange(self.user_num), dtype=torch.long).to(self.device)
        items_tensor = torch.tensor(np.arange(self.item_num), dtype=torch.long).to(self.device)

        experiment.set_name("MFBPR")
        experiment.log_parameter("batch_size", self.batch_size)
        experiment.log_parameter("n_factor", self.n_factor)
        experiment.log_parameter("epochs", self.epochs)
        experiment.log_parameter("learning_rate", self.lr)
        experiment.log_parameter("weight_decay", self.weight_decay)




        max_valid_map = 0
        for epoch in tqdm(range(self.epochs)):
            train_loader = self.get_data_loader(self.batch_size)
            print("Epoch:", epoch+1)
            train_loss = 0
            self.model.train()
            # pbar = tqdm(total=self.num_iteration)
            for batch in train_loader:
                # pbar.update()
                # batch[0]: x
                # batch[1]: y
                inputs = self.bpr_data_preprocess(batch[0])
                targets = batch[1]
                logits = self.model(inputs)
                loss = self.criterion(logits, targets)
                self.opt.zero_grad()
                loss.backward()
                self.opt.step()
                train_loss += loss.item()
            train_loss /= self.num_iteration*self.batch_size
            self.model.eval()
            ratings = self.model.get_ratings(users_tensor, items_tensor)

            # self.scheduler.step()
            
            # Calculate train MAP
            train_map = 0
            ratings_array = ratings.cpu().detach().numpy()
            min_rating = ratings_array.min()
            ratings_array[self.valid_mask] = min_rating
            result = ratings_array.argsort(axis=1)[:,::-1]
            for u in np.arange(self.user_num):
                train_map += calc_ap(result[u][:50], self.pos_items_list_train[u])
            train_map /= self.user_num

            # Calculate valid MAP
            valid_map = 0
            ratings_array = ratings.cpu().detach().numpy()
            min_rating = ratings_array.min()
            ratings_array[self.train_mask] = min_rating
            result = ratings_array.argsort(axis=1)[:,::-1]
            for u in np.arange(self.user_num):
                valid_map += calc_ap(result[u][:50], self.pos_items_list_valid[u])
            valid_map /= self.user_num

            
            experiment.log_metric("train_loss", train_loss, epoch=epoch)
            experiment.log_metric("train_map", train_map, epoch=epoch)
            experiment.log_metric("valid_map", valid_map, epoch=epoch)

            print("\n")
            print("Train Loss:", train_loss)
            print("Train MAP:", train_map)
            print("Valid MAP:", valid_map)

            # If valid_map > max_valid_map, save the model
            if valid_map > max_valid_map:
                max_valid_map = valid_map
                torch.save(self.model, f'val_{valid_map}.model')

            


    def predict(self):
        users_tensor = torch.tensor(np.arange(self.user_num), dtype=torch.long).to(self.device)
        items_tensor = torch.tensor(np.arange(self.item_num), dtype=torch.long).to(self.device)
        ratings = self.model.get_ratings(users_tensor, items_tensor).cpu().detach().numpy()
        min_rating = ratings.min()
        ratings[self.mask] = min_rating
        result = ratings.argsort(axis=1)[:,::-1]

        with open(f"{self.output_file}", 'w') as f:
            f.write("UserId,ItemId\n")
            for i in range(self.user_num):
                f.write(f"{i},{' '.join(map(str, result[i][:50]))}\n")

        print(f"{self.output_file} saved!")






if __name__ == '__main__':
    recommender = Recommender()
    if args.do_train:
        recommender.train()
    elif args.do_test:
        if args.model:
            recommender.model = torch.load(args.model)
            recommender.predict()
        else:
            print("You have to specify --model {MODEL_PATH}!!")
    else:
        print("Please specify --do_train/--do_test to train/test the model!!")
