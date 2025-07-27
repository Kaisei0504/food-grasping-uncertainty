import os
import glob
import time
import pickle
import cv2
import math
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from random import randint
from time import time
from tqdm import tqdm

import collections
import random
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.autograd import Variable
from torch.distributions import Categorical
from torch.utils.data import DataLoader, Dataset, Subset
import torch.nn.functional as F

from lib import dataset_factory
from lib import models as model_fuctory
from lib import loss_func
from lib import utils
from sklearn.model_selection import KFold

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# サンプリング確率に基づいてミニバッチを作成するデータローダー
class CustomSampler(torch.utils.data.Sampler):
    def __init__(self, data_source, sampling_probabilities):
        self.data_source = data_source
        if len(sampling_probabilities) != len(data_source):
            raise ValueError("The size of sampling_probabilities must be equal to the size of data_source.")
        self.sampling_probabilities = sampling_probabilities

    def __iter__(self):
        return iter(np.random.choice(len(self.data_source), size=len(self.data_source), p=self.sampling_probabilities))

    def __len__(self):
        return len(self.data_source)

    def update_sampling_probabilities(self, new_probabilities):
        if len(new_probabilities) != len(self.data_source):
            raise ValueError("The size of new_probabilities must be equal to the size of data_source.")
        self.sampling_probabilities = new_probabilities

def main():
    # ********** 学習条件の設定 **********
    save_path = "./saved_models/2000epoch_lr0x001_NAdam_100samples_with_RND"
    num_gauss = 1
    init_lr = 0.001
    max_epoch = 2000
    warmup_epoch = 10

    os.makedirs(save_path, exist_ok=True)

    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # ********** 学習の準備 **********
    with open('./datasets/train_data/divide_ids/data_train_100.pickle', mode='br') as fi:
        id_train = pickle.load(fi)
    id_train = id_train[0].tolist()
    print(id_train)
    print("")
    
    # データセット
    train_data = dataset_factory.RGBD_DATASET(root="./datasets/train_data", use_ids = id_train, train=True, img_size=150, crop_size=140)
        
    num_data = len(train_data)
    #print(len(train_data))
    #print("# data : ", num_data)
    
    train_loader = torch.utils.data.DataLoader(train_data,
                                               batch_size=8,
                                               shuffle=True,
                                               num_workers=8,
                                               pin_memory=True,
                                               drop_last=True)

    # 初期サンプリング確率の設定
    initial_sampling_probabilities = np.random.dirichlet(np.ones(len(train_data)))

    net = model_fuctory.CNN_MDN(num_gaussians=num_gauss)
    predict_net = model_fuctory.CNN_RND()
    target_net = model_fuctory.CNN_RND()

    net = net.cuda()
    predict_net = predict_net.cuda()
    target_net = target_net.cuda()

    optimizer = optim.NAdam(net.parameters(), lr=init_lr)
    optimizer_rnd = optim.NAdam(predict_net.parameters(), lr=init_lr)

    criterion = loss_func.MDN_loss().cuda()
    criterion_rnd = nn.MSELoss(reduction='mean').cuda()

    net.train()
    predict_net.train()
    target_net.eval()

    epoch_list = []
    loss_list = []
    loss_rnd_list = []
    time_list = []
    
    new_sampling_probabilities = [0]*num_data
    
    start = time()
    
    for epoch in range(1, max_epoch+1):
        sum_loss = 0.0
        sum_loss_rnd = 0.0
        item_id = []    #itemをリストに保存    
        new_list = [0] * (num_data)
        use_ids_list = []
        losses_rnd_list = [] 
        losses_rnd_list_up = [0] * num_data

        utils.adjust_learning_rate(optimizer, init_lr, epoch, max_epoch, warmup_epoch)
        utils.adjust_learning_rate(optimizer_rnd, init_lr, epoch, max_epoch, warmup_epoch)

        if epoch <= 2000:
            sampler = CustomSampler(train_data, initial_sampling_probabilities)
            train_loader = DataLoader(train_data, batch_size=8, shuffle=True, num_workers=8, pin_memory=True, drop_last=True)
        else:
            sampler = CustomSampler(train_data, new_sampling_probabilities)
            train_loader = DataLoader(train_data, batch_size=8, sampler=sampler, num_workers=8, pin_memory=True, drop_last=True)  
            
        for image_rgb, image_depth, z_gripper, target, use_id, item in train_loader:   
            image_rgb = image_rgb.cuda()
            image_depth = image_depth.cuda()
            z_gripper = z_gripper.cuda()
            target = target.cuda()
            
            #itemをリストに保存
            item_id.extend(item.cpu().numpy().flatten().tolist())  #####
            
            #use_idsをリストに保存
            use_ids_list.extend(use_id.cpu().numpy().flatten().tolist())  #####
            
            z_gripper = z_gripper.reshape([z_gripper.size()[0], -1])
            target = target.reshape([target.size()[0], -1]) * 0.001
            
            pi, sigma, mu = net(image_rgb, image_depth, 0*z_gripper)
            output_rnd = predict_net(image_rgb, image_depth)
            with torch.no_grad():
                target_rnd = target_net(image_rgb, image_depth)

            loss = criterion(pi, sigma, mu, target)
            
            loss_rnd = criterion_rnd(output_rnd, target_rnd.detach())
            losses_rnd = torch.mean((output_rnd - target_rnd.detach()) ** 2, dim=1)
            mean_loss = torch.mean(losses_rnd)     
            
            #サンプルごとのRNDのスコアをリストに保存
            losses_rnd_list.extend(losses_rnd.detach().cpu().numpy().flatten().tolist())
            
            # item_idの値をインデックスとして、新しいリストにuse_ids_listの値を保存する
            for i, item in enumerate(item_id):
                new_list[item] = use_ids_list[i]
                losses_rnd_list_up[item] = losses_rnd_list[i]
                
            net.zero_grad()
            loss.backward()
            optimizer.step()
            predict_net.zero_grad()
            loss_rnd.backward()
            optimizer_rnd.step()

            sum_loss += loss.item()
            sum_loss_rnd += loss_rnd.item()

        # サンプリング確率の計算
        #print("「rnd値の逆数」")
        inv_scores = [1.0 / score if score != 0 else score for score in losses_rnd_list_up]
        
        #全体で割って正規化
        sum_inv_scores = sum(inv_scores)
        new_sampling_probabilities = [score / sum_inv_scores for score in inv_scores]
        
        # 新しいサンプリング確率を更新
        sampler.update_sampling_probabilities(new_sampling_probabilities)
        
        
        # 0以外の値をフィルタリングしたリストを作成
        non_zero_losses = [x for x in new_sampling_probabilities if x != 0]
               
        epoch_list.append(epoch)
        loss_list.append(sum_loss / len(train_loader))
        loss_rnd_list.append(sum_loss_rnd / len(train_loader))
        time_list.append(time() - start)

        if (epoch % 5 == 0) or (epoch == max_epoch):
            print(f"epoch: {epoch},\
                    mean MDN loss: {round(sum_loss / len(train_loader), 3)},\
                    mean RND loss: {round(sum_loss_rnd / len(train_loader), 10)},\
                    elapsed_time :{round(time() - start, 2)}")

    with open(os.path.join(save_path, 'end_log.txt'), mode="a") as f:
        f.write(f"epoch: {epoch},\
                  mean MDN loss: {round(sum_loss / len(train_loader), 3)},\
                  mean RND loss: {round(sum_loss_rnd / len(train_loader), 10)},\
                  elapsed_time :{round(time() - start, 2)}")

    torch.save(net.cpu().state_dict(), os.path.join(save_path, 'checkpoint.pth'))
    torch.save(predict_net.cpu().state_dict(), os.path.join(save_path, 'predict_net.pth'))
    torch.save(target_net.cpu().state_dict(), os.path.join(save_path, 'target_net.pth'))
        
    print("End of training")

if __name__ == "__main__":
    main()
