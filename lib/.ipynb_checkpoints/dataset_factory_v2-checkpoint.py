import os
import glob
import cv2
import math
import numpy as np
from random import randint

import torch
import torchvision.transforms as transforms


# データセットクラスの作成
class RGBD_DATASET(torch.utils.data.Dataset):
    # RGB image       : ./dataset/train/color/{timestamp}.csv
    # Depth image     : ./dataset/train/depth/{timestamp}.png
    # Mass data       : ./dataset/train/grams/{timestamp}.txt
    # grasping detail : ./dataset/train/xyzu/{timestamp}.xxx
    def __init__(self, root, use_ids, train=True, img_size=160, crop_size=150):
        super().__init__()
        self.root = root
        self.train = train
        self.img_size = img_size
        self.crop_size = crop_size
        self.use_ids = use_ids                  #use_ids属性を追加
        self.to_tensor = transforms.ToTensor()

        #L515_SCALING = 0.00025
        #CAMERA_OFFSET_METERS = 0.6511

        # データのPATH
        image_path_list = glob.glob(os.path.join(root, "color_crop/*"))
        print("Number of data : ", len(image_path_list))
        #print(len(use_ids))

        # PATHをソートしてuse_idsで指定されたデータのPATHのみを取り出し
        image_path_list = sorted(image_path_list)
        image_path_list = [image_path_list[i] for i in use_ids]
        
        
        # 読み込んだデータを保存するリスト
        self.image_list, self.depth_list, self.grasp_z_list, self.targets_list = [],[],[],[]

        # PATHを元にデータを読み込み
        for image_path in image_path_list:
            # depthと把持量を読み込むための準備
            base_path = os.path.splitext(image_path)[0]
            #print(image_path)
            #print(base_path)

            # 画像データの読み込み
            image = cv2.imread(image_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # チャンネルの順番を変更 BGR -> RGB
            
            # depthデータの読み込み
            # *** pathの作成 ***
            depth_path = base_path.split('/')
            depth_path[3] = "csv_depth_crop"
            depth_path = os.path.join(*depth_path) + ".csv"
            # *** データの読み込み ***
            with open(depth_path) as file:
                depth = np.loadtxt(file, delimiter=',')
            
            # 画像データとdepthデータを把持点を中心にクロップ
            #image, depth = self._point_crop(image, depth, self.img_size, grasp_y, grasp_x)

            # depthデータの正規化とRGB画像化
            # *** min-max正規化 ***
            depth = (depth - depth.min()) / (depth.max() - depth.min())
            # *** 値の範囲の変更 (0~1 -> 0~255) ***
            depth = depth*255
            # *** 型の変更 ***
            depth = depth.astype('uint8')
            # *** BGR画像化 ***
            depth = cv2.applyColorMap(depth, cv2.COLORMAP_VIRIDIS)
            # *** チャンネルの順番を変更 (BGR -> RGB) ***
            depth = cv2.cvtColor(depth, cv2.COLOR_BGR2RGB)
            
            # 把持量データの読み込み
            # *** pathの作成 ***
            target_path = base_path.split('/')
            target_path[3] = "grams"
            target_path = os.path.join(*target_path) + ".txt"
            # *** データの読み込み ***
            with open(target_path) as f:
                target = f.read()
                
            xyz_path = base_path.split('/')
            xyz_path[3] = "xyz_robot"
            xyz_path = os.path.join(*xyz_path) + ".csv"
            grasp_info = np.genfromtxt(xyz_path, delimiter=',')
            grasp_z  = grasp_info[1, 2]
            
            # 各データとをリストに保存
            if float(target) > 0 and float(target) < 8:
                if len(self.targets_list) < 200:  # targetの数が150未満なら追加
                    self.image_list.append(image)
                    self.depth_list.append(depth)
                    #self.grasp_z_list.append(float(40))
                    self.grasp_z_list.append(float(grasp_z))
                    self.targets_list.append(float(target))
                
        print("target：",len(self.targets_list))
        print("color ：",len(self.image_list))
            
    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, item):
        # データの取り出し
        image_rgb = self.image_list[item]
        image_depth = self.depth_list[item]
        grasp_z = self.grasp_z_list[item]
        target = self.targets_list[item]
        
        # データ増幅の適用
        if self.train:
            # ランダムクロップの適用
            image_rgb, image_depth = self._random_crop(image_rgb, image_depth, self.img_size, self.crop_size)
        else:
            # センタークロップの適用
            center_x = math.floor(self.img_size/2)
            center_y = math.floor(self.img_size/2)
            image_rgb, image_depth = self._point_crop(image_rgb, image_depth, self.crop_size, center_y, center_x)

        # 各データをPyTorchのTensor形式に変換
        image_rgb   = self.to_tensor(image_rgb)
        image_depth = self.to_tensor(image_depth)
        grasp_z     = torch.Tensor(np.array(grasp_z))
        target      = torch.Tensor(np.array(target))
        
        # use_idsからitemに対応するIDを取得して返す   #####
        use_id = self.use_ids[item]
        
        return image_rgb, image_depth, grasp_z, target, use_id, item    #####

    @staticmethod
    def _point_crop(rgb, depth, after_size, grasp_y, grasp_x):
        # numpy形式の画像のクロップ
        if after_size%2 == 0:
            crop_size = int(after_size/2)
            rgb   = rgb[grasp_y-crop_size:grasp_y+crop_size, grasp_x-crop_size:grasp_x+crop_size, :]
            depth = depth[grasp_y-crop_size:grasp_y+crop_size, grasp_x-crop_size:grasp_x+crop_size]
        else:
            crop_size = math.floor(after_size/2)
            rgb   = rgb[grasp_y-crop_size:grasp_y+1+crop_size, grasp_x-crop_size:grasp_x+1+crop_size, :]
            depth = depth[grasp_y-crop_size:grasp_y+1+crop_size, grasp_x-crop_size:grasp_x+1+crop_size]
        return rgb, depth

    @staticmethod
    def _random_crop(image_rgb, image_depth, before_size, after_size):
        # クロップ位置の決定
        top  = randint(0, before_size-after_size-1)
        left = randint(0, before_size-after_size-1)
        # 画像のクロップ
        image_rgb   = image_rgb[left:left+after_size, top:top+after_size, :]
        image_depth = image_depth[left:left+after_size, top:top+after_size, :]
        return image_rgb, image_depth
