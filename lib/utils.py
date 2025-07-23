import os
import math
import cv2
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.distributions import Categorical


# ======================================== 学習用 ========================================
def adjust_learning_rate(optimizer, init_lr, epoch, max_epoch, warmup_epoch):
    if epoch <= warmup_epoch:
        # Warmupによる学習率の増幅
        cur_lr = init_lr * (epoch/warmup_epoch)
    else:
        # コサイン関数に基づいた学習率の減衰
        cur_lr = init_lr * 0.5 * (1. + math.cos(math.pi * (epoch-warmup_epoch) / (max_epoch-warmup_epoch)))

    # optimizerに格納されている学習率の値の変更
    for param_group in optimizer.param_groups:
        if 'fix_lr' in param_group and param_group['fix_lr']:
            param_group['lr'] = init_lr
        else:
            param_group['lr'] = cur_lr
            


def evaluate_candidate_point(net, input_size, image_rgb, raw_depth, x_split, y_split, x_low, y_low, x_stride, y_stride):
    ### x,y座標のRGBカメラ座標とロボット座標の統一 ####################################################
    # カメラ座標とロボット座標の対応点
    y = 3
    camera_points_2d = np.array([[329.55405, 145.20818],[1023.65424, 159.20818],[480.4894, 450.58234],[878.0951, 450.11392],[480.53217, 148.20818],[580.5995 ,247.37337],[629.83734 ,297.77988],[629.8873 ,247.38222],[481.31372,196.16563],[876.26416,197.00854],[876.6912,247.1483],[877.0696,297.7281]], dtype=np.float32)  # カメラ座標系の点
    camera_points_2d_2 = camera_points_2d.copy()
    camera_points_2d = np.array([camera_points_2d[0],camera_points_2d[1],camera_points_2d[2],camera_points_2d[3]])
    robot_points_2d = np.array([[119.0+y, -171.2],[111.5+y, 192.0],[270.4+y, -84.2],[263.3+y, 117.7]], dtype=np.float32)  # ロボット座標系の点

    # 2Dアフィン変換行列を推定
    affine_matrix_2d, inliers = cv2.estimateAffine2D(camera_points_2d, robot_points_2d)

    if affine_matrix_2d is not None:
        print("推定された2Dアフィン変換行列:\n", affine_matrix_2d)
    else:
        print("2Dアフィン変換の推定に失敗しました")
        exit(1)
    ### ロボット座標からカメラ座標への変換 ##############################
    # アフィン変換行列を逆変換（ロボット座標からカメラ座標への変換用）
    inverse_affine_matrix_2d = cv2.invertAffineTransform(affine_matrix_2d)
    print(f"inverse_affine_matrix_2d:\n {inverse_affine_matrix_2d}")

    """
    # ロボット座標からカメラ座標に変換する関数
    def robot_to_camera_coordinates(robot_point, inverse_affine_matrix_2d):
        # ロボット座標を1x1x2の形式に変換
        robot_point = np.array([robot_point], dtype=np.float32)
        # アフィン変換でカメラ座標に変換
        transformed_camera_point = cv2.transform(robot_point, inverse_affine_matrix_2d)
        return transformed_camera_point[0][0]

    ####################################################################################################
    """
    # その他の細かな条件の設定
    crop_size  = math.floor(input_size/2)
    #camera_robot_matrix = np.genfromtxt('./dataset/camera_robot_matrix_170_okamoto3.csv', delimiter=',')
    to_tensor_func = transforms.ToTensor()

    # 各候補位置の把持量と標準偏差をMDNにより推定
    image_list = []
    depth_list = []
    xy_robot_coordinate_list = []
    xy_rgb_coordinate_list = []
    sigma_lsit = []
    mu_list = []

    net.eval()
    for x_i in range(x_split+1):
        for y_i in range(y_split+1):
            # 把持位置の決定(ロボット座標系)
            grasp_y = y_low + y_i*y_stride
            grasp_x = x_low + x_i*x_stride
            xy_robot_coordinate_list.append([grasp_x, grasp_y])
            #print("x : ", grasp_x, " y : ", grasp_y)
            # 座標系の変換 : ロボット座標系 -> RGB座標系
            robot_point = np.array([grasp_x, grasp_y], dtype=np.float32)
            robot_point_array = np.array([[robot_point]], dtype=np.float32)
            grasp_x, grasp_y = cv2.transform(robot_point_array, inverse_affine_matrix_2d)[0][0]
            grasp_x, grasp_y = int(grasp_x), int(grasp_y)
            #print("grasp_x, grasp_y : ",(grasp_x, grasp_y))
            xy_rgb_coordinate_list.append([grasp_x, grasp_y])
            # 把持位置を中心としたセンタークロップ
            # *** depth ***
            croped_depth = raw_depth[grasp_y-crop_size:grasp_y+crop_size, grasp_x-crop_size:grasp_x+crop_size]
            croped_depth = (croped_depth - croped_depth.min()) / (croped_depth.max() - croped_depth.min())
            croped_depth = croped_depth*255
            croped_depth = croped_depth.astype('uint8')
            croped_depth = cv2.applyColorMap(croped_depth, cv2.COLORMAP_VIRIDIS)
            croped_depth = cv2.cvtColor(croped_depth, cv2.COLOR_BGR2RGB)
            depth_list.append(croped_depth)
            # *** RGB ***
            croped_rgb = image_rgb[grasp_y-crop_size:grasp_y+crop_size, grasp_x-crop_size:grasp_x+crop_size, :]
            croped_rgb = cv2.cvtColor(croped_rgb, cv2.COLOR_BGR2RGB)
            image_list.append(croped_rgb)
            # MDNによる予測
            z_gripper     = torch.Tensor(np.array(float(40))).unsqueeze(0).cuda()
            z_gripper     = z_gripper.reshape([z_gripper.size()[0],-1])
            croped_rgb    = to_tensor_func(croped_rgb).unsqueeze(0).cuda()
            croped_depth  = to_tensor_func(croped_depth).unsqueeze(0).cuda()
            with torch.no_grad():
                pi, sigma, mu = net(croped_rgb, croped_depth, 0*z_gripper)
            # 予測結果の取得
            if pi.shape[-1] == 1:  # ガウス分布の数が1つの場合
                variance = sigma[0].item()
                mean = mu[0].item()
            else:                  # ガウス分布の数が2つ以上の場合
                # 使用するガウス分布をpiに基づいて決定
                pis = Categorical(pi).sample().view(pi.size(0), 1, 1)
                # 決定したガウス分布に対応する平均値と標準偏差の値をgatherにより取り出し
                variance = sigma.gather(1, pis).detach().squeeze().item()
                mean = mu.detach().gather(1, pis).squeeze().item()
            sigma_lsit.append(variance*1000)
            mu_list.append(mean*1000)
    return xy_rgb_coordinate_list, sigma_lsit, mu_list


def evaluate_candidate_point_with_RND(net, predict_net, target_net, input_size, image_rgb, raw_depth, x_split, y_split, x_low, y_low, x_stride, y_stride):
    ### x,y座標のRGBカメラ座標とロボット座標の統一 ####################################################
    # カメラ座標とロボット座標の対応点
    y = 3
    camera_points_2d = np.array([[329.55405, 145.20818],[1023.65424, 159.20818],[480.4894, 450.58234],[878.0951, 450.11392],[480.53217, 148.20818],[580.5995 ,247.37337],[629.83734 ,297.77988],[629.8873 ,247.38222],[481.31372,196.16563],[876.26416,197.00854],[876.6912,247.1483],[877.0696,297.7281]], dtype=np.float32)  # カメラ座標系の点
    camera_points_2d_2 = camera_points_2d.copy()
    camera_points_2d = np.array([camera_points_2d[0],camera_points_2d[1],camera_points_2d[2],camera_points_2d[3]])
    robot_points_2d = np.array([[119.0+y, -171.2],[111.5+y, 192.0],[270.4+y, -84.2],[263.3+y, 117.7]], dtype=np.float32)  # ロボット座標系の点

    # 2Dアフィン変換行列を推定
    affine_matrix_2d, inliers = cv2.estimateAffine2D(camera_points_2d, robot_points_2d)

    if affine_matrix_2d is not None:
        print("推定された2Dアフィン変換行列:\n", affine_matrix_2d)
    else:
        print("2Dアフィン変換の推定に失敗しました")
        exit(1)
    ### ロボット座標からカメラ座標への変換 ##############################
    # アフィン変換行列を逆変換（ロボット座標からカメラ座標への変換用）
    inverse_affine_matrix_2d = cv2.invertAffineTransform(affine_matrix_2d)
    print(f"inverse_affine_matrix_2d:\n {inverse_affine_matrix_2d}")

    """
    # ロボット座標からカメラ座標に変換する関数
    def robot_to_camera_coordinates(robot_point, inverse_affine_matrix_2d):
        # ロボット座標を1x1x2の形式に変換
        robot_point = np.array([robot_point], dtype=np.float32)
        # アフィン変換でカメラ座標に変換
        transformed_camera_point = cv2.transform(robot_point, inverse_affine_matrix_2d)
        return transformed_camera_point[0][0]

    ####################################################################################################
    """
    # その他の細かな条件の設定
    crop_size  = math.floor(input_size/2)
    #camera_robot_matrix = np.genfromtxt('./dataset/camera_robot_matrix_170_okamoto3.csv', delimiter=',')
    to_tensor_func = transforms.ToTensor()
    criterion_rnd = nn.MSELoss(reduction='mean').cuda()

    # 各候補位置の把持量と標準偏差をMDNにより推定
    image_list = []
    depth_list = []
    xy_robot_coordinate_list = []
    xy_rgb_coordinate_list = []
    sigma_lsit = []
    mu_list = []
    rnd_score_list = []

    net.eval()
    for x_i in range(x_split+1):
        for y_i in range(y_split+1):
            # 把持位置の決定(ロボット座標系)
            grasp_y = y_low + y_i*y_stride
            grasp_x = x_low + x_i*x_stride
            xy_robot_coordinate_list.append([grasp_x, grasp_y])
            #print("x : ", grasp_x, " y : ", grasp_y)
            # 座標系の変換 : ロボット座標系 -> RGB座標系
            robot_point = np.array([grasp_x, grasp_y], dtype=np.float32)
            robot_point_array = np.array([[robot_point]], dtype=np.float32)
            grasp_x, grasp_y = cv2.transform(robot_point_array, inverse_affine_matrix_2d)[0][0]
            grasp_x, grasp_y = int(grasp_x), int(grasp_y)
            #print("grasp_x, grasp_y : ",(grasp_x, grasp_y))
            xy_rgb_coordinate_list.append([grasp_x, grasp_y])
            # 把持位置を中心としたセンタークロップ
            # *** depth ***
            croped_depth = raw_depth[grasp_y-crop_size:grasp_y+crop_size, grasp_x-crop_size:grasp_x+crop_size]
            croped_depth = (croped_depth - croped_depth.min()) / (croped_depth.max() - croped_depth.min())
            croped_depth = croped_depth*255
            croped_depth = croped_depth.astype('uint8')
            croped_depth = cv2.applyColorMap(croped_depth, cv2.COLORMAP_VIRIDIS)
            croped_depth = cv2.cvtColor(croped_depth, cv2.COLOR_BGR2RGB)
            depth_list.append(croped_depth)
            # *** RGB ***
            croped_rgb = image_rgb[grasp_y-crop_size:grasp_y+crop_size, grasp_x-crop_size:grasp_x+crop_size, :]
            croped_rgb = cv2.cvtColor(croped_rgb, cv2.COLOR_BGR2RGB)
            image_list.append(croped_rgb)
            # MDNによる予測
            z_gripper    = torch.Tensor(np.array(float(40))).unsqueeze(0).cuda()
            z_gripper    = z_gripper.reshape([z_gripper.size()[0],-1])
            croped_rgb   = to_tensor_func(croped_rgb).unsqueeze(0).cuda()
            croped_depth = to_tensor_func(croped_depth).unsqueeze(0).cuda()
            with torch.no_grad():
                pi, sigma, mu = net(croped_rgb, croped_depth, 0*z_gripper)
            # 予測結果の取得
            if pi.shape[-1] == 1:  # ガウス分布の数が1つの場合
                variance = sigma[0].item()
                mean = mu[0].item()
            else:                  # ガウス分布の数が2つ以上の場合
                # 使用するガウス分布をpiに基づいて決定
                pis = Categorical(pi).sample().view(pi.size(0), 1, 1)
                # 決定したガウス分布に対応する平均値と標準偏差の値をgatherにより取り出し
                variance = sigma.gather(1, pis).detach().squeeze().item()
                mean = mu.detach().gather(1, pis).squeeze().item()
            sigma_lsit.append(variance*1000)
            mu_list.append(mean*1000)
            # RNDスコアの算出
            output_rnd = predict_net(croped_rgb, croped_depth)
            target_rnd = target_net(croped_rgb, croped_depth)
            loss_rnd   = criterion_rnd(output_rnd, target_rnd)
            rnd_score_list.append(loss_rnd.item())
    return xy_rgb_coordinate_list, sigma_lsit, mu_list, rnd_score_list


def plot_heatmap(image_rgb, xy_rgb_coordinate_list, sigma_lsit, mu_list):
    # ゼロ埋めされたヒートマップの用意
    mass_map = np.zeros((1080,1920))
    sd_map   = np.zeros((1080,1920))

    # ヒートマップに着色
    crop_size = 13
    for i in range(len(xy_rgb_coordinate_list)):
        xy_coordinate = xy_rgb_coordinate_list[i]
        mass_map[xy_coordinate[1]-crop_size:xy_coordinate[1]+crop_size, xy_coordinate[0]-crop_size:xy_coordinate[0]+crop_size] = mu_list[i]
        sd_map[xy_coordinate[1]-crop_size:xy_coordinate[1]+crop_size, xy_coordinate[0]-crop_size:xy_coordinate[0]+crop_size]   = sigma_lsit[i]
        #mass_map[xy_coordinate[1], xy_coordinate[0]] = 1

    # 食品トレイのみの領域となるようにクロップ
    image_tray = image_rgb[165:680, 1000:1600, :]
    mass_tray  = mass_map[165:680, 1000:1600]
    sd_tray    = sd_map[165:680, 1000:1600]

    # 可視化
    fig = plt.figure(figsize=(19,3))

    ax1 = fig.add_subplot(1, 2, 1)
    ax1.imshow(cv2.cvtColor(image_tray, cv2.COLOR_BGR2RGB))
    ax1.set_title("RGB")

    ax2 = fig.add_subplot(1, 3, 2)
    aximg = ax2.imshow(mass_tray)
    fig.colorbar(aximg, ax=ax2)
    ax2.set_title("Estimated mass [g]")

    ax3 = fig.add_subplot(1, 3, 3)
    aximg = ax3.imshow(sd_tray)
    fig.colorbar(aximg, ax=ax3)
    ax3.set_title("Estimated standard deviation [g]")
    
    #plt.savefig("candidate_point.pdf")
    plt.show()


def show_candidate_point(net, image_path_list, input_size=157, image_i=0):
    # 指定したデータidのPATHの取得
    image_path = image_path_list[image_i]
    base_path = os.path.splitext(image_path)[0]

    # 画像の読み込み
    image_rgb = cv2.imread(image_path)

    # Depthの読み込み
    depth_path = base_path.split('/')
    depth_path[3] = "depth_aligned"
    depth_path = os.path.join(*depth_path) + ".csv"
    with open(depth_path) as file:
        raw_depth = np.loadtxt(file, delimiter=',')

    # 食品トレイのある範囲に関するパラメータ(ロボット座標系)
    x_low  = -300
    x_high = -180
    y_low  = -65
    y_high = 160

    # 把持位置の候補作成のパラメータ(ロボット座標系において何mmごとに候補位置を設置するのか)
    x_split = 5 # 3 5 8
    y_split = 9 # 5 9 15
    x_stride = (x_high-x_low)/x_split
    y_stride = (y_high-y_low)/y_split

    # 候補位置の評価
    xy_rgb_coordinate_list, sigma_lsit, mu_list = evaluate_candidate_point(net, input_size, image_rgb, raw_depth,
                                                                           x_split, y_split, x_low, y_low, x_stride, y_stride)

    # MDNの予測結果の可視化
    plot_heatmap(image_rgb, xy_rgb_coordinate_list, sigma_lsit, mu_list)
