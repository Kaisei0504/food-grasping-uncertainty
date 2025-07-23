import pyrealsense2 as rs
import numpy as np
import time
import cv2
import matplotlib.pyplot as plt
from pymycobot.mypalletizer import MyPalletizer
import os
import csv
import math
import torch
from torchvision import transforms
import models as model_factory
import torch.nn as nn
from torch.distributions import Categorical

# RealSenseの初期化
def init_realsense():
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)
    pipeline.start(config)
    return pipeline

# Mypalletizerの初期化
def init_mypalletizer(port='/dev/ttyACM4', baudrate=115200):
    try:
        mypalletizer = MyPalletizer(port, baudrate)
        print(f"MyPalletizer initialized on {port}")
        return mypalletizer
    except Exception as e:
        print(f"Error initializing MyPalletizer: {e}")
        return None

mypalletizer = init_mypalletizer('/dev/ttyACM4')

### x,y座標のRGBカメラ座標とロボット座標の統一 ####################################################
# カメラ座標とロボット座標の対応点
camera_points_2d = np.array([[329.55405, 145.20818],[1023.65424, 159.20818],[480.4894, 450.58234],[878.0951, 450.11392],[480.53217, 148.20818],[580.5995 ,247.37337],[629.83734 ,297.77988],[629.8873 ,247.38222],[481.31372,196.16563],[876.26416,197.00854],[876.6912,247.1483],[877.0696,297.7281]], dtype=np.float32)  # カメラ座標系の点
camera_points_2d_2 = camera_points_2d.copy()
camera_points_2d = np.array([camera_points_2d[0],camera_points_2d[1],camera_points_2d[2],camera_points_2d[3]])
robot_points_2d = np.array([[119.0, -171.2],[111.5, 192.0],[270.4, -84.2],[263.3, 117.7]], dtype=np.float32)  # ロボット座標系の点

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

# ロボット座標からカメラ座標に変換する関数
def robot_to_camera_coordinates(robot_point, inverse_affine_matrix_2d):
    # ロボット座標を1x1x2の形式に変換
    robot_point = np.array([robot_point], dtype=np.float32)
    # アフィン変換でカメラ座標に変換
    transformed_camera_point = cv2.transform(robot_point, inverse_affine_matrix_2d)
    return transformed_camera_point[0][0]

####################################################################################################


def predict_picking(depth_frame, depth_image, rgb_image, inverse_affine_matrix_2d):
    """ Do something here that predicts the pick point.
    Returns a point of type epson_scara_point (see class definition above)
    """
    
    start_time = time.time()
    
    # 1. Set candidate points and obtain RGB and depth of candidate points
    # Limit points that do not hit the tray
    # 把持可能な領域（画像上のxyの上限下限）を設定

    x_low  = 156
    x_high = 240
    y_low  = -120
    y_high = 120

    # Set the spacing of candidate points
    #  * For example: when set x_split=6 and y_split=10, there are 60 candidate points
    # 把持位置の候補をトレイ内に等間隔に設定
    x_split  = 5   # x方向の候補の数
    y_split  = 15  # y方向の候補の数
    x_stride = (x_high-x_low)/(x_split-1)  # ストライドするピクセル数
    y_stride = (y_high-y_low)/(y_split-1)  # ストライドするピクセル数
    num_candidate = x_split*y_split
    
    # Create input data (RGB image, Depth image) of ML model
    count_ci  = 0
    data_ids = 0
    img_size  = 140  # Input size of ML model : img_size x img_size
    crop_size = math.floor(img_size/2)
    xy_robot_coordinate_list = []
    coordinate_z_list = []
    coordinate_rgb   = torch.zeros([num_candidate, 3, img_size, img_size])
    coordinate_depth = torch.zeros([num_candidate, 3, img_size, img_size])
    to_tensor_func   = transforms.ToTensor()

    for x_i in range(x_split):
        for y_i in range(y_split):
            # Create candidate point in robot coordinate
            grasp_x = x_low + x_i*x_stride
            grasp_y = y_low + y_i*y_stride
            #print("grasp_x, grasp_y : ",(grasp_x, grasp_y))
            xy_robot_coordinate_list.append([grasp_x, grasp_y])
            #print("xy_robot_coordinate_list: \\", xy_robot_coordinate_list)
            robot_point = np.array([grasp_x, grasp_y], dtype=np.float32)

            # ロボット座標を2次元形式に整形 (shape: (1, 1, 2))
            robot_point_array = np.array([[robot_point]], dtype=np.float32)

            # Convert robot coordinate to RGBD coordinate
            grasp_x, grasp_y = cv2.transform(robot_point_array, inverse_affine_matrix_2d)[0][0]
            grasp_x, grasp_y = int(grasp_x), int(grasp_y)
            #print("grasp_x, grasp_y : ",(grasp_x, grasp_y))

            # カメラ座標(x,y)の位置のz座標を取得
            x, y = grasp_x, grasp_y
            z_meters = depth_frame.get_distance(x, y)
            z_millimeters = z_meters * 1000  # ミリメートル単位に変換
            sink_depth = 35 * 1.09
             # 把持位置 = (ロボット原点とセンサ原点の差 - デプスの値 + オフセット値) - 把持深さ
            corrected_z = (532 - z_millimeters + 30) - sink_depth
            if corrected_z <= 70:
                print("changed z to 350")
                corrected_z = 350
            coordinate_z_list.append(corrected_z)

            # Create RGB image and Depth information centered on candidate point
            croped_rgb   = rgb_image[grasp_y-crop_size:grasp_y+crop_size, grasp_x-crop_size:grasp_x+crop_size, :]
            croped_depth = depth_image[grasp_y-crop_size:grasp_y+crop_size, grasp_x-crop_size:grasp_x+crop_size]
            #print("croped_rgb : ", croped_rgb)

            # Convert Depth information to Depth image (Depth RGB image)
            croped_depth = (croped_depth - croped_depth.min()) / (croped_depth.max() - croped_depth.min())  # min-max normalization
            croped_depth = (croped_depth*255).astype('uint8')  # Change to 8-bit of the range 0 to 255
            croped_depth = cv2.applyColorMap(croped_depth, cv2.COLORMAP_VIRIDIS)

            # Change channel from BGR to RGB
            croped_rgb   = cv2.cvtColor(croped_rgb, cv2.COLOR_BGR2RGB)
            croped_depth = cv2.cvtColor(croped_depth, cv2.COLOR_BGR2RGB)

            # Covert numpy.array to torch.tensor
            croped_rgb   = to_tensor_func(croped_rgb)
            croped_depth = to_tensor_func(croped_depth)

            # Store RGB and Depth image
            coordinate_rgb[count_ci]   = croped_rgb
            coordinate_depth[count_ci] = croped_depth
            count_ci += 1

    # 2. Predict grasp mass at each candidate point using machine learning (ML) model
    # Load ML model
    torch.set_printoptions(precision=10, sci_mode=True)
    mdn_net     = model_factory.CNN_MDN(num_gaussians=1)
    target_net  = model_factory.CNN_RND()
    predict_net = model_factory.CNN_RND()
    msg1 = mdn_net.load_state_dict(torch.load('./model/depth40/sampler_v1/checkpoint.pth', map_location='cpu',weights_only=True))
    msg2 = predict_net.load_state_dict(torch.load('./model/depth40/sampler_v1/predict_net.pth', map_location='cpu',weights_only=True))
    msg3 = target_net.load_state_dict(torch.load('./model/depth40/sampler_v1/target_net.pth', map_location='cpu',weights_only=True))
    print("Load MDN         weight : ", msg1)
    print("Load predict net weight : ", msg2)
    print("Load target  net weight : ", msg3)

    # Set picking point z (depth)
    z_gripper = torch.Tensor(np.array(coordinate_z_list)).unsqueeze(0)
    z_gripper = z_gripper.reshape([z_gripper.size()[1], z_gripper.size()[0]])

    # Predict by ML model
    mdn_net.eval()
    target_net.eval()
    predict_net.eval()
    with torch.no_grad():
        pi, sigma, mu = mdn_net(coordinate_rgb, coordinate_depth, 0*z_gripper)
        target_rnd  = target_net(coordinate_rgb, coordinate_depth)
        predict_rnd = predict_net(coordinate_rgb, coordinate_depth)

    criterion_rnd  = nn.MSELoss(reduction='mean')
    estimated_mass = []
    estimated_sd   = []
    rnd_score      = []

    if pi.shape[-1] >= 2:
        # 使用するガウス分布をpiに基づいて決定
        pis = Categorical(pi).sample().view(pi.size(0), 1, 1)
        # 決定したガウス分布に対応する平均値と標準偏差の値をgatherにより取り出し
        variance_samples = sigma.gather(1, pis).detach().squeeze()#.item()
        mean_samples = mu.detach().gather(1, pis).squeeze()#.item()
            
    for i in range(len(mu)):
        data_ids += 1
        if pi.shape[-1] >= 2:
            variance = variance_samples[i].item()
            mean = mean_samples[i].item()
        else:
            variance = sigma[i].item()
            mean = mu[i].item()

        estimated_mass.append(mean*1000)
        estimated_sd.append( variance*1000)
        #print("predict_rnd : ", predict_rnd[i])
        #print("target_rnd : ", target_rnd[i])
        rnd_score.append(criterion_rnd(predict_rnd[i], target_rnd[i]).item())

    timestamp = time.strftime("%Y-%m-%dT%H-%M-%S")
    esttimated_sd_append = "evaluation/depth40/sampler_v1/RNDMDN/results_sd"
    rnd_score_append = "evaluation/depth40/sampler_v1/RNDMDN/results_rnd"  
    os.makedirs(esttimated_sd_append, exist_ok=True)
    os.makedirs(rnd_score_append, exist_ok=True) 
    
    # Estimated_SD を保存
    sd_file_path = os.path.join(esttimated_sd_append, f"{timestamp}.csv")
    with open(sd_file_path, mode='w', newline='') as file_sd:
        writer_sd = csv.writer(file_sd)
        writer_sd.writerow(["Index", "Estimated_SD"])  # ヘッダー
        for i, sd in enumerate(estimated_sd):
            writer_sd.writerow([i, sd])
    print(f"Estimated_SD を {sd_file_path} に保存しました。")

    # RND_Score を保存
    rnd_file_path = os.path.join(rnd_score_append, f"{timestamp}.csv")
    with open(rnd_file_path, mode='w', newline='') as file_rnd:
        writer_rnd = csv.writer(file_rnd)
        writer_rnd.writerow(["Index", "RND_Score"])  # ヘッダー
        for i, score in enumerate(rnd_score):
            writer_rnd.writerow([i, score])
    print(f"RND_Score を {rnd_file_path} に保存しました。")

    print("rnd_score : ", rnd_score)
    print("estimated_mass : ", estimated_mass)
    print("estimated_sd : ", estimated_sd)

    # 3. Decide picking point x, y and calculate picking point z
    argsort_sd  = np.array(estimated_sd).argsort()  # 不確実性の値を小さい順のランキングの順位に変換
    argsort_rnd = np.array(rnd_score).argsort()     # 不確実性の値を小さい順のランキングの順位に変換
    select_score = [0]*num_candidate

    for i in range(len(mu)):
        # 候補領域の2つの不確実性の順位の総和を計算
        select_score[argsort_sd[i]]  += (i+1)
        select_score[argsort_rnd[i]] += (i+1)

    print("selct_score : " ,select_score)

    # ランキングの順位の総和が最も小さいものから呼び出し
    rank_point  = np.array(select_score).argsort()
    for i in range(len(rank_point)):
        select_point = rank_point[i]
        point_x, point_y = xy_robot_coordinate_list[select_point]
        break

    print("Time [s]            : ", time.time() - start_time)
    print("Select point(x,y)   : ", point_x, point_y)
    print("Estimated mass [g]  : ", estimated_mass[select_point])
    print("Estimated SD   [g]  : ", estimated_sd[select_point])
    print("RND score (MSE)     : ", rnd_score[select_point])
    print("Rank                : ", select_score[select_point])

    return point_x, point_y, estimated_mass[select_point], estimated_sd[select_point], rnd_score[select_point], select_score[select_point], estimated_sd, rnd_score

# ********** z座標のRGBカメラ座標とロボット座標の統一 **********
# リアルセンスで取得した輪ゴムの表面z座標
def generate_grasp_coordinates(robot_point, depth_frame,inverse_affine_matrix_2d):
    """
    ランダムなロボット座標を生成し、それをカメラ座標に変換して深度情報を統合したグリップ位置を返す。

    Args:
        robot_point: ロボット座標 (x, y)
        depth_frame: RealSenseの深度フレーム
        inverse_affine_matrix_2d: ロボット座標からカメラ座標への逆アフィン変換行列 (2x3)

    Returns:
        tuple: (z_millimeters, camera_point)
    """

    # ロボット座標を2次元形式に整形 (shape: (1, 1, 2))
    robot_point_array = np.array([[robot_point]], dtype=np.float32)

    # アフィン変換を適用してカメラ座標を取得
    camera_point = cv2.transform(robot_point_array, inverse_affine_matrix_2d)[0][0]
    print("camera_point : ", camera_point)

    # カメラ座標でのZ座標を取得
    x, y = int(camera_point[0]), int(camera_point[1])
    z_meters = depth_frame.get_distance(x, y)
    z_millimeters = z_meters * 1000  # ミリメートル単位に変換

    return z_millimeters, camera_point


# キャリブレーションデータ
RS_dict = 423  # リアルセンス: 辞書の上面までの距離 (mm)
RS_table = 472  # リアルセンス: 机の表面までの距離 (mm)
Robot_dict = 109  # ロボット: 辞書の上面までの距離 (mm)
Robot_table = 60  # ロボット: 机の表面までの距離 (mm)
thickness = 45  # 辞書の厚み (mm)

# ロボットのz値を補正する関数
def correct_robot_z(z_millimeters, mm_depth=10):
    """
    z座標を補正して、沈む量を反映した値を返します。

    Args:
        z_millimeters (float): リアルセンスのz座標
        sink_depth (float): 沈む量 (mm)

    Returns:
        float: 補正後のz座標
    """
    sink_depth = mm_depth * 1.09
    # 把持位置 = (ロボット原点とセンサ原点の差 - デプスの値 + オフセット値) - 把持深さ
    corrected_z = (532 - z_millimeters + 30) - sink_depth
    if corrected_z <= 70:
        print("changed z to 350")
        corrected_z = 350
    return corrected_z, sink_depth


def main():
    try:
        # デバイスの初期化
        print("Initializing RealSense...")
        pipeline = init_realsense()
        align_to = rs.stream.color
        align = rs.align(align_to)

        print("Initializing MyPalletizer...")
        mypalletizer = init_mypalletizer()

        # RGB画像,深度情報，把持位置，把持量を保存するためのフォルダパス
        color_folder = "evaluation/depth40/sampler_v1/RNDMDN/color"
        raw_depth_folder = "evaluation/depth40/sampler_v1/RNDMDN/raw_depth"
        csv_depth_folder = "evaluation/depth40/sampler_v1/RNDMDN/csv_depth"
        image_depth_folder = "evaluation/depth40/sampler_v1/RNDMDN/image_depth"
        xyz_robot_folder = "evaluation/depth40/sampler_v1/RNDMDN/xyz_robot"
        xyz_camera_folder = "evaluation/depth40/sampler_v1/RNDMDN/xyz_camera"
        grams_folder = "evaluation/depth40/sampler_v1/RNDMDN/grams"
        grams_folder_txt = "evaluation/depth40/sampler_v1/RNDMDN/grams_txt"
        estimated_sd_folder = "evaluation/depth40/sampler_v1/RNDMDN/estimated_sd"
        rnd_score_folder = "evaluation/depth40/sampler_v1/RNDMDN/rnd_score"
        Rank_folder = "evaluation/depth40/sampler_v1/RNDMDN/rank"


        # フォルダが存在しない場合は作成
        os.makedirs(color_folder, exist_ok=True)
        os.makedirs(raw_depth_folder, exist_ok=True)
        os.makedirs(csv_depth_folder, exist_ok=True)
        os.makedirs(image_depth_folder, exist_ok=True)
        os.makedirs(xyz_robot_folder, exist_ok=True)
        os.makedirs(xyz_camera_folder, exist_ok=True) 
        os.makedirs(grams_folder, exist_ok=True)    
        os.makedirs(grams_folder_txt, exist_ok=True) 
        os.makedirs(estimated_sd_folder, exist_ok=True)
        os.makedirs(rnd_score_folder, exist_ok=True)
        os.makedirs(Rank_folder, exist_ok=True)


        #ロボットの初期値 
        print("Devices initialized successfully!")
        #coord = [-9.7, 287.9, 108, 46]    #台座を置く場合
        coord1 = [-0.8, -210, 240, 46]       #台座を置かない場合
        coord2 = [171 , 7 , 240, 46]       #初期値1
        estimated_mass_list = []

        # 動作ループ
        while True:
            # RealSenseからフレームを取得
            frames = pipeline.wait_for_frames(timeout_ms=20000)
            aligned_frames = align.process(frames)
            depth_frame = aligned_frames.get_depth_frame()
            color_frame = aligned_frames.get_color_frame()

            if not depth_frame or not color_frame:
                continue

            if depth_frame and color_frame:
                # タイムスタンプの生成（ISO 8601形式：YYYY-MM-DDTHH-MM-SS）
                timestamp = time.strftime("%Y-%m-%dT%H-%M-%S")

                # フレームをnumpy配列に変換
                depth_image = np.asanyarray(depth_frame.get_data())
                color_image = np.asanyarray(color_frame.get_data())

                # フレームのサイズを表示
                #print(f"Depth frame size: {depth_image.shape}, Color frame size: {color_image.shape}")

                # RGB画像を保存
                rgb_filename = os.path.join(color_folder, f"{timestamp}.png")
                cv2.imwrite(rgb_filename, color_image)
                print(f"Saved RGB image as {rgb_filename}")

                # 深度データを保存（生データ形式）
                depth_raw_filename = os.path.join(raw_depth_folder, f"{timestamp}.npy")
                np.save(depth_raw_filename, depth_image)
                print(f"Saved depth data as {depth_raw_filename}")

                # 深度データをCSV形式で保存 
                depth_csv_filename = os.path.join(csv_depth_folder, f"{timestamp}.csv")
                np.savetxt(depth_csv_filename, depth_image, delimiter=",", fmt="%d")
                print(f"Saved depth data as {depth_csv_filename}")                
                
                # 深度データを画像形式（カラーマップ）で保存
                # 深度画像のクリッピング
                depth_image1 = np.clip(depth_image, 400, 600)
                # 深度画像を正規化（0-255の範囲にスケーリング）
                depth_image_normalized = cv2.normalize(depth_image1, None, 0, 255, cv2.NORM_MINMAX)

                # 深度画像をカラーマッピング
                depth_colormap = cv2.applyColorMap(depth_image_normalized.astype(np.uint8), cv2.COLORMAP_JET)

                # 深度画像を保存
                depth_image_filename = os.path.join(image_depth_folder, f"{timestamp}_depth.png")
                cv2.imwrite(depth_image_filename, depth_colormap)
                print(f"Saved depth image as {depth_image_filename}")

            else:
                print("Failed to capture frames.")

            # ********** ロボットの把持位置(x,y,z)を生成 **********
            # ランダムに把持位置を生成(x,y)
            point_x, point_y, estimated_mass, estimated_sd, rnd_score, rank, esttimated_sd_append, rnd_score_append = predict_picking(depth_frame, depth_image, color_image, inverse_affine_matrix_2d)

            estimated_mass_list.append(estimated_mass)

            #print(f"robot_point_xy: {point_x, point_y}")
            robot_point = [point_x,point_y]
            
            # カメラで取得した輪ゴム表面までの座標（(x,y) → camera_point, z → z_millimeters）
            z_millimeters, camera_point = generate_grasp_coordinates(robot_point, depth_frame,inverse_affine_matrix_2d)
            print(f"z_millimeters : {z_millimeters}")

            #カメラ座標からロボット座標に変換したz座標
            corrected_z, sink_depth = correct_robot_z(z_millimeters, mm_depth=40)
            print(f"カメラ座標からロボット座標に変換したz座標 : {corrected_z}")
            
            # x, y, z座標を統合して最終的な座標を生成
            final_grasp_position = [point_x, point_y, corrected_z]
            print(f"Final robot grasp position (x, y, z): {final_grasp_position}")

            final_grasp_position_new = final_grasp_position + [46]
        
            # ********** ロボット操作プログラム **********
            #グリッパを開く
            mypalletizer.set_gripper_state(0, 30)
            time.sleep(3)        
            #初期位置に移動
            mypalletizer.send_coords(coord1, 30, 1)
            time.sleep(3)
            mypalletizer.send_coords(coord2, 30, 1)
            time.sleep(5)
            #ある程度の高さまで移動
            final_grasp = [final_grasp_position_new[0],final_grasp_position_new[1],184,46]
            mypalletizer.send_coords(final_grasp,30,1)
            time.sleep(3)
            #把持位置に移動(zのみ)
            mypalletizer.send_coords(final_grasp_position_new,12,1)
            time.sleep(3)
            #グリッパを閉じる
            mypalletizer.set_gripper_state(1, 30)
            time.sleep(5)
            #グリッパの高さ上昇
            final_grasp = [final_grasp_position_new[0],final_grasp_position_new[1],200,46]
            mypalletizer.send_coords(final_grasp, 8 ,1)
            time.sleep(5)
            mypalletizer.send_coords(coord2, 20, 1)
            time.sleep(3)
            # 輪ゴムを計測器の位置まで移動           
            mypalletizer.send_coords(coord1, 30, 1)
            time.sleep(5)
            #グリッパを開く
            mypalletizer.set_gripper_state(0, 30)
            time.sleep(3)
            print("MyPalletizer moved successfully!")

            # ********** ロボットの把持位置の保存 **********
            # 保存するファイルのパス
            file_path = os.path.join(xyz_robot_folder, f"{timestamp}.csv")

            # 座標データをCSV形式で保存
            with open(file_path, mode="w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["X", "Y", "Z"])  # ヘッダー
                writer.writerow(final_grasp_position)  # 座標データを行として書き込む

            print(f"ロボットでの把持位置: {final_grasp_position} to {file_path}")

            # **********　カメラで取得した把持位置 **********
            # 保存するファイルのパス
            file_path = os.path.join(xyz_camera_folder, f"{timestamp}.csv")

            # [x, y, z] に統合
            grasp_coordinates = [camera_point[0], camera_point[1], z_millimeters + sink_depth]
            # 統合した座標を出力
            print(f"カメラで取得した把持位置: {grasp_coordinates}")

            # 座標データをCSV形式で保存
            with open(file_path, mode="w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["X", "Y", "Z"])  # ヘッダー
                writer.writerow(grasp_coordinates)  # 座標データを行として書き込む       

            # **********　実際の把持量と予測把持量の保存 **********
            # ファイルパス
            csv_file_path = os.path.join(grams_folder, f"{timestamp}.csv")

            # 手動で把持量を入力
            grasp_amount = float(input("Enter grasp amount (manual input): "))

            # CSV ファイルに書き込み
            if not os.path.exists(csv_file_path):
                # ファイルが存在しない場合はヘッダーを追加
                with open(csv_file_path, mode='w', newline='') as file:
                    writer = csv.writer(file)
                    writer.writerow(["predicted_mass", "target_mass"])  # ヘッダー行

            # データを追加
            with open(csv_file_path, mode='a', newline='') as file:
                writer = csv.writer(file)
                writer.writerow([estimated_mass, grasp_amount])

            print(f"Saved predicted: {estimated_mass}, target: {grasp_amount} to {csv_file_path}")


            # **********　把持量を手動で保存 **********
            grasp_amount = input("Enter grasp amount (manual input): ")

            # 保存するファイルのパス
            file_path = os.path.join(grams_folder_txt, f"{timestamp}.txt")

            # ファイルに把持量を保存
            with open(file_path, "a") as f:
                f.write(f"{grasp_amount}\n")

            print(f"Saved grasp amount: {grasp_amount} to {file_path}")
            
            #　**********　把持量のばらつきの保存 **********

            # 保存するファイルのパス
            file_path = os.path.join(estimated_sd_folder, f"{timestamp}.txt")

            # ファイルに把持量を保存
            with open(file_path, "a") as f:
                f.write(f"{estimated_sd}\n")
            
            # ********** 未知の度合いの保存 **********
            
            # 保存するファイルのパス
            file_path = os.path.join(rnd_score_folder, f"{timestamp}.txt")

            # ファイルに把持量を保存
            with open(file_path, "a") as f:
                f.write(f"{rnd_score}\n")
            
            # ********** 順位の保存 **********
            # 保存するファイルのパス
            file_path = os.path.join(Rank_folder, f"{timestamp}.txt")

            # ファイルに把持量を保存
            with open(file_path, "a") as f:
                f.write(f"{rank}\n")

            # 次の動作まで待機
            time.sleep(5)

    except KeyboardInterrupt:
        print("Program interrupted by user.")

    finally:
        # 終了処理
        pipeline.stop()
        print("RealSense pipeline stopped.")
        print("Exiting program.")

if __name__ == "__main__":
    main()