import pyrealsense2 as rs
import numpy as np
import time
import cv2
import random
import matplotlib.pyplot as plt
from pymycobot.mypalletizer import MyPalletizer
import os
import csv

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

### ランダムなロボット座標を生成する関数 ###########################################################
def generate_random_robot_position():
    """
    ランダムなロボット座標を生成する。

    Returns:
        ndarray: [ロボット座標のX, ロボット座標のY]（NumPy配列形式）
    """
    robot_x = random.uniform(156, 240)  # ロボット座標系でのX範囲
    robot_y = random.uniform(-117, 117) 
    return np.array([robot_x, robot_y], dtype=np.float32)

### z座標のRGBカメラ座標とロボット座標の統一 ##########################################################
# リアルセンスで取得した輪ゴムの表面z座標
def generate_grasp_coordinates(robot_point, depth_frame, inverse_affine_matrix_2d):
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
############################################################################################################

### メイン関数 #############################################################################################
def main():
    try:
        # デバイスの初期化
        print("Initializing RealSense...")
        pipeline = init_realsense()
        align_to = rs.stream.color
        align = rs.align(align_to)

        print("Initializing MyPalletizer...")
        mypalletizer = init_mypalletizer()

        #ロボットの初期値 
        print("Devices initialized successfully!")
        coord1 = [-0.8, -210, 240, 46]    #輪ゴムを落として量る位置
        coord2 = [171 , 7 , 240, 46]      

        # RGB画像と深度情報を保存するためのフォルダパス
        color_folder = "new_dataset_er21046/train100/color"
        raw_depth_folder = "new_dataset_er21046/train100/raw_depth"
        csv_depth_folder = "new_dataset_er21046/train100/csv_depth"
        image_depth_folder = "new_dataset_er21046/train100/image_depth"

        # フォルダが存在しない場合は作成
        os.makedirs(color_folder, exist_ok=True)
        os.makedirs(raw_depth_folder, exist_ok=True)
        os.makedirs(csv_depth_folder, exist_ok=True)
        os.makedirs(image_depth_folder, exist_ok=True)

        # 把持位置(ロボットのxyz)を保存するためのフォルダパス
        xyz_robot_folder = "new_dataset_er21046/train100/xyz_robot"
        # フォルダが存在しない場合は作成
        if not os.path.exists(xyz_robot_folder):
            os.makedirs(xyz_robot_folder)

        # 把持位置(カメラのxyz)を保存するためのフォルダパス
        xyz_camera_folder = "new_dataset_er21046/train100/xyz_camera"
        # フォルダが存在しない場合は作成
        if not os.path.exists(xyz_camera_folder):
            os.makedirs(xyz_camera_folder)        

        # 把持量を保存するためのフォルダパス
        grams_folder = "new_dataset_er21046/train100/grams"  # datasetフォルダを指定
        # datasetフォルダが存在しない場合は作成
        if not os.path.exists(grams_folder):
            os.makedirs(grams_folder)

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
                # フレームをnumpy配列に変換
                depth_image = np.asanyarray(depth_frame.get_data())
                color_image = np.asanyarray(color_frame.get_data())

                # フレームのサイズを表示
                print(f"Depth frame size: {depth_image.shape}, Color frame size: {color_image.shape}")

                # タイムスタンプの生成（ISO 8601形式：YYYY-MM-DDTHH-MM-SS）
                timestamp = time.strftime("%Y-%m-%dT%H-%M-%S")

                # RGB画像を保存
                rgb_filename = os.path.join(color_folder, f"{timestamp}.png")
                cv2.imwrite(rgb_filename, color_image)
                print(f"Saved RGB image as {rgb_filename}")

                ### 深度データをCSV形式で保存 ###############################################################
                depth_csv_filename = os.path.join(csv_depth_folder, f"{timestamp}.csv")
                np.savetxt(depth_csv_filename, depth_image, delimiter=",", fmt="%d")
                print(f"Saved depth data as {depth_csv_filename}")    

                ### 深度データを保存（生データ形式）########################################################
                depth_raw_filename = os.path.join(raw_depth_folder, f"{timestamp}.npy")
                np.save(depth_raw_filename, depth_image)
                print(f"Saved depth data as {depth_raw_filename}")

                ### 深度データを画像形式（カラーマップ）で保存する場合 #######################################
                # 深度画像を正規化（0-255の範囲にスケーリング）
                # 深度画像のクリッピング
                depth_image1 = np.clip(depth_image, 400, 600)
                depth_image_normalized = cv2.normalize(depth_image1, None, 0, 255, cv2.NORM_MINMAX)

                # 深度画像をカラーマッピング
                depth_colormap = cv2.applyColorMap(depth_image_normalized.astype(np.uint8), cv2.COLORMAP_JET)

                # 深度画像を保存
                depth_image_filename = os.path.join(image_depth_folder, f"{timestamp}_depth.png")
                cv2.imwrite(depth_image_filename, depth_colormap)
                print(f"Saved depth image as {depth_image_filename}")
                
                #################################################################################################

            else:
                print("Failed to capture frames.")

            ### ロボットの把持位置(x,y,z)を生成 ####################################
            # ランダムに把持位置を生成(x,y)
            robot_point = generate_random_robot_position()
            print(f"robot point: {robot_point}")
            
            # カメラで取得した輪ゴム表面までの座標（(x,y) → camera_point, z → z_millimeters）
            z_millimeters, camera_point = generate_grasp_coordinates(robot_point, depth_frame, inverse_affine_matrix_2d)
            print(f"z_millimeters : {z_millimeters}")
            print(f"camera_point : {camera_point}")

            #カメラ座標からロボット座標に変換したz座標
            corrected_z, sink_depth = correct_robot_z(z_millimeters, mm_depth=40)
            print(f"カメラ座標からロボット座標に変換したz座標 : {corrected_z}")
            
            # x, y, z座標を統合して最終的な座標を生成
            final_grasp_position = [robot_point[0], robot_point[1], corrected_z]
            print(f"Final robot grasp position (x, y, z): {final_grasp_position}")

            final_grasp_position_new = final_grasp_position + [46]
            ##########################################################################
            ### ロボット操作プログラム ###############################################
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
            mypalletizer.send_coords(coord2, 30, 1)
            time.sleep(3)
            # 輪ゴムを計測器の位置まで移動           
            mypalletizer.send_coords(coord1, 30, 1)
            time.sleep(5)
            #グリッパを開く
            mypalletizer.set_gripper_state(0, 30)
            time.sleep(3)
            print("MyPalletizer moved successfully!")

            ### 把持位置の保存 ########################################################
            ### ロボットの把持位置 ############################

            # 保存するファイルのパス
            file_path = os.path.join(xyz_robot_folder, f"{timestamp}.csv")

            # 座標データをCSV形式で保存
            with open(file_path, mode="w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["X", "Y", "Z"])  # ヘッダー
                writer.writerow(final_grasp_position)  # 座標データを行として書き込む

            print(f"ロボットでの把持位置: {final_grasp_position} to {file_path}")

            ### カメラで取得した把持位置 ########################
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

            ### 把持量を手動で保存 ################################################
            grasp_amount = input("Enter grasp amount (manual input): ")

            # 保存するファイルのパス
            file_path = os.path.join(grams_folder, f"{timestamp}.txt")

            # ファイルに把持量を保存
            with open(file_path, "a") as f:
                f.write(f"{grasp_amount}\n")

            print(f"Saved grasp amount: {grasp_amount} to {file_path}")
            #######################################################################

            # 次の動作まで待機
            time.sleep(3)

    except KeyboardInterrupt:
        print("Program interrupted by user.")

    finally:
        # 終了処理
        pipeline.stop()
        print("RealSense pipeline stopped.")
        print("Exiting program.")

if __name__ == "__main__":
    main()
