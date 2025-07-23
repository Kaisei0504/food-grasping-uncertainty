import torch
import torch.nn as nn
from torch.distributions import Categorical


# mixture density networksのヘッド
class MDN(nn.Module):
    def __init__(self, in_features, out_features, num_gaussians):
        super(MDN, self).__init__()
        self.in_features = in_features      # 入力の特徴ベクトルの次元数
        self.out_features = out_features    # 出力の次元数
        self.num_gaussians = num_gaussians  # ガウス分布の数
        # 特徴ベクトルを入力としてガウス分布の数分の混合係数 pi を出力するLinear層
        #  * 混合係数の総和は1となる必要があるためソフトマックス関数で正規化
        self.pi = nn.Sequential(nn.Linear(in_features, num_gaussians),
                                nn.Softmax(dim=1))
        # 特徴ベクトルを入力としてガウス分布の数分の標準偏差 sigma を出力するLinear層
        self.sigma = nn.Linear(in_features, out_features * num_gaussians)
        # 特徴ベクトルを入力としてガウス分布の数分の平均 mu を出力するLinear層
        self.mu = nn.Linear(in_features, out_features * num_gaussians)

    def forward(self, minibatch):
        # 混合係数
        pi = self.pi(minibatch)
        # 標準偏差
        sigma = torch.exp(self.sigma(minibatch))  # 標準偏差は0以上なので指数関数を適用
        sigma = sigma.view(-1, self.num_gaussians, self.out_features)  # (データ数, ガウス分布の数*標準偏差の数) -> (データ数, ガウス分布の数, 標準偏差の数)
        # 平均
        mu = self.mu(minibatch)
        mu = mu.view(-1, self.num_gaussians, self.out_features)  # (データ数, ガウス分布の数*平均値の数) -> (データ数, ガウス分布の数, 平均値の数)
        return pi, sigma, mu


class MDN_act_sigma(nn.Module):
    # 標準偏差 sigma に活性化関数を適用 (以下の論文のテクニックを導入)
    # Mixture Density Networks for distribution and uncertainty estimation
    def __init__(self, in_features, out_features, num_gaussians):
        super(MDN_act_sigma, self).__init__()
        self.in_features   = in_features    # 入力の特徴ベクトルの次元数
        self.out_features  = out_features   # 出力の次元数
        self.num_gaussians = num_gaussians  # ガウス分布の数
        # 特徴ベクトルを入力としてガウス分布の数分の混合係数 pi を出力するLinear層
        #  * 混合係数の総和は1となる必要があるためソフトマックス関数で正規化
        self.pi = nn.Sequential(nn.Linear(in_features, num_gaussians),
                                nn.Softmax(dim=1))
        # 特徴ベクトルを入力としてガウス分布の数分の標準偏差 sigma を出力するLinear層
        self.sigma = nn.Linear(in_features, out_features * num_gaussians)
        # 特徴ベクトルを入力としてガウス分布の数分の平均 mu を出力するLinear層
        self.mu = nn.Linear(in_features, out_features * num_gaussians)
        # 標準偏差 sigmaに適用する活性化関数
        self.act = nn.ELU()
        
    def forward(self, minibatch):
        # 混合係数
        pi = self.pi(minibatch)
        # 標準偏差
        sigma = self.sigma(minibatch)        # 標準偏差 sigma を予測
        sigma = self.act(sigma) + 1 + 1e-15  # 活性化関数を適用, +1:値の範囲を0~に変更, +1e-15:0除算を避けるための値
        sigma = sigma.view(-1, self.num_gaussians, self.out_features)  # (データ数, ガウス分布の数*標準偏差の数) -> (データ数, ガウス分布の数, 標準偏差の数)
        # 平均
        mu = self.mu(minibatch)
        mu = mu.view(-1, self.num_gaussians, self.out_features)  # (データ数, ガウス分布の数*平均値の数) -> (データ数, ガウス分布の数, 平均値の数)
        return pi, sigma, mu


    # mixture density networksの平均値と標準偏差に基づいたガウス分布からサンプルを作成
    def sample(pi, sigma, mu, cuda=False):
        # Categorical(pi) : 混合係数の関係性に基づいたカテゴリカル分布の作成
        # Categorical(pi).sample() : カテゴリカル分布からサンプリング = 使用するガウス分布の選択
        # pis : 使用するガウス分布の番号
        pis = Categorical(pi).sample().view(pi.size(0), 1, 1)
        # 乱数を生成
        gaussian_noise = torch.randn((sigma.size(2), sigma.size(0)), requires_grad=False)
        if cuda:
            gaussian_noise = gaussian_noise.cuda()
        # pisの値に対応するガウス分布の平均値と標準偏差の値をgatherにより取り出し
        variance_samples = sigma.gather(1, pis).detach().squeeze()
        mean_samples = mu.detach().gather(1, pis).squeeze()
        # 平均値と標準偏差に基づいた乱数を生成
        return (gaussian_noise * variance_samples + mean_samples).transpose(0, 1)


# ネットワーク構造
class CNN_MDN(nn.Module):
    def __init__(self, num_gaussians=1):
        # 想定 : Input_size = (150,150)
        super(CNN_MDN, self).__init__()
        self.rgb_conv1 = nn.Conv2d( 3, 16, kernel_size=3, stride=1)
        self.rgb_conv2 = nn.Conv2d(16, 16, kernel_size=3, stride=1)
        self.rgb_conv3 = nn.Conv2d(16, 16, kernel_size=3, stride=1)
        self.rgb_conv4 = nn.Conv2d(16, 16, kernel_size=3, stride=1)
        self.rgb_conv5 = nn.Conv2d(16, 16, kernel_size=3, stride=1)

        self.depth_conv1 = nn.Conv2d( 3, 16, kernel_size=3, stride=1)
        self.depth_conv2 = nn.Conv2d(16, 16, kernel_size=3, stride=1)
        self.depth_conv3 = nn.Conv2d(16, 16, kernel_size=3, stride=1)
        self.depth_conv4 = nn.Conv2d(16, 16, kernel_size=3, stride=1)
        self.depth_conv5 = nn.Conv2d(16, 16, kernel_size=3, stride=1)

        #self.lz = nn.Linear(1, 1)
        self.l1 = nn.Linear(128, 128)
        self.l2 = nn.Linear(128, 64)
        self.l3 = nn.Linear(65, 65)
        self.mdn = MDN(65, 1, num_gaussians)

        self.act = nn.ELU()
        self.pool = nn.MaxPool2d(2, 2)

    def forward(self, x_rgb, x_depth, z_gripper): # x_rgb, x_depth:[batch, 3, 150, 150], z_gripper:[batch, 1]
        # RGB画像から畳み込み層により特徴マップを抽出
        h_rgb = self.pool(self.act(self.rgb_conv1(x_rgb)))  # 出力される特徴マップのサイズ: [batch, 16, 74, 74]
        #print(h_rgb.shape)
        h_rgb = self.pool(self.act(self.rgb_conv2(h_rgb)))  # 出力される特徴マップのサイズ: [batch, 16, 36, 36]
        #print(h_rgb.shape)
        h_rgb = self.pool(self.act(self.rgb_conv3(h_rgb)))  # 出力される特徴マップのサイズ: [batch, 16, 17, 17]
        #print(h_rgb.shape)
        h_rgb = self.pool(self.act(self.rgb_conv4(h_rgb)))  # 出力される特徴マップのサイズ: [batch, 16,  7,  7]
        #print(h_rgb.shape)
        h_rgb = self.pool(self.act(self.rgb_conv5(h_rgb)))  # 出力される特徴マップのサイズ: [batch, 16,  2,  2]
        #print(h_rgb.shape)
        #print(" ")
        # Depth画像から畳み込み層により特徴マップを抽出
        h_depth = self.pool(self.act(self.depth_conv1(x_depth)))  # 出力される特徴マップのサイズ: [batch, 16, 74, 74]
        h_depth = self.pool(self.act(self.depth_conv2(h_depth)))  # 出力される特徴マップのサイズ: [batch, 16, 36, 36]
        h_depth = self.pool(self.act(self.depth_conv3(h_depth)))  # 出力される特徴マップのサイズ: [batch, 16, 17, 17]
        h_depth = self.pool(self.act(self.depth_conv4(h_depth)))  # 出力される特徴マップのサイズ: [batch, 16,  7,  7]
        h_depth = self.pool(self.act(self.depth_conv5(h_depth)))  # 出力される特徴マップのサイズ: [batch, 16,  2,  2]

        # 特徴マップの配列サイズを [batch, channel, height, width] --> [batch, channel * height * width] に変更
        h_rgb   = h_rgb.view(h_rgb.size()[0], -1)      # 出力される特徴ベクトルのサイズ: [batch, 64]
        h_depth = h_depth.view(h_depth.size()[0], -1)  # 出力される特徴ベクトルのサイズ: [batch, 64]

        # 画像の特徴量とDepthの特徴量をConcat
        h = torch.cat((h_rgb, h_depth), 1)  # 出力される特徴ベクトルのサイズ: [batch, 128]

        # FC層により特徴量を抽出
        h = self.act(self.l1(h))  # 出力される特徴ベクトルのサイズ: [batch, 128]
        h = self.act(self.l2(h))  # 出力される特徴ベクトルのサイズ: [batch, 64]

        # Insertion depth
        #z_gripper = self.lz(z_gripper) # mm単位のz_gripperに対していい感じのスケーリングをしてくれることを期待
        h = torch.cat((h, z_gripper), 1)  # 出力される特徴ベクトルのサイズ: [batch, 65]

        # FC層により特徴量を抽出
        h = self.act(self.l3(h))     # 出力される特徴ベクトルのサイズ: [batch, 65]
        pi, sigma, mu = self.mdn(h)  # 出力サイズ pi: [batch, num_gaussians], sigma: [batch, num_gaussians, 1], mu: [batch, num_gaussians, 1]
        return pi, sigma, mu
    
    
class CNN_MDN_act_sigma(nn.Module):
    def __init__(self, num_gaussians=1):
        # 想定 : Input_size = (150,150)
        super(CNN_MDN_act_sigma, self).__init__()
        self.rgb_conv1 = nn.Conv2d( 3, 16, kernel_size=3, stride=1)
        self.rgb_conv2 = nn.Conv2d(16, 16, kernel_size=3, stride=1)
        self.rgb_conv3 = nn.Conv2d(16, 16, kernel_size=3, stride=1)
        self.rgb_conv4 = nn.Conv2d(16, 16, kernel_size=3, stride=1)
        self.rgb_conv5 = nn.Conv2d(16, 16, kernel_size=3, stride=1)

        self.depth_conv1 = nn.Conv2d( 3, 16, kernel_size=3, stride=1)
        self.depth_conv2 = nn.Conv2d(16, 16, kernel_size=3, stride=1)
        self.depth_conv3 = nn.Conv2d(16, 16, kernel_size=3, stride=1)
        self.depth_conv4 = nn.Conv2d(16, 16, kernel_size=3, stride=1)
        self.depth_conv5 = nn.Conv2d(16, 16, kernel_size=3, stride=1)

        #self.lz = nn.Linear(1, 1)
        self.l1 = nn.Linear(128, 128)
        self.l2 = nn.Linear(128, 64)
        self.l3 = nn.Linear(65, 65)
        self.mdn = MDN_act_sigma(65, 1, num_gaussians)

        self.act = nn.ELU()
        self.pool = nn.MaxPool2d(2, 2)

    def forward(self, x_rgb, x_depth, z_gripper): # x_rgb, x_depth:[batch, 3, 150, 150], z_gripper:[batch, 1]
        # RGB画像から畳み込み層により特徴マップを抽出
        h_rgb = self.pool(self.act(self.rgb_conv1(x_rgb)))  # 出力される特徴マップのサイズ: [batch, 16, 74, 74]
        h_rgb = self.pool(self.act(self.rgb_conv2(h_rgb)))  # 出力される特徴マップのサイズ: [batch, 16, 36, 36]
        h_rgb = self.pool(self.act(self.rgb_conv3(h_rgb)))  # 出力される特徴マップのサイズ: [batch, 16, 17, 17]
        h_rgb = self.pool(self.act(self.rgb_conv4(h_rgb)))  # 出力される特徴マップのサイズ: [batch, 16,  7,  7]
        h_rgb = self.pool(self.act(self.rgb_conv5(h_rgb)))  # 出力される特徴マップのサイズ: [batch, 16,  2,  2]

        # Depth画像から畳み込み層により特徴マップを抽出
        h_depth = self.pool(self.act(self.depth_conv1(x_depth)))  # 出力される特徴マップのサイズ: [batch, 16, 74, 74]
        h_depth = self.pool(self.act(self.depth_conv2(h_depth)))  # 出力される特徴マップのサイズ: [batch, 16, 36, 36]
        h_depth = self.pool(self.act(self.depth_conv3(h_depth)))  # 出力される特徴マップのサイズ: [batch, 16, 17, 17]
        h_depth = self.pool(self.act(self.depth_conv4(h_depth)))  # 出力される特徴マップのサイズ: [batch, 16,  7,  7]
        h_depth = self.pool(self.act(self.depth_conv5(h_depth)))  # 出力される特徴マップのサイズ: [batch, 16,  2,  2]

        # 特徴マップの配列サイズを [batch, channel, height, width] --> [batch, channel * height * width] に変更
        h_rgb   = h_rgb.view(h_rgb.size()[0], -1)      # 出力される特徴ベクトルのサイズ: [batch, 64]
        h_depth = h_depth.view(h_depth.size()[0], -1)  # 出力される特徴ベクトルのサイズ: [batch, 64]

        # 画像の特徴量とDepthの特徴量をConcat
        h = torch.cat((h_rgb, h_depth), 1)  # 出力される特徴ベクトルのサイズ: [batch, 128]

        # FC層により特徴量を抽出
        h = self.act(self.l1(h))  # 出力される特徴ベクトルのサイズ: [batch, 128]
        h = self.act(self.l2(h))  # 出力される特徴ベクトルのサイズ: [batch, 64]

        # Insertion depth
        #z_gripper = self.lz(z_gripper) # mm単位のz_gripperに対していい感じのスケーリングをしてくれることを期待
        h = torch.cat((h, z_gripper), 1)  # 出力される特徴ベクトルのサイズ: [batch, 65]

        # FC層により特徴量を抽出
        h = self.act(self.l3(h))     # 出力される特徴ベクトルのサイズ: [batch, 65]
        pi, sigma, mu = self.mdn(h)  # 出力サイズ pi: [batch, num_gaussians], sigma: [batch, num_gaussians, 1], mu: [batch, num_gaussians, 1]
        return pi, sigma, mu



# Random Network Distillation (RND)
class CNN_RND(nn.Module):
    def __init__(self):
        # 想定 : Input_size = (150,150)
        super(CNN_RND, self).__init__()
        self.rgb_conv1 = nn.Conv2d( 3, 16, kernel_size=3, stride=1)
        self.rgb_conv2 = nn.Conv2d(16, 16, kernel_size=3, stride=1)
        self.rgb_conv3 = nn.Conv2d(16, 16, kernel_size=3, stride=1)
        self.rgb_conv4 = nn.Conv2d(16, 16, kernel_size=3, stride=1)
        self.rgb_conv5 = nn.Conv2d(16, 16, kernel_size=3, stride=1)

        self.depth_conv1 = nn.Conv2d( 3, 16, kernel_size=3, stride=1)
        self.depth_conv2 = nn.Conv2d(16, 16, kernel_size=3, stride=1)
        self.depth_conv3 = nn.Conv2d(16, 16, kernel_size=3, stride=1)
        self.depth_conv4 = nn.Conv2d(16, 16, kernel_size=3, stride=1)
        self.depth_conv5 = nn.Conv2d(16, 16, kernel_size=3, stride=1)

        self.l1 = nn.Linear(128, 128)
        self.l2 = nn.Linear(128, 64)

        self.act = nn.ELU()
        self.pool = nn.MaxPool2d(2, 2)

    def forward(self, x_rgb, x_depth): # x_rgb, x_depth:[batch, 3, 150, 150], z_gripper:[batch, 1]
        # RGB画像から畳み込み層により特徴マップを抽出
        h_rgb = self.pool(self.act(self.rgb_conv1(x_rgb)))  # 出力される特徴マップのサイズ: [batch, 16, 74, 74]
        h_rgb = self.pool(self.act(self.rgb_conv2(h_rgb)))  # 出力される特徴マップのサイズ: [batch, 16, 36, 36]
        h_rgb = self.pool(self.act(self.rgb_conv3(h_rgb)))  # 出力される特徴マップのサイズ: [batch, 16, 17, 17]
        h_rgb = self.pool(self.act(self.rgb_conv4(h_rgb)))  # 出力される特徴マップのサイズ: [batch, 16,  7,  7]
        h_rgb = self.pool(self.act(self.rgb_conv5(h_rgb)))  # 出力される特徴マップのサイズ: [batch, 16,  2,  2]

        # Depth画像から畳み込み層により特徴マップを抽出
        h_depth = self.pool(self.act(self.depth_conv1(x_depth)))  # 出力される特徴マップのサイズ: [batch, 16, 74, 74]
        h_depth = self.pool(self.act(self.depth_conv2(h_depth)))  # 出力される特徴マップのサイズ: [batch, 16, 36, 36]
        h_depth = self.pool(self.act(self.depth_conv3(h_depth)))  # 出力される特徴マップのサイズ: [batch, 16, 17, 17]
        h_depth = self.pool(self.act(self.depth_conv4(h_depth)))  # 出力される特徴マップのサイズ: [batch, 16,  7,  7]
        h_depth = self.pool(self.act(self.depth_conv5(h_depth)))  # 出力される特徴マップのサイズ: [batch, 16,  2,  2]

        # 特徴マップの配列サイズを [batch, channel, height, width] --> [batch, channel * height * width] に変更
        h_rgb   = h_rgb.view(h_rgb.size()[0], -1)      # 出力される特徴ベクトルのサイズ: [batch, 64]
        h_depth = h_depth.view(h_depth.size()[0], -1)  # 出力される特徴ベクトルのサイズ: [batch, 64]

        # 画像の特徴量とDepthの特徴量をConcat
        h = torch.cat((h_rgb, h_depth), 1)  # 出力される特徴ベクトルのサイズ: [batch, 128]

        # FC層により特徴量を抽出
        h = self.act(self.l1(h))  # 出力される特徴ベクトルのサイズ: [batch, 128]
        h = self.act(self.l2(h))  # 出力される特徴ベクトルのサイズ: [batch, 64]
        return h

