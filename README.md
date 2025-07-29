# 🚀 複数の観点に基づく不確実性を考慮した食品把持

## ✨ 研究概要

本研究は，食品製造工場における不定形状食品の自動把持，高精度把持を目的とした研究です．ポテトサラダ等の惣菜の詰め込み作業の自動化，高精度把持に向けて，複数の観点に基づいて不確実性を考慮した深層学習アプローチを提案しています．

## ⚠️ 背景・課題

- 食品製造工場における惣菜等の詰め込み作業は人手で行われている → ロボットによる自動化
- 季節の移り変わり等でメニューが頻繁に変わる → 少数の学習データで高精度な把持
- ポテトサラダ等の不定形状食品は不確実性が発生しやすい → 不確実性を考慮した把持

## 💡 提案手法

1. **複数の観点に基づいた不確実性を考慮した把持位置選択**
   - 不確実性である未知度と把持量のばらつきの両方が低い領域を把持位置として選択
   - 推論時にモデルが最も確信を持てる領域に焦点を当てる

2. **UA-Sampler（不確実性考慮サンプラー）**
   - 学習時に未知度が低いデータを優先的に学習
   - 少数の学習データでの性能向上を実現

## 🔬 実験設定

- **把持対象**：疑似食品として輪ゴムを使用
- **評価指標**：モデルの予測把持量とピッキングロボットの実際把持量との差
- **比較方法**：各手法による把持精度の比較評価

各手法とは？

| 手法 | 説明 | 特徴 |
|------|------|------|
| **未知度のみ** | Random Network Distillationによる未知度のみを考慮 | 従来手法 |
| **ばらつきのみ** | Mixture Density Networkによる把持量のばらつきのみを考慮 | 従来手法 |
| **未知度＋ばらつき** | 未知度と把持量のばらつきの両方を考慮 | 多観点不確実性考慮 |
| **未知度+ばらつき+Sampler** | 上記 + UA-Samplerによる学習データ選択最適化 | **提案手法** |

## 📊 実験結果

| 未知度 | ばらつき | UASampler | ±0.1g | ±0.5g | ±1.0g | ±1.5g | ±2.0g | ±2.5g | ±3.0g |
| ---------- | -------- | --------- | ----- | ----- | ----- | ----- | ----- | ----- | ----- |
| ✓ |  |  | **7** | 14 | 29 | 45 | 54 | 58 | 73 |
|  | ✓ |  | 1 | 11 | 23 | 38 | 51 | 61 | 73 |
| ✓ | ✓ |  | 3 | 23 | 35 | 55 | 62 | 68 | 82 |
| ✓ | ✓ | ✓ | 6 | **31** | **49** | **64** | **74** | **81** | **88** |


✅ **未知度と把持量のばらつきの考慮により把持成功率が向上**  
✅ **UA-Sampler導入により把持成功率がさらに向上**  

## 📂 ディレクトリ構造
```
food-grasping-uncertainty/
├── docs/                                         
|   ├── evaluation_guide.md                       # 評価手順ガイド1
|   └── evaluation_guide2.md                      # 評価手順ガイド2
|
├── lib/                                          # コア実装
│   ├── models.py                                 # ニューラルネットワークモデル
│   ├── loss_func.py                              # 損失関数（MDN）
│   ├── dataset_factory.py                        # データセット生成
|   ├── dataset_factory_v1.py                     
│   └── utils.py                                  # ユーティリティ関数
│
├── scripts/                                      # 実行スクリプト
│   ├── data_collection.py                        # 実機を用いたデータ収集
│   ├── dataset_create.ipynb                      # データセット作成（学習用）
│   ├── evaluate_data_create.ipynb                # データセット作成（評価用）
│   ├── train_mdn_with_RND_no_sampler.py          # モデルの学習（UA-Samplerなし）
│   ├── train_mdn_with_RND_sampler.py             # モデルの学習（UA-Samplerあり）
│   ├── show_results_with_RND_no_sampler.ipynb    # 学習用・テスト用データに対する予測結果の評価（UA-Samplerなし）
│   ├── show_results_with_RND_sampler.ipynb       # 学習用・テスト用データに対する予測結果の評価（UA-Samplerあり）
│   └── robot_haji.py                             # 実機テスト
│
├── datasets/                                     # 📊 データセット配置先
├── saved_models/                                 # 💾 学習済みモデル配置先
├── environment.yml                               # 🔧 Conda環境
├── README.md                                     # 📖 プロジェクト説明
├── LICENSE                                       # ⚖️ ライセンス
└── .gitignore                                    # 🚫 Git除外設定
```

## 🚀 クイックスタート（再現実験）

### 🛠 Conda のインストール（Conda が入っていない場合）

以下の手順で Miniconda をインストールしてください.（Ubuntu の例）

```bash
cd ~
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh
```

### ⚙️ 環境構築
このリポジトリは NVIDIA NGC の PyTorch コンテナ (23.08) 相当の環境を想定しています．
以下の `environment.yml` を使って Conda 環境を構築してください．

#### 環境ファイルから環境作成
```bash
conda env create -f environment.yml
```
#### 環境をアクティベート
```bash
conda activate mdn_rnd_foods
```

### 📥 データセットのダウンロード

以下のリンクから **datasets.zip** をダウンロードしてください。

<div align="center">
<a href="https://drive.google.com/file/d/12knaV2kl6SQ5Z_WgBstMfIae5uKI1By0/view?usp=sharing">
📥 <b>datasets.zip</b>
</a>
</div>
<br>

ダウンロード後，解凍して `datasets/` フォルダをプロジェクト直下に配置してください．

### 📥 学習済みモデルのダウンロード

すぐに実験を再現できるように、学習済みモデルを用意しています。  
以下のリンクから **saved_models.zip** をダウンロードしてください。

<div align="center">
<a href="https://drive.google.com/file/d/1sxDVwFR5AENHmJfcUAM32IJDczkuxUOO/view?usp=sharing">
📥 <b>saved_models.zip</b>
</a>
</div>
<br>

ダウンロード後，解凍して `saved_models/` フォルダをプロジェクト直下に配置してください．


### ⚡評価のみ実行（約10分）
各比較手法の把持成功率を確認できます．

#### 未知度+ばらつき+Sampler の評価（提案手法）
1. JupyterLab を起動
```bash
jupyter lab
```
2. ノートブックを開く
JupyterLab がブラウザで開いたら、
`scripts/` フォルダの中にある `show_results_with_RND_sampler.ipynb` を選択して開きます．

3. セルを実行（セル」→「すべて実行」or Shift + Enter で上から順に）

#### その他の手法の評価
1. ノートブックを開く
JupyterLab がブラウザで開いたら，
`scripts/` フォルダの中にある `show_results_with_RND_no_sampler.ipynb` を選択して開きます．

2. パス変更が必要

   **📋 詳細な評価手順・比較手法については → [評価手順ガイド1](docs/evaluation_guide.md)**

3. セルを実行（「セル」→「すべて実行」or Shift + Enter で上から順に）

### ⚡学習から実行（約90分）
各比較手法の把持成功率を確認できます．

#### 未知度+ばらつき+Sampler の評価（提案手法）
1. JupyterLab を起動 (起動してない場合)
```bash
jupyter lab
```

2. 学習用コードをターミナルで実行
```bash
python train_mdn_with_RND_sampler.py
```
3. `scripts/` フォルダの中にある `show_results_with_RND_sampler.ipynb` を選択して開き，実行

#### その他の手法の評価

1. 学習用コードをターミナルで実行
```bash
python train_mdn_with_RND_no_sampler.py
```
2. `scripts/` フォルダの中にある `show_results_with_RND_no_sampler.ipynb` を選択して開き，実行

   **📋 詳細な評価手順・比較手法については → [評価手順ガイド2](docs/evaluation_guide2.md)**


## ⚠️ 補足・注意事項

📌 **実機を用いたデータ収集やロボットテストの詳細な手順は，今後随時更新予定です．**  
最新の手順や追加情報は，このリポジトリの README または `docs/` ディレクトリをご確認ください．

