## 📊 評価手順ガイド2

### 学習後の評価（モデルパス変更が必要）
学習実行後は、各ノートブック内のモデル読み込みパスを変更してください：

**`show_results_with_RND_no_sampler.ipynb`**
```python
# 変更前（デフォルト）
load_path = "./saved_models/2000epoch_lr0x001_NAdam_100samples_with_RND_sampler"

# 変更後（学習後）  
load_path = "./saved_models/2000epoch_lr0x001_NAdam_100samples_with_RND_own"
#  
```

**`show_results_with_RND_sampler.ipynb`**
```python
# 変更前（デフォルト）
load_path = "./saved_models/2000epoch_lr0x001_NAdam_100samples_with_RND"

# 変更後（学習後）
load_path = "./saved_models/2000epoch_lr0x001_NAdam_100samples_with_RND_sampler_own"
#        
```

**`show_results_with_RND_no_sampler.ipynb`使用の場合，以下のコードを変更することで各手法の比較可能：**

### 比較手法一覧
| 手法 | パス変更先 | 説明 |
|------|------------|------|
| **RNDのみ** | `no_sampler_RND_only` | 未知度のみ考慮 |
| **MDNのみ** | `no_sampler_MDN_only` | 把持量ばらつきのみ考慮 |
| **RND+MDN** | `no_sampler_RNDMDN` | 両方の不確実性を考慮 |
| **RND+MDN+Sampler** | `show_results_with_RND_sampler.ipynb`で確認 | **提案手法** |

```python
# 評価用データ
with open('./datasets/evaluate_data/no_sampler_RND_only/divide_ids/data_test_100.pickle', mode='br') as fi:
#                                   ^^^^^^^^^^^^^^^^^^
#                             この部分を上記表のパス変更先に変更
```

```python
# データセット
test_data = dataset_factory_v1.RGBD_DATASET(root="./datasets/evaluate_data/no_sampler_RND_only", use_ids = id_test, train=False, img_size=150, crop_size=140)
#                                                                          ^^^^^^^^^^^^^^^^^^^
#                                                                             同じパスに変更
```

```python
path_list = sorted(glob.glob( "./datasets/evaluate_data/no_sampler_RND_only/color/*"))
#                                                       ^^^^^^^^^^^^^^^^^^^
#                                                          同じパスに変更
```

