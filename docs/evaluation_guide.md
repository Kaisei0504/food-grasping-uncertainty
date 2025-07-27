## 📊 評価手順ガイド

### 比較手法一覧

| 手法 | パス変更先 | 説明 |
|------|------------|------|
| **RNDのみ** | `no_sampler_RND_only` | 未知度のみ考慮 |
| **MDNのみ** | `no_sampler_MDN_only` | 把持量ばらつきのみ考慮 |
| **RND+MDN** | `no_sampler_RNDMDN` | 両方の不確実性を考慮 |
| **RND+MDN+Sampler** | `show_results_with_RND_sampler.ipynb`で確認 | **提案手法** |

### パス変更方法

`show_results_with_RND_no_sampler.ipynb`内の以下のコードを変更：

```python
# 評価用データ
with open('./datasets/evaluate_data/no_sampler_RND_only/divide_ids/data_test_100.pickle', mode='br') as fi:
#                                   ^^^^^^^^^^^^^^^^^^
#                                   この部分を上記表のパス変更先に変更
```