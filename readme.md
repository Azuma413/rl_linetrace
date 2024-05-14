# このリポジトリについて
DrQ-v2を用いてWebカメラから取得した画像を観測とするライントレースを行うプログラムです。
https://github.com/facebookresearch/drqv2


ラズベリーパイ上で動作させることを想定しています。（メモリ8GB以下）

# 実機動作(ラズパイ上で実行)
```
cd [this package path]
source ~/rl_env/bin/activate
python3 /drqv2/main.py
```
# 評価(wsl2で実行)
```
cd [this package path]
conda activate jax_env
python my_eval.py
```
# トレーニング(wsl2で実行)
```
cd [this package path]
conda activate jax_env
python train.py
```