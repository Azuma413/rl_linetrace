# このリポジトリについて
[DrQ-v2](https://github.com/facebookresearch/drqv2)を用いてWebカメラから取得した画像を観測とするライントレースを行うプログラムです。  
ラズベリーパイ5上で動作させることを想定しています。

# アクチュエータ，センサーについて
アクチュエータは[タミヤのウォームギヤーボックスHE](https://www.tamiya.com/japan/products/72004/index.html)使います。  
センサーはロジクールのWebカメラを使います。  
モータードライバは[TB628A4 6612FNG](https://toshiba.semicon-storage.com/jp/semiconductor/product/motor-driver-ics/brushed-dc-motor-driver-ics/detail.TB6612FNG.html)を使います。
![image.png](images/システム電源.drawio.png)

# ソフトウェア全体について
ソフトウェアは学習段階と実行段階に分けて考えることができます。
## 学習段階
学習段階ではシミュレータを用いて強化学習モデルの学習を行います。  
強化学習には詳しく後述します。
## 実行段階
実行段階ではWebカメラから取り込んだ映像を加工して，できるだけシミュレータの観測に似たものを作成します。  


# 強化学習について
強化学習の基本要素は`環境`と`Agent`（頭脳）です。

# 使い方
## インストール
Ubuntuへのインストールを想定しています。(Windowsならwsl2を使うこと)
```
git clone https://github.com/Azuma413/rl_linetrace.git
```
環境構築については`/drqv2/conda_env.yml`を使えば簡単に仮想環境を構築できるはずなのですが，なぜかその環境でコードを実行するとエラーが出るので，試行錯誤しながら環境構築するしかなさそうです。

## 実機動作(ラズパイ上で実行)
windowsのターミナルからssh接続します。
```
ssh raspi3@raspi3.local
cd [this package path]
source ~/rl_env/bin/activate
python3 /drqv2/main.py
```

## 評価(wsl2で実行)
`jax_env`はanacondaの仮想環境です。適宜自分の環境名に置き換えてください。
```
cd [this package path]
conda activate jax_env
python my_eval.py
```

## トレーニング(wsl2で実行)
```
cd [this package path]
conda activate jax_env
python train.py
```