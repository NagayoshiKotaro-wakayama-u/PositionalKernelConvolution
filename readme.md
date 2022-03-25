# **実行環境について**

すべてのコードは以下に存在します
~~~
nagayoshi@deep2:~/research/PositionalKernelConvolution$
~~~

# <font color="Yellow">**学習**</font>
## 概要
~~~
python main.py experiment [-dataset] [-imgw] [-imgh] [-lr] [-pt] [-ope] [-learnSitePKConv] [-learnSitePKConv] [-branchlSitePConv] [-branchlSitePKConv] [-siteLayers] [-phase] [-nonNegSConv] [-existOnly] [-lossw] [-epochs] [-patience] [-switchPhase]
~~~

[　]は省略可能

引数がNoneとなっているものは引数不要です。

***
- experiment
    - String : 実験名(必須)
***
- dataset
    - String : データセットのディレクトリ名(default="toyData")
***
- imgw
    - Int : 入力画像の横幅(default=512)
***
- imgh
    - Int : 入力画像の縦幅(default=512)
***
- lr
    - Float : 学習率(default=0.01)
***
- pt
    - String : pre-Train済みのモデルの実験名を指定（fine-Tuningなど行う場合に使用）
***
- ope
    - String : 位置特性の演算方法。位置特性をカーネルに乗算するならば"mul"、加算なら"add"の2種類のみ指定可能(default="mul")
***
- lossw
    - String : 3つの損失項（観測部、欠損部、欠損部周囲1ピクセル）の重み(default="1,6,0.1")
***
- existOnly
    - None : 損失計算時にデータが存在する範囲のみを使用
***
- epochs
    - Int : 最大エポック数(default=400)
***
- siteLayers
    - String : 位置特性を導入する層をカンマ区切りで指定（default="1"）
***
- learnSitePConv
    - None : 入力層に位置特性を導入するPConvのモデル（特徴マップに位置特性を加算・乗算）
***
- learnSitePKConv
    - None : 入力層に位置特性を導入するPKConvのモデル（カーネルに位置特性を加算・乗算）
***
- branchlSitePConv
    - None : 中間層に位置特性を導入するために、チャネル数を合わせる位置特性CNNがついたPConvのモデル（特徴マップに位置特性を加算・乗算）
***
- branchlSitePKConv
    - None : 中間層に位置特性を導入するために、チャネル数を合わせる位置特性CNNがついたPKConvのモデル（カーネルに対する加算・乗算を行う）
***
- phase
    - int : 学習の段階を指定(default=0)
        - phase=0：全てのモデルを同時に学習
        - phase=1：pconvUNetのみを学習
        - phase=2：phase=1をロードして位置特性・位置特性Convのみを学習
        - phase=3：phase=2をロードしてpconvUNetのみ学習）
***
- nonNegSConv
    - None : 位置特性のConv層に非負のカーネルを使用
***
- patience
    - int : EarlyStoppingのpatienceを指定
***
***
## **既存法を実行**
- Partial Convolution
    - 値のない領域の損失を考慮しない
~~~bash
python main.py pconv_quake -dataset quakeData-balance -existOnly
~~~
***
## **提案法を実行**
### 設定例①
- 提案法
    - 入力層に位置特性を導入
    - 特徴マップに乗算
    - 値のない領域の損失を考慮しない
~~~bash
python main.py learnSitePConv_mul_quake -learnSitePConv -dataset quakeData-balance -ope mul -existOnly
~~~
***
### 設定例②
- 提案法
    - 入力層に位置特性を導入
    - 特徴マップに加算
    - 値のない領域の損失を考慮しない
~~~bash
python main.py learnSitePConv_add1_quake -learnSitePConv -dataset quakeData-balance -ope add -existOnly
~~~
***
### 設定例③
- 提案法
    - 入力層に位置特性を導入
    - カーネルに加算
    - 値のない領域の損失を考慮しない
~~~bash
python main.py learnSitePKConv_add2_quake -learnSitePKConv -dataset quakeData-balance -ope add -existOnly
~~~
***
### 設定例④
- 提案法
    - 中間層(3層目)に位置特性を導入
    - チャネル数を合わせるために位置特性CNN（branch）追加
    - 特徴マップに加算
    - 値のない領域の損失を考慮しない
~~~bash
python main.py branchlSitePConv_add_quake -branchlSitePConv -dataset quakeData-balance -siteLayers 3
~~~
***
### 設定例⑤
- 提案法
    - UNetと位置特性の学習を別々で行う
    - 値のない領域の損失を考慮しない
    - UNetの事前学習(phase=1)
    - 位置特性を学習(phase=2)
    - UNetを再学習　(phase=3)
~~~bash
dataset="-dataset quakeData-balance"
pretrainExp="phase1_pkconv_branchlsiteF-nonNeg-l3"
config="-ope mul -branchlSitePKConv -siteLayers 3 -nonNegSConv"
python main.py ${pretrainExp} ${dataset} ${config} -existOnly -phase 1

exp="phase2-3_pkconv_mul_branchlsiteF-nonNeg-l3"
python main.py ${exp} ${dataset} ${config} -pt ${pretrainExp} -existOnly -phase 2
python main.py ${exp} ${dataset} ${config} -pt ${exp} -existOnly -phase 3
~~~
***
## **交差検証を実行**
交差検証をする際には実験名の末尾を _cv${i} でそろえる必要がある 
- 既存法の交差検証
~~~bash
for i in `seq 0 4`
do
    python main.py pconv_quake_cv${i} -dataset quakeData-h01h02h06h07-crossVaridation${i}
done
~~~


# **作成済みデータセット**
すべてのデータセットは`data`ディレクトリ内に存在します。

***
## 地震動データ
震源・震源域など均等に分割したデータ
> quakeData-balance
***
## 地震動データ（交差検証用）
交差検証用に分割したデータ0~9まで存在
> quakeData-crossVaridation0
***
## 矩形データ(横縞)
> stripe-rectData
***
## 位置特性画像について
`data/siteImage/`ディレクトリに保存されている。学習やテストで使用するには、このディレクトリに入れておく必要がある。
> nagayoshi@deep2:~/research/PositionalKernelConvolution/data/siteImage