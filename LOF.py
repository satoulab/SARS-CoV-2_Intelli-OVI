# -*- coding: utf-8 -*

# ------------------------------------------------------------------------
# LOFを計算するプログラム
# ブートストラップサンプルで多変量正規分布を推定し乱数サンプルを生成する
#  [実行方法]
#    > python LOF.py
#  [注意]
#    説明変数の個数（プローブの個数）をRandomSample()で設定する必要がある
#    目的変数（クラス）は０から始める必要がある
# ------------------------------------------------------------------------

import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import LocalOutlierFactor
from sklearn.metrics import accuracy_score
from sklearn.manifold import MDS
import matplotlib.pyplot as plt

def main():
    # データの読込み
    Y,X = LoadData()
    print("データの読込みが終了しました")
    # データの正規化
    NorX = Normalize(X)
    print("データの正規化が終了しました")
    # 多変量正規分布から乱数サンプルを生成する
    trainX = RandomSample(Y, NorX)
    print("多変量正規分布から乱数サンプルを生成しました")
    # LOFの計算
    Score = LOF(Y, NorX, trainX)
    print("LOFの計算を終了しました")
    # MDSの空間にLOF値を表示する
    ShowLOF(Y, NorX, Score)
    print("MDSの空間にLOF値を表示しました")

# ----------------------------------------------------------
#    データを読込むための関数
#  [RETURN]
#    Y : 目的変数（クラス情報）
#    X : 説明変数（プローブの値）
# ----------------------------------------------------------
def LoadData():
   df = pd.read_csv('InputDataMDS.csv', header=None, skiprows=1) # 1行目をヘッダとして読み飛ばしてデータを読込む
   print(df)
   Y = df.iloc[:,0]                    # １行目のデータ（クラス情報）を抽出
   X = df.drop(df.columns[[0]],axis=1) # １行目のデータ（クラス情報）を削除する
   return(Y,X)

# ----------------------------------------------------------
#   データを正規化するための関数
#   各症例をベクトルの大きさで正規化する項目を付け加えた
# [INPUT]
#   X : 説明変数（プローブの値）
# [RETURN]
#   NorX : 正規化されたデータ
# ----------------------------------------------------------
def Normalize(X):
    mm = preprocessing.MinMaxScaler() # 最小値を０，最大値を１に正規化するインスタンスを生成
    tmp = mm.fit_transform(X)  # 正規化する
    norm = np.linalg.norm(tmp, ord=2, axis=1) # ord＝２：L2ノルムを求める．axis=1: 行に関して求める
    tmp2 = tmp / norm[:, np.newaxis]          # ベクトルの大きさで正規化  
    NorX = pd.DataFrame(tmp2)   # numpy配列をデータフレームに変換する
    return (NorX)

# ----------------------------------------------------------------------
#    多変量正規分布から乱数サンプルを生成するためのプログラム
#  [INPUT]
#    Y : 目的変数（クラス情報）
#    X : 説明変数（プローブの値） 
#  [RETRUN]
#    data : 生成された乱数サンプル 
# ----------------------------------------------------------------------
def RandomSample(Y, X):
    FeaNum = 22        # 説明変数の数（プローブの数）を設定する
    SampleNum = 2000   # ブートストラップサンプルの数を設定する
    data = np.empty((0,FeaNum))
    for n in range(max(Y)):      # nは0からYの最大クラス-1まで，(注意)クラスは０から始める必要がある
        # データの取り出し
        tmp = X[Y==n]            # クラスがnのときのデータを取り出す
        np_tmp = tmp.to_numpy()  # PandasのDataFrameをNumpy配列に変換する
        # ブートストラップ標本を生成する
        Num = np.random.choice(len(np_tmp), SampleNum, replace=True)
        np_tmp2 = np_tmp[Num,:]
        # 平均値の計算
        mean = np.mean(np_tmp2, axis=0)
        # 分散共分散の計算
        cov = np.cov(np_tmp2, rowvar=False)
        # 乱数サンプルの生成
        np_tmp3 = np.random.multivariate_normal(mean, cov, size=SampleNum)
        data = np.vstack([data, np_tmp3])
    print(np_tmp3.shape)
    print(data.shape)
    print(data)
    return(data)

# ----------------------------------------------------------------------
#    Local Outlier Factorを計算するための関数
#  [INPUT]
#    Y : 目的変数（クラス情報）
#    X : 説明変数（プローブの値）
#    X_train : 乱数サンプルとして生成された訓練データ
#  [RETURN]
#    Score : LOFの値
#  [注意]
#    外れ値検出に利用する場合，n_neighborsは大きめに設定する方がうまく機能する
# ----------------------------------------------------------------------
def LOF(Y, X, X_train):
    # モデルの学習
    clf = LocalOutlierFactor(n_neighbors=10, novelty=True)
    clf.fit(X_train) 
    # LOFの計算
    Lof = []
    for n in range(len(X)):
        # テストデータの作成
        X_test = X.iloc[n,:]
        X_test = pd.DataFrame(X_test)  # SeriesからDataFrameに変換
        X_test = X_test.T              # 行と列を交換（DataFrameに変換した際の出力方向をもとに戻す）  
        # 推論
        Y_Lof = clf.predict(X_test)          # 外れ点を-1，それ以外を１で出力，閾値はscore_samplesで確認することができる
        Y_Lof2 = clf.score_samples(X_test)   # LOFの値を出力
        Lof.append(Y_Lof2[0])
    # LOF値の出力
    Score = pd.DataFrame(Lof)
    tmp = pd.concat([Y, Score], axis=1)
    tmp.columns=["class","LOF value"]
    tmp.to_excel('Output_LOF.xlsx')
    # 戻り値
    return(Score)

# ----------------------------------------------------------------------
#    Local Outlier Factorを計算するための関数
#  [INPUT]
#    Y     : 目的変数（クラス情報）
#    NorX  : 説明変数（正規化されたプローブの値）
#    Score : LOFの値 
# ----------------------------------------------------------------------
def ShowLOF(Y, NorX, Score):
    mds = MDS(n_components=2, metric=True, dissimilarity='euclidean', random_state=0)
    feature = mds.fit_transform(NorX)
    # 可視化
    plt.figure(figsize=(7, 7))
    # MDSの空間に各症例の点を配置する
    for i in range(len(feature)):
        plt.scatter(feature[i,0], feature[i,1], color="k", s=3)
    # LOF値を円で表示する
    radius = (Score.max()-Score) / (Score.max()-Score.min()) # LOF値を０から１の値に線形変換
    plt.scatter(feature[:,0], feature[:,1], s=1000*radius, edgecolors="r", facecolors="none") 
    plt.xlim((-1.25, 1.25))
    plt.ylim((-1.25, 1.25))
    plt.show()
    plt.savefig('Lof')

if __name__ == '__main__':
    main()
