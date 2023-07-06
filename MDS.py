# -*- coding: utf-8 -*

# ----------------------------------------------------------------
# 多次元尺度構成法を用いてデータを可視化するためのプログラム
# [実行方法]
# > python MDS.py
# [注意]
#  (1) 入力データファイル名をLoadData()関数で設定する必要がある
#  (2) 最大で12個の変異ウィルスを表示できるように設定している
#      増やす場合には，MDS()関数でcolorを定義する必要がある
# [修正履歴]
#  　ヘッダを含んだ入力データが読み込めるようにLoadData()を修正した
# ---------------------------------------------------------------

from pickle import TRUE
import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.manifold import MDS
import matplotlib.pyplot as plt

def main():
    # データの読込み
    Y,X = LoadData()
    print("Data loading finished")
    # データの正規化
    NorX = Normalize(X)
    print("data normalization finished")
    # MDSによる可視化
    myMDS(Y, NorX)
    print("Visualization by MDS finished")

# ----------------------------------------------------------
#    データを読込むための関数
#  [RETURN]
#    Y : 目的変数（クラス情報）
#    X : 説明変数（プローブの値）
# ----------------------------------------------------------
def LoadData():
   df = pd.read_csv('InputDataMDS.csv', header=None, skiprows=1) # 1行目をヘッダとして読み飛ばしてデータを読込む
   print(df)
   Y = df.iloc[:,0]                    # １行目のデータを抽出
   X = df.drop(df.columns[[0]],axis=1) # １行目のデータを削除する
   return(Y,X)

# ----------------------------------------------------------
#   データを正規化するための関数
#   各症例のベクトルの大きさで正規化する項目を付け加えた
# [INPUT]
#   X : データ（説明変数）
# [RETURN]
#   NorX : 正規化されたデータ
# ----------------------------------------------------------
def Normalize(X):
    mm = preprocessing.MinMaxScaler() # インスタンスの生成
    tmp = mm.fit_transform(X)  # 最小値を０，最大値を１に正規化する
    norm = np.linalg.norm(tmp, ord=2, axis=1) # ord＝２：L2ノルムを求める．axis=1: 行に関して求める
    tmp2 = tmp / norm[:, np.newaxis]          # ベクトルの大きさで正規化  
    NorX = pd.DataFrame(tmp2)   # numpy配列をデータフレームに変換する
    return (NorX)

# ----------------------------------------------------------
#   MDSを行うための関数
# [INPUT]
#   Y    : 目的変数（クラス情報）
#   NorX : 説明変数 （正規化されたデータ）
# ----------------------------------------------------------
def myMDS(Y, NorX):
    mds = MDS(n_components=2, metric=TRUE, dissimilarity='euclidean', random_state=0)
    feature = mds.fit_transform(NorX)
    # 可視化
    plt.figure(figsize=(7, 7))
    colors =["b", "m", "g", "y", "coral", "r", "c", "k", "fuchsia", "gray", "brown", "moccasin"]
    for i in range(len(feature)):
        plt.scatter(feature[i,0], feature[i,1], color=colors[Y[i]])
        # plt.annotate(i+1, (feature[i,0], feature[i,1]))
    # plt.xlabel('MDS Dimension1')
    # plt.ylabel('MDS Dimension2')
    plt.xlim((-1.25, 1.25)) # X軸の描画範囲を指定
    plt.ylim((-1.25, 1.25)) # Y軸の描画範囲を指定
    plt.show()
    plt.savefig('MDS')
    tmp = pd.DataFrame([feature[:,0], feature[:,1]])
    tmp = tmp.T
    tmp.columns=["Axis 1","Axis 2"]
    # MDSの座標点を出力する
    tmp.to_excel('Output_MDS.xlsx')

if __name__ == '__main__':
    main()