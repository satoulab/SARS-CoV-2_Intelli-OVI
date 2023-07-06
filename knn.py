# -*- coding: utf-8 -*

# --------------------------------------------------------------------
#  kNNを用いて多クラス分類を行うプログラム
#  学習と評価には，Leave-one-out法を用いた
#  [実行方法]
#    > python knn.py
#  [注意]
#    LoadData()で読込むファイル名を指定する必要がある
#    kNN()で何点までの近似を考慮するか n_neighbors= を設定する必要がある
# --------------------------------------------------------------------

import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import LocalOutlierFactor
from sklearn.metrics import accuracy_score

def main():
    # データの読込み
    Y,X = LoadData()
    # データの正規化
    NorX = Normalize(X)
    # kNNによる識別
    kNN(Y, NorX)

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
#   各症例をベクトルの大きさで正規化する
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

# ----------------------------------------------------------
#    kNNを用いて多クラス分類するための関数
#  [INPUT]
#    Y：目的変数 クラス情報
#    X：説明変数 正規化されたデータ
# ----------------------------------------------------------
def kNN(Y, X):
    # インスタンス
    knn = KNeighborsClassifier(n_neighbors=3) # 何点までの近似を考慮するか
    # Leave-one-out法による学習とテスト
    Pred = []
    for n in range(len(X)):
        # 学習データの作成（n行目のデータを削除）
        X_train = X.drop(n)  # n行目のデータを削除
        Y_train = Y.drop(n)  # n行目のデータを削除
        # テストデータの作成（n番目のデータをテスト）
        X_test = X.iloc[n,:]
        X_test = pd.DataFrame(X_test)  # SeriesからDataFrameに変換
        X_test = X_test.T              # 行と列を交換
        # モデル学習
        knn.fit(X_train, np.ravel(Y_train))
        # 推論
        Y_pred = knn.predict(X_test)
        Pred.append(Y_pred[0])
    # 性能評価
    print("正解率(%): " + str(round(accuracy_score(Y,Pred),3)*100))
    # 識別結果をファイルに保存
    Pred = pd.DataFrame(Pred)
    tmp = pd.concat([Y, Pred], axis=1)
    tmp.columns=["true class","estimated class"]
    tmp.to_excel('Output_kNN.xlsx')

if __name__ == '__main__':
    main()
