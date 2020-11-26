#
# カメラ画像をキャプチャして、グーかチョキかパーをあてる
#
import numpy as np
from keras.models import Sequential, model_from_json
from keras.layers.core import Dense, Dropout
from keras.optimizers import RMSprop
from keras.datasets import mnist
from keras.utils import np_utils, to_categorical
from keras.preprocessing.image import load_img, img_to_array
import matplotlib.pyplot as plt
import cv2
import os

img_width = 100 # 入力画像の幅
img_height = 100 # 入力画像の高さ

img_ch = 3 # 3ch画像（RGB）

# 入力データ数
num_data = 1

# データ格納用のディレクトリパス
SAVE_DATA_DIR_PATH = "data\\"

# ラベル
labels =['グー', 'チョキ', 'パー']

# 保存したモデル構造の読み込み
model = model_from_json(open(SAVE_DATA_DIR_PATH + "model.json", 'r').read())

# 保存した学習済みの重みを読み込み
model.load_weights(SAVE_DATA_DIR_PATH + "weight.hdf5")

cap = cv2.VideoCapture(0) # 任意のカメラ番号に変更する

while True:
    ret, frame = cap.read()
    cv2.imshow("camera", frame)

    k = cv2.waitKey(1)&0xff # キー入力を待つ
    if k == ord('p'):
        # 「p」キーで画像を保存
        path = "./data/" + "Hand.jpg"
        cv2.imwrite(path, frame) # ファイル保存

        #cv2.imshow(path, frame) # キャプチャした画像を表示

        # 画像の読み込み（32×32にリサイズ）
        # 正規化, 4次元配列に変換（モデルの入力が4次元なので合わせる）
        img = load_img(SAVE_DATA_DIR_PATH + "Hand.jpg", target_size=(img_width, img_height))
        img = img_to_array(img) 
        img = img.astype('float32')/255.0
        img = np.array([img])

        # 分類機に入力データを与えて予測（出力：各クラスの予想確率）
        y_pred = model.predict(img)

        # 最も確率の高い要素番号
        number_pred = np.argmax(y_pred) 

        # 予測結果の表示
        print("y_pred:", y_pred)  # 出力値
        print("number_pred:", number_pred)  # 最も確率の高い要素番号
        print('label_pred：', labels[int(number_pred)]) # 予想ラベル（最も確率の高い要素）
        
    elif k == ord('q'):
        # 「q」キーが押されたら終了する
        break

# キャプチャをリリースして、ウィンドウをすべて閉じる
cap.release()
cv2.destroyAllWindows()
