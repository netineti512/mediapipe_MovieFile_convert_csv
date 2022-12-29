今回はPythonのMediapipeを使い、動画ファイルの手を認識し、手の座標データをCSVファイルに保存する方法を紹介します。
動画ファイルを画像ファイルに変換してから、Mediapipeを使います。
***

# ライブラリのインストール
```
pip install opencv-python
pip install glob
pip install mediapipe
```

# 動画ファイルを画像ファイルに変換
```convert_movie.py
import cv2
import os

def save_all_frames(video_path, dir_path, basename, ext='jpg'):
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        return

    os.makedirs(dir_path, exist_ok=True)
    base_path = os.path.join(dir_path, basename)

    digit = len(str(int(cap.get(cv2.CAP_PROP_FRAME_COUNT))))

    n = 0

    while True:
        ret, frame = cap.read()
        if ret:
            cv2.imwrite('{}_{}.{}'.format(base_path, str(n).zfill(digit), ext), frame)
            n += 1
        else:
            return

save_all_frames('./video/sample_video.mp4', './image', 'sample_video_img')

save_all_frames('data/temp./video/sample_video.mp4', 'image/result_png', 'sample_video_img', 'png')
```
>Mediapipeを使い、画像ファイルの座標をCSVに出力
```python
import cv2
import glob
import os
import csv
import mediapipe as mp

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

def fields_name():
    # CSVのヘッダを準備
    fields = []
    fields.append('file_name')
    for i in range(21):
        fields.append(str(i)+'_x')
        fields.append(str(i)+'_y')
        fields.append(str(i)+'_z')
    return fields

if __name__ == '__main__':
    # 元の画像ファイルの保存先を準備
    resource_dir = r'./image'
    # 対象画像の一覧を取得
    file_list = glob.glob(os.path.join(resource_dir, "*.jpg"))

    # 保存先の用意
    save_csv_dir = './csv/result'
    os.makedirs(save_csv_dir, exist_ok=True)
    save_csv_name = 'landmark.csv'
    save_image_dir = './image/image_landmark'
    os.makedirs(save_image_dir, exist_ok=True)

    with mp_hands.Hands(static_image_mode=True,
            max_num_hands=1, # 検出する手の数（最大2まで）
            min_detection_confidence=0.5) as hands, \
        open(os.path.join(save_csv_dir, save_csv_name), 
            'w', encoding='utf-8', newline="") as f:

        # csv writer の用意
        writer = csv.DictWriter(f, fieldnames=fields_name())
        writer.writeheader()

        for file_path in file_list:
            # 画像の読み込み
            image = cv2.imread(file_path)

            # 鏡写しの状態で処理を行うため反転
            image = cv2.flip(image, 1)

            # OpenCVとMediaPipeでRGBの並びが違うため、
            # 処理前に変換しておく。
            # CV2:BGR → MediaPipe:RGB
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False

            # 推論処理
            results = hands.process(image)

            # 前処理の変換を戻しておく。
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            if not results.multi_hand_landmarks:
                # 検出できなかった場合はcontinue
                continue

            # ランドマークの座標情報
            landmarks = results.multi_hand_landmarks[0]

            # CSVに書き込み
            record = {}
            record["file_name"] = os.path.basename(file_path)
            for i, landmark in enumerate(landmarks.landmark):
                record[str(i) + '_x'] = landmark.x
                record[str(i) + '_y'] = landmark.y
                record[str(i) + '_z'] = landmark.z
            writer.writerow(record)

            # 元画像上にランドマークを描画
            mp_drawing.draw_landmarks(
                image,
                landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style())
            # 画像を保存
            cv2.imwrite(
                os.path.join(save_image_dir, os.path.basename(file_path)),
                cv2.flip(image, 1))
```
終わり
***
>他の記事
・[[2022年]SVMの学習セットのフォーマット(CSVファイル)](https://qiita.com/netineti512/items/2596d4fbdebd700a7aa0)
・[[2022年]MediaPipeで取得した座標データの統一化の方法[CSVファイル]](https://qiita.com/netineti512/items/fd5929361a6fdb8f629b)
・[[2022年]PythonのMediaPipeを使い、動画ファイルの手を認識し、CSVファイルに座標データを保存する方法](https://qiita.com/netineti512/items/b79ff4f878c7795b6b91)
・[[2022年]PythonのOpenCVを使って動画を録画する方法](https://qiita.com/netineti512/items/57b532d5acd29ab36e67)
***
