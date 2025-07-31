# MediaPipeを使用した両目のランドマーク検出・開瞼度検出・顔の向き推定の手順書

## 1. 目的
この手順書は、PythonとMediaPipeを使用して、ウェブカメラまたは動画から顔のランドマークを検出し、左目および右目の開瞼度（EAR）、検出信頼度（間接的）、および顔の向き（yaw, pitch, roll）を推定するプログラムのセットアップと実行方法を説明します。

## 2. 必要条件
- **OS**: Windows, macOS, Linux
- **Python**: 3.7以上
- **ハードウェア**: ウェブカメラ（リアルタイム処理の場合）または動画ファイル
- **インターネット接続**: ライブラリのインストール用

## 3. 環境構築手順
### 3.1 Pythonのインストール
1. [Python公式サイト](https://www.python.org/)からPython 3.7以上をインストール。
2. インストール時に「Add Python to PATH」をチェック。
3. コマンドプロンプトまたはターミナルで以下を実行し、Pythonが正しくインストールされたことを確認：
   ```bash
   python --version
   ```

### 3.2 仮想環境の作成（推奨）
1. プロジェクトフォルダを作成（例: `eye_detection_project`）。
2. 仮想環境を作成・有効化：
   ```bash
   python -m venv venv
   source venv/bin/activate  # Linux/macOS
   venv\Scripts\activate     # Windows
   ```

### 3.3 必要なライブラリのインストール
以下のコマンドで必要なライブラリをインストール：
```bash
pip install opencv-python mediapipe numpy scipy
```

- **opencv-python**: 画像処理およびウェブカメラ入力用。
- **mediapipe**: 顔ランドマーク検出用。
- **numpy**: 数値計算用。
- **scipy**: 距離計算（EAR計算）用。

### 3.4 サンプルコードの保存
1. 上記のサンプルコードを`eye_detection.py`としてプロジェクトフォルダに保存。
2. コード内で`cv2.VideoCapture(0)`を動画ファイルに変更する場合は、以下のように修正：
   ```python
   cap = cv2.VideoCapture("path/to/your/video.mp4")
   ```

## 4. プログラムの実行
1. 仮想環境が有効な状態で、プロジェクトフォルダに移動：
   ```bash
   cd eye_detection_project
   ```
2. プログラムを実行：
   ```bash
   python eye_detection.py
   ```
3. ウェブカメラが起動し、顔が検出されると以下が表示される：
   - 両目のランドマーク（視覚的に描画）
   - 左目および右目のEAR（開瞼度）
   - 信頼度（ランドマークのvisibilityの平均）
   - 顔の向き（yaw, pitch, roll）
4. `q`キーを押してプログラムを終了。

## 5. 出力の詳細
- **目のランドマーク**: 
  - 左目: インデックス33, 160, 158, 133, 153, 144
  - 右目: インデックス263, 387, 385, 362, 380, 373
- **開瞼度（EAR）**: 左目と右目の縦横比を計算。値が小さいほど目が閉じている。
- **信頼度**: ランドマークの`visibility`属性の平均（0～1）。1に近いほど信頼性が高い。
- **顔の向き**: PnPアルゴリズムでyaw（左右）、pitch（上下）、roll（回転）を計算（度数表示）。

## 6. カスタマイズのポイント
- **結果の保存**:
  EARや顔の向きをCSVファイルに保存する場合は、以下を追加：
  ```python
  import csv
  with open("output.csv", "a") as f:
      writer = csv.writer(f)
      writer.writerow([left_ear, right_ear, confidence, yaw, pitch, roll])
  ```
- **信頼度の改善**:
  フレーム間のランドマーク変動を計算し、安定性を信頼度の指標として追加可能。
- **動画入力**:
  動画ファイルを使用する場合は、ファイルパスを指定。
- **追加の可視化**:
  両目のEARをグラフィカルに表示（例: ゲージやグラフ）する場合は、OpenCVの描画関数を活用。

## 7. トラブルシューティング
- **ウェブカメラが動作しない**:
  - カメラが接続されているか確認。
  - `cv2.VideoCapture(1)`など、別のインデックスを試す。
- **MediaPipeが顔を検出しない**:
  - 照明を改善し、顔が正面を向いていることを確認。
  - `min_detection_confidence`を0.3などに下げる。
- **エラー: モジュールが見つからない**:
  - ライブラリが正しくインストールされているか確認（`pip list`）。
  - 仮想環境が有効か確認。

## 8. 参考資料
- [MediaPipe Face Mesh公式ドキュメント](https://google.github.io/mediapipe/solutions/face_mesh.html)
- [OpenCVドキュメント](https://docs.opencv.org/)
- [SciPyドキュメント](https://docs.scipy.org/doc/scipy/reference/)