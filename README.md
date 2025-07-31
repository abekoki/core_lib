# 顔検出・顔向き推定システム

## 概要
このシステムは、MediaPipeを使用して顔のランドマークを検出し、開瞼度と顔の向き（yaw、pitch）を推定するPythonアプリケーションです。

## 機能
- **開瞼度検出**: 左右の目の開瞼度を鼻の長さで正規化して計算
- **顔向き推定**: 三角形の重心と鼻先の位置関係からyaw（左右）とpitch（上下）を推定
- **リアルタイム処理**: ウェブカメラからのリアルタイム処理
- **動画分析**: 動画ファイルからの一括分析とCSV出力

## ファイル構成
- `eye_detection.py`: メインの顔検出・推定モジュール
- `video_analysis.py`: 動画ファイル分析スクリプト
- `mov_files.txt`: 分析対象動画ファイルのリスト
- `run_analysis.bat`: 動画分析実行用バッチファイル

## 使用方法

### 1. リアルタイム処理（ウェブカメラ）
```bash
python eye_detection.py
```

### 2. 動画ファイル分析
```bash
python video_analysis.py
```
または
```bash
run_analysis.bat
```

## 出力CSVファイルの列
- `frame`: フレーム番号
- `leye_openness`: 左目の開瞼度
- `reye_openness`: 右目の開瞼度
- `confidence`: 信頼度
- `face_yaw`: 顔の左右の向き（度）
- `face_pitch`: 顔の上下の向き（度）
- `nose_length`: 鼻の長さ（ピクセル）
- `normal_vector`: 三角形の法線ベクトル
- `distance_x`: X軸方向の距離
- `distance_y`: Y軸方向の距離
- `arrow_length`: Yaw矢印の長さ
- `pitch_arrow_length`: Pitch矢印の長さ
- `triangle_area`: 三角形の面積
- `eye_distance`: 両目の距離

## 技術仕様
- **顔向き推定**: 三角形の重心と鼻先の距離から計算
- **方向性**: 
  - Yaw: 左向き=負、右向き=正
  - Pitch: 下向き=負、上向き=正
- **範囲制限**: ±45度
- **信頼度計算**: 三角形の面積を正規化

## 必要なライブラリ
- OpenCV (cv2)
- MediaPipe
- NumPy
- SciPy

## インストール
```bash
pip install opencv-python mediapipe numpy scipy
```

## 設定

### 環境変数ファイル（.env）
以下の設定を`.env`ファイルで指定できます：

```env
# 出力ディレクトリを指定（デフォルト: analysis_results）
OUTPUT_DIR=analysis_results

# 動画ファイルリストのパスを指定（デフォルト: mov_files.txt）
MOV_FILES_PATH=mov_files.txt

# プレビュー表示の有無（true/false、デフォルト: false）
SHOW_PREVIEW=false

# 進捗表示間隔（フレーム数、デフォルト: 30）
PROGRESS_INTERVAL=30
```

### 設定項目の説明
- **OUTPUT_DIR**: CSVファイルの出力先ディレクトリ
- **MOV_FILES_PATH**: 分析対象動画ファイルのリストファイル
- **SHOW_PREVIEW**: 分析中のプレビュー表示の有無
- **PROGRESS_INTERVAL**: 進捗表示の間隔（フレーム数）

## 注意事項
- 動画ファイルは`mov_files.txt`に1行1ファイルで記載してください
- 分析結果は`analysis_results`ディレクトリ（または.envで指定したディレクトリ）に保存されます
- 顔が検出されないフレームは全て0.0で出力されます
- `.env`ファイルが存在しない場合はデフォルト設定が使用されます
