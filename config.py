#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
設定管理モジュール
作成日: 2025-07-31
作成者: AI Assistant
"""

import os
from pathlib import Path

# 設定値の定義
# 出力ディレクトリ設定
OUTPUT_DIR = 'analysis_results'

# 動画ファイルリストのパス
MOV_FILES_PATH = 'mov_files.txt'

# プレビュー表示の有無
SHOW_PREVIEW = False

# 進捗表示間隔（フレーム数）
PROGRESS_INTERVAL = 30

# サングラス検出関連の設定
# 輝度閾値（この値より低い輝度の場合、サングラス装着と判定）
SUNGLASSES_BRIGHTNESS_THRESHOLD = 50

# 目のROIスケールファクター（目と目の距離に対するROIの相対サイズ）
EYE_ROI_SCALE_FACTOR = 1.5

# 目のROIのリサイズ目標サイズ（幅,高さ）
EYE_ROI_TARGET_SIZE = (50, 50)

# 複数フレーム分析時のフレーム数
MULTI_FRAME_COUNT = 5

# 設定ファイルのパス
CONFIG_FILE = 'config.ini'

def load_config_from_file():
    """設定ファイルから設定を読み込む（オプション）"""
    if os.path.exists(CONFIG_FILE):
        try:
            import configparser
            config = configparser.ConfigParser()
            config.read(CONFIG_FILE, encoding='utf-8')
            
            # 設定値を更新
            global OUTPUT_DIR, MOV_FILES_PATH, SHOW_PREVIEW, PROGRESS_INTERVAL
            global SUNGLASSES_BRIGHTNESS_THRESHOLD, EYE_ROI_SCALE_FACTOR
            global EYE_ROI_TARGET_SIZE, MULTI_FRAME_COUNT
            
            if 'DEFAULT' in config:
                OUTPUT_DIR = config.get('DEFAULT', 'OUTPUT_DIR', fallback=OUTPUT_DIR)
                MOV_FILES_PATH = config.get('DEFAULT', 'MOV_FILES_PATH', fallback=MOV_FILES_PATH)
                SHOW_PREVIEW = config.getboolean('DEFAULT', 'SHOW_PREVIEW', fallback=SHOW_PREVIEW)
                PROGRESS_INTERVAL = config.getint('DEFAULT', 'PROGRESS_INTERVAL', fallback=PROGRESS_INTERVAL)
            
            if 'SUNGLASSES_DETECTION' in config:
                SUNGLASSES_BRIGHTNESS_THRESHOLD = config.getint('SUNGLASSES_DETECTION', 'BRIGHTNESS_THRESHOLD', fallback=SUNGLASSES_BRIGHTNESS_THRESHOLD)
                EYE_ROI_SCALE_FACTOR = config.getfloat('SUNGLASSES_DETECTION', 'SCALE_FACTOR', fallback=EYE_ROI_SCALE_FACTOR)
                MULTI_FRAME_COUNT = config.getint('SUNGLASSES_DETECTION', 'MULTI_FRAME_COUNT', fallback=MULTI_FRAME_COUNT)
                
                # ターゲットサイズの処理
                target_size_str = config.get('SUNGLASSES_DETECTION', 'TARGET_SIZE', fallback=f"{EYE_ROI_TARGET_SIZE[0]},{EYE_ROI_TARGET_SIZE[1]}")
                try:
                    width, height = map(int, target_size_str.split(','))
                    EYE_ROI_TARGET_SIZE = (width, height)
                except:
                    pass  # デフォルト値を使用
            
            print(f"設定ファイル {CONFIG_FILE} から設定を読み込みました")
            
        except Exception as e:
            print(f"設定ファイルの読み込みに失敗しました: {e}")
            print("デフォルト設定を使用します")

def get_output_dir():
    """CSVファイルの出力先ディレクトリを取得"""
    return OUTPUT_DIR

def get_mov_files_path():
    """動画ファイルリストのパスを取得"""
    return MOV_FILES_PATH

def get_show_preview():
    """プレビュー表示の有無を取得"""
    return SHOW_PREVIEW

def get_progress_interval():
    """進捗表示間隔を取得"""
    return PROGRESS_INTERVAL

def get_sunglasses_brightness_threshold():
    """サングラス検出の輝度閾値を取得"""
    return SUNGLASSES_BRIGHTNESS_THRESHOLD

def get_eye_roi_scale_factor():
    """目のROIスケールファクターを取得"""
    return EYE_ROI_SCALE_FACTOR

def get_eye_roi_target_size():
    """目のROIのリサイズ目標サイズを取得"""
    return EYE_ROI_TARGET_SIZE

def get_multi_frame_count():
    """複数フレーム分析時のフレーム数を取得"""
    return MULTI_FRAME_COUNT

def create_config_file():
    """設定ファイルのテンプレートを作成"""
    config_content = """# 設定ファイル
# このファイルを編集して設定を変更できます

[DEFAULT]
# 出力ディレクトリ設定
OUTPUT_DIR = analysis_results

# 動画ファイルリストのパス
MOV_FILES_PATH = mov_files.txt

# プレビュー表示の有無
SHOW_PREVIEW = false

# 進捗表示間隔（フレーム数）
PROGRESS_INTERVAL = 30

[SUNGLASSES_DETECTION]
# 輝度閾値（この値より低い輝度の場合、サングラス装着と判定）
# 低い値（例: 30）: より厳密なサングラス検出
# 高い値（例: 70）: より緩やかなサングラス検出
# 推奨値: 50（環境光に応じて調整）
BRIGHTNESS_THRESHOLD = 50

# 目のROIスケールファクター（目と目の距離に対するROIの相対サイズ）
# 低い値（例: 1.2）: より小さなROI（高解像度向け）
# 高い値（例: 1.8）: より大きなROI（低解像度向け）
# 推奨値: 1.5
SCALE_FACTOR = 1.5

# 目のROIのリサイズ目標サイズ（幅,高さ）
# 小さいサイズ（例: 30,30）: 高速処理、低精度
# 大きいサイズ（例: 80,80）: 低速処理、高精度
# 推奨値: 50,50
TARGET_SIZE = 50,50

# 複数フレーム分析時のフレーム数
# 小さい値: 高速処理、ノイズに敏感
# 大きい値: 低速処理、ノイズに強い
# 推奨値: 5
MULTI_FRAME_COUNT = 5
"""
    
    try:
        with open(CONFIG_FILE, 'w', encoding='utf-8') as f:
            f.write(config_content)
        print(f"設定ファイル {CONFIG_FILE} を作成しました")
        print("このファイルを編集して設定を変更できます")
    except Exception as e:
        print(f"設定ファイルの作成に失敗しました: {e}")

# 初期化時に設定ファイルを読み込む
if __name__ == "__main__":
    # 設定ファイルが存在しない場合は作成
    if not os.path.exists(CONFIG_FILE):
        create_config_file()
    else:
        # 既存の設定ファイルを読み込み
        load_config_from_file()
else:
    # モジュールとしてインポートされた場合も設定ファイルを読み込み
    load_config_from_file() 