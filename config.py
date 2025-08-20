#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
設定管理モジュール
作成日: 2025-07-31
更新日: 2025-08-20
作成者: AI Assistant
"""

import os
import configparser
from pathlib import Path

# .envファイルの読み込み
def load_env_file():
    """環境変数ファイル(.env)を読み込み"""
    env_file = Path('.env')
    if env_file.exists():
        with open(env_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    key, value = line.split('=', 1)
                    os.environ[key.strip()] = value.strip()

# config.iniファイルの読み込み
def load_config_file():
    """設定ファイル(config.ini)を読み込み"""
    config_file = Path('config.ini')
    if config_file.exists():
        try:
            config = configparser.ConfigParser()
            config.read(config_file, encoding='utf-8')
            
            if 'DEFAULT' in config:
                for key, value in config['DEFAULT'].items():
                    os.environ[key] = value
                    
            print(f"設定ファイル {config_file} から設定を読み込みました")
            
        except Exception as e:
            print(f"設定ファイルの読み込みに失敗しました: {e}")
            print("デフォルト設定を使用します")

# 設定ファイルを読み込み
load_env_file()
load_config_file()

# 出力ディレクトリの設定
def get_output_dir():
    """CSVファイルの出力先ディレクトリを取得"""
    return os.environ.get('OUTPUT_DIR', 'analysis_results')

# 動画ファイルリストのパス設定
def get_mov_files_path():
    """動画ファイルリストのパスを取得"""
    return os.environ.get('MOV_FILES_PATH', 'mov_files.txt')

# プレビュー表示の設定
def get_show_preview():
    """プレビュー表示の有無を取得"""
    return os.environ.get('SHOW_PREVIEW', 'false').lower() == 'true'

# 進捗表示間隔の設定
def get_progress_interval():
    """進捗表示間隔を取得"""
    return int(os.environ.get('PROGRESS_INTERVAL', '30'))

# DataWareHouseデータベースのパス設定
def get_dwh_database_path():
    """DataWareHouseデータベースのパスを取得"""
    return os.environ.get('DWH_DATABASE_PATH', '../DataWareHouse/database.db')

# コアライブラリのバージョン設定
def get_core_lib_version():
    """コアライブラリのバージョンを取得"""
    return os.environ.get('CORE_LIB_VERSION', '1.0.0')

# コアライブラリの更新情報設定
def get_core_lib_update_info():
    """コアライブラリの更新情報を取得"""
    return os.environ.get('CORE_LIB_UPDATE_INFO', '顔検出・分析機能の統合')

# コアライブラリのコミットハッシュ設定
def get_core_lib_commit_hash():
    """コアライブラリのコミットハッシュを取得"""
    return os.environ.get('CORE_LIB_COMMIT_HASH', 'a1b2c3d4e5f6a1b2c3d4e5f6a1b2c3d4e5f6a1b2')

# 出力ベースディレクトリの設定
def get_output_base_dir():
    """出力のベースディレクトリを取得"""
    return os.environ.get('OUTPUT_BASE_DIR', '02_core_lib_output')

# サングラス検出関連の設定
def get_sunglasses_brightness_threshold():
    """サングラス検出の輝度閾値を取得"""
    return int(os.environ.get('SUNGLASSES_BRIGHTNESS_THRESHOLD', '50'))

def get_eye_roi_scale_factor():
    """目のROI抽出のスケールファクターを取得"""
    return float(os.environ.get('EYE_ROI_SCALE_FACTOR', '1.5'))

def get_eye_roi_target_size():
    """目のROI抽出のターゲットサイズを取得"""
    size_str = os.environ.get('EYE_ROI_TARGET_SIZE', '50,50')
    try:
        width, height = map(int, size_str.split(','))
        return (width, height)
    except:
        return (50, 50)

def get_multi_frame_count():
    """マルチフレーム処理のフレーム数を取得"""
    return int(os.environ.get('MULTI_FRAME_COUNT', '3')) 