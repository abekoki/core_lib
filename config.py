#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
設定管理モジュール
作成日: 2025-07-31
作成者: AI Assistant
"""

import os
from pathlib import Path

def get_output_dir():
    """出力ディレクトリを取得"""
    return os.getenv('OUTPUT_DIR', 'analysis_results')

def get_mov_files_path():
    """動画ファイルリストのパスを取得"""
    return os.getenv('MOV_FILES_PATH', 'mov_files.txt')

def get_show_preview():
    """プレビュー表示の有無を取得"""
    return os.getenv('SHOW_PREVIEW', 'false').lower() == 'true'

def get_progress_interval():
    """進捗表示間隔を取得"""
    return int(os.getenv('PROGRESS_INTERVAL', '30'))

def load_env_file():
    """環境変数ファイルを読み込み"""
    env_file = Path('.env')
    if env_file.exists():
        with open(env_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    key, value = line.split('=', 1)
                    os.environ[key.strip()] = value.strip()

# 環境変数ファイルを読み込み
load_env_file() 