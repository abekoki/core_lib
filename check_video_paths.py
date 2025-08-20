#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
動画ファイルのパスを確認し、絶対パスに変換するスクリプト
作成日: 2025-08-20
作成者: AI Assistant
"""

import sqlite3
import os
from pathlib import Path

def check_video_paths():
    """動画ファイルのパスを確認し、絶対パスに変換"""
    print("=== 動画ファイルパス確認 ===")
    
    # データベースに接続
    db_path = "../DataWareHouse/database.db"
    if not os.path.exists(db_path):
        print(f"エラー: データベースファイルが見つかりません: {db_path}")
        return
    
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    try:
        # 動画テーブルからパスを取得
        cursor.execute("SELECT video_ID, video_dir FROM video_table")
        videos = cursor.fetchall()
        
        print(f"データベースに登録されている動画ファイル数: {len(videos)}")
        print()
        
        print("現在の動画パス:")
        for video_id, video_path in videos:
            print(f"  video_ID: {video_id}, パス: {video_path}")
        
        print()
        print("絶対パスに変換:")
        for video_id, video_path in videos:
            # 相対パスを絶対パスに変換
            absolute_path = os.path.abspath(os.path.join("../DataWareHouse", video_path))
            print(f"  video_ID: {video_id}, 絶対パス: {absolute_path}")
            
            # ファイルの存在確認
            if os.path.exists(absolute_path):
                print(f"    ✓ ファイル存在: はい")
            else:
                print(f"    ✗ ファイル存在: いいえ")
        
        print()
        print("パス修正の提案:")
        print("データベースの動画パスを絶対パスに更新することを推奨します")
        
    except Exception as e:
        print(f"エラー: {e}")
    
    finally:
        conn.close()

if __name__ == "__main__":
    check_video_paths()
