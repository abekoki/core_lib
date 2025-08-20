#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DataWareHouseデータベースの初期設定スクリプト
作成日: 2025-08-20
作成者: AI Assistant
"""

import os
import sys
from pathlib import Path

# DataWareHouseのAPIライブラリをインポート
sys.path.append('../DataWareHouse')
try:
    from datawarehouse import (
        DWHConnection,
        create_subject,
        create_task,
        create_video,
        create_core_lib_version
    )
    DWH_AVAILABLE = True
except ImportError as e:
    print(f"エラー: DataWareHouse APIライブラリのインポートに失敗しました: {e}")
    sys.exit(1)

def setup_database():
    """データベースの初期設定を実行"""
    print("=== DataWareHouseデータベースの初期設定 ===")
    
    # データベースパス
    db_path = "../DataWareHouse/database.db"
    
    if not os.path.exists(db_path):
        print(f"エラー: データベースファイルが見つかりません: {db_path}")
        sys.exit(1)
    
    print(f"データベースパス: {db_path}")
    
    try:
        # 被験者を作成
        print("\n1. 被験者を作成中...")
        subject_id = create_subject("Subject A", db_path)
        print(f"✓ 被験者を作成しました: subject_ID={subject_id}")
        
        # タスクを作成
        print("\n2. タスクを作成中...")
        task_id = create_task(1, "顔検出・分析", "動画ファイルからの顔検出と分析", db_path)
        print(f"✓ タスクを作成しました: task_ID={task_id}")
        
        # コアライブラリのバージョンを作成
        print("\n3. コアライブラリのバージョンを作成中...")
        core_lib_id = create_core_lib_version(
            version="1.0.0",
            update_info="顔検出・分析機能の統合",
            commit_hash="a1b2c3d4e5f6a1b2c3d4e5f6a1b2c3d4e5f6a1b2",
            db_path=db_path
        )
        print(f"✓ コアライブラリのバージョンを作成しました: core_lib_ID={core_lib_id}")
        
        # サンプル動画ファイルをデータベースに登録
        print("\n4. サンプル動画ファイルを登録中...")
        
        # 01_mov_dataディレクトリ内の動画ファイルを検索
        mov_data_dir = "../DataWareHouse/01_mov_data"
        if os.path.exists(mov_data_dir):
            video_files = []
            for ext in ['*.mp4', '*.avi', '*.mov', '*.mkv']:
                video_files.extend(Path(mov_data_dir).glob(ext))
            
            if video_files:
                print(f"動画ファイルを発見: {len(video_files)}件")
                
                for i, video_file in enumerate(video_files, 1):
                    print(f"  [{i}/{len(video_files)}] {video_file.name} を登録中...")
                    
                    # 動画の長さを取得
                    import cv2
                    cap = cv2.VideoCapture(str(video_file))
                    if cap.isOpened():
                        fps = cap.get(cv2.CAP_PROP_FPS)
                        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                        cap.release()
                        
                        if fps > 0:
                            video_length = int(frame_count / fps)
                            
                            # 動画をデータベースに登録
                            video_id = create_video(
                                video_dir=str(video_file),
                                subject_id=subject_id,
                                video_date="2025-08-20",
                                video_length=video_length,
                                db_path=db_path
                            )
                            print(f"    ✓ 登録完了: video_ID={video_id}")
                        else:
                            print(f"    ✗ 動画の長さを取得できませんでした")
                    else:
                        print(f"    ✗ 動画ファイルを開けませんでした")
            else:
                print("動画ファイルが見つかりませんでした")
        else:
            print("01_mov_dataディレクトリが見つかりませんでした")
        
        print("\n=== 初期設定が完了しました ===")
        print(f"被験者ID: {subject_id}")
        print(f"タスクID: {task_id}")
        print(f"コアライブラリID: {core_lib_id}")
        
    except Exception as e:
        print(f"エラー: データベースの初期設定に失敗しました: {e}")
        sys.exit(1)

if __name__ == "__main__":
    setup_database()
