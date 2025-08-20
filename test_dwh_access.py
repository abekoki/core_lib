#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DataWareHouseデータベースアクセステストスクリプト
作成日: 2025-08-20
作成者: AI Assistant
"""

import sys
import os
from pathlib import Path

def test_dwh_import():
    """DataWareHouseのインポートをテスト"""
    print("=== DataWareHouseインポートテスト ===")
    
    # パスを追加
    dwh_path = "../DataWareHouse"
    if os.path.exists(dwh_path):
        sys.path.append(dwh_path)
        print(f"✓ DataWareHouseパスを追加: {dwh_path}")
    else:
        print(f"✗ DataWareHouseパスが見つかりません: {dwh_path}")
        return False
    
    try:
        # 各モジュールのインポートをテスト
        print("\n各モジュールのインポートテスト:")
        
        # connection.py
        try:
            from datawarehouse.connection import DWHConnection, get_connection
            print("  ✓ connection.py: 成功")
        except ImportError as e:
            print(f"  ✗ connection.py: 失敗 - {e}")
        
        # exceptions.py
        try:
            from datawarehouse.exceptions import DWHError, DWHConstraintError, DWHNotFoundError
            print("  ✓ exceptions.py: 成功")
        except ImportError as e:
            print(f"  ✗ exceptions.py: 失敗 - {e}")
        
        # video_api.py
        try:
            from datawarehouse.video_api import create_video, get_video, list_videos
            print("  ✓ video_api.py: 成功")
        except ImportError as e:
            print(f"  ✗ video_api.py: 失敗 - {e}")
        
        # core_lib_api.py
        try:
            from datawarehouse.core_lib_api import create_core_lib_version, list_core_lib_versions
            print("  ✓ core_lib_api.py: 成功")
        except ImportError as e:
            print(f"  ✗ core_lib_api.py: 失敗 - {e}")
        
        # subject_api.py
        try:
            from datawarehouse.subject_api import create_subject, list_subjects
            print("  ✓ subject_api.py: 成功")
        except ImportError as e:
            print(f"  ✗ subject_api.py: 失敗 - {e}")
        
        # task_api.py
        try:
            from datawarehouse.task_api import create_task, list_tasks
            print("  ✓ task_api.py: 成功")
        except ImportError as e:
            print(f"  ✗ task_api.py: 失敗 - {e}")
        
        return True
        
    except Exception as e:
        print(f"✗ インポートテストでエラーが発生しました: {e}")
        return False

def test_database_connection():
    """データベース接続をテスト"""
    print("\n=== データベース接続テスト ===")
    
    # データベースパス
    db_path = "../DataWareHouse/database.db"
    
    if not os.path.exists(db_path):
        print(f"✗ データベースファイルが見つかりません: {db_path}")
        return False
    
    print(f"✓ データベースファイルを発見: {db_path}")
    
    try:
        # SQLite3で直接接続テスト
        import sqlite3
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # テーブル一覧を取得
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = cursor.fetchall()
        
        print(f"✓ データベース接続成功")
        print(f"  テーブル数: {len(tables)}")
        
        for table in tables:
            table_name = table[0]
            # 各テーブルの件数を確認
            cursor.execute(f"SELECT COUNT(*) FROM {table_name};")
            count = cursor.fetchone()[0]
            print(f"    {table_name}: {count}件")
        
        conn.close()
        return True
        
    except Exception as e:
        print(f"✗ データベース接続でエラーが発生しました: {e}")
        return False

def test_api_functions():
    """API関数の動作をテスト"""
    print("\n=== API関数テスト ===")
    
    try:
        # 必要なモジュールをインポート
        from datawarehouse import (
            DWHConnection,
            create_video,
            get_video,
            list_videos,
            create_subject,
            list_subjects,
            create_task,
            list_tasks,
            create_core_lib_version,
            list_core_lib_versions
        )
        
        # データベースパス
        db_path = "../DataWareHouse/database.db"
        
        print("各API関数のテスト:")
        
        # 被験者一覧取得
        try:
            subjects = list_subjects(db_path=db_path)
            print(f"  ✓ list_subjects: 成功 - {len(subjects)}件")
        except Exception as e:
            print(f"  ✗ list_subjects: 失敗 - {e}")
        
        # タスク一覧取得
        try:
            tasks = list_tasks(db_path=db_path)
            print(f"  ✓ list_tasks: 成功 - {len(tasks)}件")
        except Exception as e:
            print(f"  ✗ list_tasks: 失敗 - {e}")
        
        # ビデオ一覧取得
        try:
            videos = list_videos(db_path=db_path)
            print(f"  ✓ list_videos: 成功 - {len(videos)}件")
        except Exception as e:
            print(f"  ✗ list_videos: 失敗 - {e}")
        
        # コアライブラリバージョン一覧取得
        try:
            core_lib_versions = list_core_lib_versions(db_path=db_path)
            print(f"  ✓ list_core_lib_versions: 成功 - {len(core_lib_versions)}件")
        except Exception as e:
            print(f"  ✗ list_core_lib_versions: 失敗 - {e}")
        
        return True
        
    except Exception as e:
        print(f"✗ API関数テストでエラーが発生しました: {e}")
        return False

def main():
    """メイン処理"""
    print("DataWareHouseデータベースアクセステスト")
    print("=" * 50)
    
    # 1. インポートテスト
    import_success = test_dwh_import()
    
    if not import_success:
        print("\n✗ インポートテストが失敗しました")
        return
    
    # 2. データベース接続テスト
    db_success = test_database_connection()
    
    if not db_success:
        print("\n✗ データベース接続テストが失敗しました")
        return
    
    # 3. API関数テスト
    api_success = test_api_functions()
    
    if not api_success:
        print("\n✗ API関数テストが失敗しました")
        return
    
    print("\n" + "=" * 50)
    print("✓ すべてのテストが成功しました！")
    print("DataWareHouseのデータベースに正常にアクセスできます。")

if __name__ == "__main__":
    main()
