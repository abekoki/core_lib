#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
動画ファイルから顔検出結果をCSV出力するスクリプト（DataWareHouse統合版）
作成日: 2025-07-31
更新日: 2025-08-20
作成者: AI Assistant
"""

import cv2
import csv
import os
import sys
from pathlib import Path
import numpy as np
from datetime import datetime
import sqlite3
import traceback  # デバッグ用

# デバッグ用の設定
DEBUG_MODE = True

def debug_print(message):
    """デバッグ用の出力関数"""
    if DEBUG_MODE:
        print(f"[DEBUG] {message}")

def debug_error(message, error=None):
    """デバッグ用のエラー出力関数"""
    if DEBUG_MODE:
        print(f"[ERROR] {message}")
        if error:
            print(f"[ERROR] 詳細: {error}")
            print(f"[ERROR] トレースバック:")
            traceback.print_exc()

debug_print("スクリプト開始: モジュールインポート開始")

# eye_detectionモジュールをインポート
try:
    from eye_detection import (
        face_mesh, 
        calculate_eye_opening, 
        estimate_pose_simple,
        draw_landmarks_with_colors,
        draw_pose_arrow
    )
    debug_print("✓ eye_detectionモジュールのインポート成功")
except ImportError as e:
    debug_error("eye_detectionモジュールのインポート失敗", e)
    sys.exit(1)

# 設定ファイルをインポート
try:
    from config import (
        get_output_dir,
        get_dwh_database_path,
        get_core_lib_version,
        get_core_lib_update_info,
        get_core_lib_commit_hash,
        get_output_base_dir,
        get_show_preview,
        get_progress_interval
    )
    debug_print("✓ configモジュールのインポート成功")
except ImportError as e:
    debug_error("configモジュールのインポート失敗", e)
    sys.exit(1)

# DataWareHouseのAPIライブラリをインポート
debug_print("DataWareHouse APIライブラリのインポート開始")
sys.path.append('../DataWareHouse')
try:
    from datawarehouse import (
        DWHConnection,
        get_video,
        list_videos,
        create_core_lib_version,
        create_core_lib_output,
        find_core_lib_by_commit_hash
    )
    DWH_AVAILABLE = True
    debug_print("✓ DataWareHouse APIライブラリのインポート成功")
except ImportError as e:
    debug_print(f"警告: DataWareHouse APIライブラリのインポートに失敗しました: {e}")
    print("データベース機能は無効になります")
    DWH_AVAILABLE = False

debug_print("モジュールインポート完了")

def get_video_length(video_path):
    """動画ファイルの長さ（秒）を取得"""
    debug_print(f"動画ファイルの長さ取得開始: {video_path}")
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        debug_error(f"動画ファイルを開けませんでした: {video_path}")
        return 0
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    
    debug_print(f"動画情報 - FPS: {fps}, フレーム数: {frame_count}")
    
    if fps > 0:
        duration = int(frame_count / fps)
        debug_print(f"動画の長さ: {duration}秒")
        return duration
    
    debug_error("FPSが0以下のため、動画の長さを計算できません")
    return 0

def get_or_create_core_lib_version(db_path=None):
    """コアライブラリのバージョンを取得または作成"""
    if not DWH_AVAILABLE:
        print("警告: DataWareHouseが利用できないため、コアライブラリバージョンの作成をスキップします")
        return None
    
    if db_path is None:
        db_path = get_dwh_database_path()
    
    try:
        version = get_core_lib_version()
        commit_hash = get_core_lib_commit_hash()
        
        # 既存のバージョンを検索
        existing_version = find_core_lib_by_commit_hash(commit_hash, db_path)
        if existing_version:
            print(f"既存のコアライブラリバージョンを使用: {existing_version['core_lib_version']}")
            return existing_version['core_lib_ID']
        
        # 新しいバージョンを作成
        update_info = get_core_lib_update_info()
        core_lib_id = create_core_lib_version(
            version=version,
            update_info=update_info,
            commit_hash=commit_hash,
            db_path=db_path
        )
        
        print(f"新しいコアライブラリバージョンを作成しました: core_lib_ID={core_lib_id}")
        return core_lib_id
        
    except Exception as e:
        print(f"エラー: コアライブラリバージョンの作成に失敗しました: {e}")
        return None

def save_analysis_result_to_database(video_id, core_lib_id, output_dir, db_path=None):
    """分析結果をデータベースに保存"""
    if not DWH_AVAILABLE:
        print("警告: DataWareHouseが利用できないため、データベース保存をスキップします")
        return None
    
    if db_path is None:
        db_path = get_dwh_database_path()
    
    try:
        # パスを正規化してLinux互換に変換
        normalized_output_dir = os.path.normpath(output_dir).replace('\\', '/')
        debug_print(f"データベース保存用パス正規化: {output_dir} -> {normalized_output_dir}")
        
        # コアライブラリの出力結果を登録
        output_id = create_core_lib_output(
            core_lib_id=core_lib_id,
            video_id=video_id,
            output_dir=normalized_output_dir,
            db_path=db_path
        )
        
        print(f"分析結果をデータベースに保存しました: core_lib_output_ID={output_id}")
        return output_id
        
    except Exception as e:
        print(f"エラー: 分析結果のデータベース保存に失敗しました: {e}")
        return None

def get_videos_from_database(db_path=None):
    """データベースから動画ファイルのリストを取得"""
    if not DWH_AVAILABLE:
        print("警告: DataWareHouseが利用できないため、データベースからの動画取得をスキップします")
        return []
    
    if db_path is None:
        db_path = get_dwh_database_path()
    
    try:
        # データベースから動画一覧を取得
        videos = list_videos(db_path=db_path)
        
        # ファイルパスのみを抽出（相対パスを絶対パスに変換）
        video_paths = []
        for video in videos:
            video_path = video['video_dir']
            
            # 相対パスの場合、DataWareHouseディレクトリからの絶対パスに変換
            if not os.path.isabs(video_path):
                # データベースファイルのディレクトリを基準に絶対パスを作成
                db_dir = os.path.dirname(os.path.abspath(db_path))
                absolute_path = os.path.join(db_dir, video_path)
                debug_print(f"相対パスを絶対パスに変換: {video_path} -> {absolute_path}")
                video_path = absolute_path
            
            if os.path.exists(video_path):
                video_paths.append(video_path)
                debug_print(f"動画ファイル確認済み: {video_path}")
            else:
                debug_print(f"警告: 動画ファイルが見つかりません: {video_path}")
        
        print(f"データベースから {len(video_paths)} 件の動画ファイルを取得しました")
        return video_paths
        
    except Exception as e:
        print(f"エラー: データベースからの動画取得に失敗しました: {e}")
        return []

def draw_debug_frame(frame, face_landmarks, landmarks_info, frame_count, confidence):
    """
    デバッグ用のフレームを描画（ランドマーク、三角形、矢印を含む）
    """
    debug_frame = frame.copy()
    
    # ランドマークを描画
    draw_landmarks_with_colors(debug_frame, face_landmarks, frame.shape[:2])
    
    # 三角形を描画
    if landmarks_info:
        # 三角形の頂点を取得
        left_eye = landmarks_info.get('left_eye', (0, 0))
        right_eye = landmarks_info.get('right_eye', (0, 0))
        nose_tip = landmarks_info.get('nose_tip', (0, 0))
        
        # 三角形を描画（黄色）
        triangle_points = np.array([left_eye, right_eye, nose_tip], np.int32)
        cv2.polylines(debug_frame, [triangle_points], True, (0, 255, 255), 2)
        
        # 三角形の重心を描画
        triangle_center = landmarks_info.get('triangle_center', (0, 0))
        cv2.circle(debug_frame, triangle_center, 5, (255, 0, 255), -1)
        
        # 顔の向きを矢印で描画
        yaw = landmarks_info.get('distance_x', 0) * 0.5
        pitch = landmarks_info.get('distance_y', 0) * 0.5
        draw_pose_arrow(debug_frame, yaw, pitch, confidence, landmarks_info)
    
    # 情報テキストを描画
    cv2.putText(debug_frame, f"Frame: {frame_count}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(debug_frame, f"Confidence: {confidence:.2f}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    if landmarks_info:
        yaw = landmarks_info.get('distance_x', 0) * 0.5
        pitch = landmarks_info.get('distance_y', 0) * 0.5
        cv2.putText(debug_frame, f"Yaw: {yaw:.1f}deg, Pitch: {pitch:.1f}deg", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # サングラス検出結果を表示
        sunglasses_detected = landmarks_info.get('sunglasses_detected', False)
        sunglasses_text = "Sunglasses: Yes" if sunglasses_detected else "Sunglasses: No"
        sunglasses_color = (0, 0, 255) if sunglasses_detected else (0, 255, 0)
        cv2.putText(debug_frame, sunglasses_text, (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, sunglasses_color, 2)
        
        # 輝度値と閾値を表示
        brightness = landmarks_info.get('brightness', 0)
        threshold = landmarks_info.get('threshold', 50)
        cv2.putText(debug_frame, f"Brightness: {brightness:.1f}", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(debug_frame, f"Threshold: {threshold}", (10, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # 設定値を表示
        scale_factor = landmarks_info.get('scale_factor', 1.5)
        target_size = landmarks_info.get('target_size', (50, 50))
        cv2.putText(debug_frame, f"Scale Factor: {scale_factor}", (10, 210), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(debug_frame, f"Target Size: {target_size[0]}x{target_size[1]}", (10, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    return debug_frame

def analyze_video(video_path, output_csv_path, show_preview=False):
    """
    動画ファイルを分析してCSVファイルに結果を出力
    
    Args:
        video_path (str): 入力動画ファイルのパス
        output_csv_path (str): 出力CSVファイルのパス
        show_preview (bool): プレビュー表示の有無
    """
    
    # 動画ファイルを開く
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"エラー: 動画ファイルを開けませんでした: {video_path}")
        return False
    
    # 動画の情報を取得
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    print(f"動画情報: {total_frames}フレーム, {fps:.2f}fps, {width}x{height}")
    
    # デバッグ動画の出力設定
    debug_video_path = output_csv_path.replace('.csv', '_debug.mp4')
    debug_video_writer = None
    
    # CSVファイルを開く
    with open(output_csv_path, 'w', newline='', encoding='utf-8') as csvfile:
        # CSVヘッダーを定義（サングラス検出に特化）
        fieldnames = [
            'frame', 'leye_openness', 'reye_openness', 'confidence', 
            'face_yaw', 'face_pitch', 'nose_length', 'normal_vector',
            'distance_x', 'distance_y', 'triangle_area', 'eye_distance',
            'sunglasses_detected', 'eye_roi_width', 'eye_roi_height',
            # ランドマーク位置
            'left_eye_x', 'left_eye_y', 'left_eye_z',
            'right_eye_x', 'right_eye_y', 'right_eye_z',
            'nose_tip_x', 'nose_tip_y', 'nose_tip_z',
            'nose_bridge_x', 'nose_bridge_y', 'nose_bridge_z',
            'triangle_center_x', 'triangle_center_y'
        ]
        
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        
        frame_count = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            
            # 進捗表示
            progress_interval = get_progress_interval()
            if frame_count % progress_interval == 0:  # 設定された間隔ごとに表示
                progress = (frame_count / total_frames) * 100
                print(f"進捗: {frame_count}/{total_frames} ({progress:.1f}%)")
            
            # RGB変換
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(rgb_frame)
            
            # デフォルト値（顔が検出されない場合）
            row_data = {
                'frame': frame_count,
                'leye_openness': 0.0,
                'reye_openness': 0.0,
                'confidence': 0.0,
                'face_yaw': 0.0,
                'face_pitch': 0.0,
                'nose_length': 0.0,
                'normal_vector': 0.0,
                'distance_x': 0.0,
                'distance_y': 0.0,
                'triangle_area': 0.0,
                'eye_distance': 0.0,
                'sunglasses_detected': False,
                'eye_roi_width': 0,
                'eye_roi_height': 0,
                # ランドマーク位置のデフォルト値
                'left_eye_x': 0.0, 'left_eye_y': 0.0, 'left_eye_z': 0.0,
                'right_eye_x': 0.0, 'right_eye_y': 0.0, 'right_eye_z': 0.0,
                'nose_tip_x': 0.0, 'nose_tip_y': 0.0, 'nose_tip_z': 0.0,
                'nose_bridge_x': 0.0, 'nose_bridge_y': 0.0, 'nose_bridge_z': 0.0,
                'triangle_center_x': 0.0, 'triangle_center_y': 0.0
            }
            
            # 検出結果の処理
            if results.multi_face_landmarks:
                # 最初の顔のみを処理（複数顔がある場合）
                face_landmarks = results.multi_face_landmarks[0]
                
                # 開瞼度計算
                left_eye_opening, right_eye_opening, nose_length = calculate_eye_opening(face_landmarks, frame.shape[:2])
                
                # 顔の向き推定
                yaw, pitch, confidence, landmarks_info = estimate_pose_simple(face_landmarks, frame.shape[:2], frame)
                
                # ランドマーク位置を取得
                left_eye_landmark = face_landmarks.landmark[33]  # 左目
                right_eye_landmark = face_landmarks.landmark[263]  # 右目
                nose_tip_landmark = face_landmarks.landmark[1]  # 鼻先
                nose_bridge_landmark = face_landmarks.landmark[168]  # 鼻根
                
                # データを更新
                row_data.update({
                    'leye_openness': left_eye_opening,
                    'reye_openness': right_eye_opening,
                    'confidence': confidence,
                    'face_yaw': yaw,
                    'face_pitch': pitch,
                    'nose_length': nose_length,
                    'normal_vector': landmarks_info.get('normal_vector', 0.0),
                    'distance_x': landmarks_info.get('distance_x', 0.0),
                    'distance_y': landmarks_info.get('distance_y', 0.0),
                    'triangle_area': landmarks_info.get('triangle_area', 0.0),
                    'eye_distance': landmarks_info.get('eye_distance', 0.0),
                    'sunglasses_detected': landmarks_info.get('sunglasses_detected', False),
                    'eye_roi_width': landmarks_info.get('eye_roi', None).shape[1] if landmarks_info.get('eye_roi') is not None else 0,
                    'eye_roi_height': landmarks_info.get('eye_roi', None).shape[0] if landmarks_info.get('eye_roi') is not None else 0,
                    # ランドマーク位置を追加
                    'left_eye_x': left_eye_landmark.x, 'left_eye_y': left_eye_landmark.y, 'left_eye_z': left_eye_landmark.z,
                    'right_eye_x': right_eye_landmark.x, 'right_eye_y': right_eye_landmark.y, 'right_eye_z': right_eye_landmark.z,
                    'nose_tip_x': nose_tip_landmark.x, 'nose_tip_y': nose_tip_landmark.y, 'nose_tip_z': nose_tip_landmark.z,
                    'nose_bridge_x': nose_bridge_landmark.x, 'nose_bridge_y': nose_bridge_landmark.y, 'nose_bridge_z': nose_bridge_landmark.z,
                    'triangle_center_x': landmarks_info.get('triangle_center', (0, 0))[0],
                    'triangle_center_y': landmarks_info.get('triangle_center', (0, 0))[1]
                })
                
                # デバッグフレームを作成
                debug_frame = draw_debug_frame(frame, face_landmarks, landmarks_info, frame_count, confidence)
                
                # デバッグ動画の初期化（最初のフレームで）
                if debug_video_writer is None:
                    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                    debug_video_writer = cv2.VideoWriter(debug_video_path, fourcc, fps, (width, height))
                
                # デバッグ動画にフレームを書き込み
                if debug_video_writer is not None:
                    debug_video_writer.write(debug_frame)
                
                # プレビュー表示
                if show_preview:
                    cv2.imshow("Video Analysis", debug_frame)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break
            else:
                # 顔が検出されない場合は元のフレームをデバッグ動画に書き込み
                if debug_video_writer is not None:
                    debug_frame = frame.copy()
                    cv2.putText(debug_frame, f"Frame: {frame_count} - No Face Detected", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    debug_video_writer.write(debug_frame)
                
                # プレビュー表示
                if show_preview:
                    cv2.imshow("Video Analysis", frame)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break
            
            # CSVに書き込み
            writer.writerow(row_data)
    
    # リソースを解放
    cap.release()
    if debug_video_writer is not None:
        debug_video_writer.release()
    if show_preview:
        cv2.destroyAllWindows()
    
    print(f"分析完了: {output_csv_path}")
    print(f"デバッグ動画保存: {debug_video_path}")
    return True

def main():
    """メイン処理"""
    
    debug_print("=== メイン処理開始 ===")
    print("=== 動画分析システム（DataWareHouse統合版）===")
    
    # DataWareHouseの利用可能性を確認
    if not DWH_AVAILABLE:
        debug_print("警告: DataWareHouse APIライブラリが利用できません")
        print("従来のファイルベースの処理を実行します")
        # 従来の処理を実行（既存のコード）
        run_legacy_analysis()
        return
    
    debug_print("DataWareHouse APIライブラリが利用可能です")
    
    # データベースパスを取得
    db_path = get_dwh_database_path()
    debug_print(f"データベースパス: {db_path}")
    print(f"データベースパス: {db_path}")
    
    # データベースファイルの存在確認
    if not os.path.exists(db_path):
        debug_error(f"データベースファイルが見つかりません: {db_path}")
        print(f"エラー: データベースファイルが見つかりません: {db_path}")
        print("従来のファイルベースの処理を実行します")
        run_legacy_analysis()
        return
    
    debug_print("データベースファイルの存在確認完了")
    
    # コアライブラリのバージョンを取得または作成
    debug_print("コアライブラリバージョンの取得/作成開始")
    core_lib_id = get_or_create_core_lib_version(db_path)
    if core_lib_id is None:
        debug_error("コアライブラリバージョンの作成に失敗しました")
        print("エラー: コアライブラリバージョンの作成に失敗しました")
        return
    
    debug_print(f"コアライブラリID: {core_lib_id}")
    
    # データベースから動画ファイルのリストを取得
    debug_print("データベースからの動画ファイル取得開始")
    video_paths = get_videos_from_database(db_path)
    
    if not video_paths:
        debug_print("データベースに動画ファイルが登録されていません")
        print("データベースに動画ファイルが登録されていません")
        print("従来のファイルベースの処理を実行します")
        run_legacy_analysis()
        return
    
    debug_print(f"取得した動画ファイル数: {len(video_paths)}")
    
    # 出力ベースディレクトリを作成
    output_base_dir = get_output_base_dir()
    version_dir = os.path.join(output_base_dir, f"v{get_core_lib_version()}")
    debug_print(f"出力ディレクトリ作成: {version_dir}")
    os.makedirs(version_dir, exist_ok=True)
    
    print(f"分析対象動画ファイル数: {len(video_paths)}")
    
    # 各動画ファイルを分析
    for i, video_path in enumerate(video_paths, 1):
        debug_print(f"動画分析開始 [{i}/{len(video_paths)}]: {video_path}")
        print(f"\n[{i}/{len(video_paths)}] 分析中: {video_path}")
        
        # 動画ファイルの存在確認
        if not os.path.exists(video_path):
            debug_error(f"動画ファイルが見つかりません: {video_path}")
            print(f"警告: 動画ファイルが見つかりません: {video_path}")
            continue
        
        debug_print("動画ファイルの存在確認完了")
        
        # 動画IDを取得（既存のvideo_IDを使用）
        # video_tableは読み込み専用のため、新規登録は行わない
        try:
            # データベースから既存の動画情報を取得
            # 絶対パスから相対パスに変換して検索（Windowsのパス区切り文字を/に統一）
            relative_path = os.path.relpath(video_path, os.path.dirname(db_path)).replace('\\', '/')
            debug_print(f"相対パスに変換: {video_path} -> {relative_path}")
            
            # データベースから動画情報を検索
            from datawarehouse import list_videos
            all_videos = list_videos(db_path=db_path)  # db_pathを明示的に指定
            video_info = None
            
            for video in all_videos:
                if video['video_dir'] == relative_path:
                    video_info = video
                    break
            
            if video_info:
                video_id = video_info['video_ID']
                debug_print(f"既存の動画IDを使用: {video_id}")
            else:
                debug_error(f"動画ファイルがデータベースに登録されていません: {relative_path}")
                print(f"警告: 動画ファイルがデータベースに登録されていません: {relative_path}")
                continue
        except Exception as e:
            debug_error(f"動画情報の取得に失敗しました: {e}")
            print(f"警告: 動画情報の取得に失敗しました: {video_path}")
            continue
        
        debug_print(f"動画ID: {video_id}")
        
        # 出力ディレクトリを作成（video_ID別）
        video_name = Path(video_path).stem
        output_dir = os.path.join(version_dir, f"{video_id}")
        debug_print(f"出力ディレクトリ: {output_dir}")
        os.makedirs(output_dir, exist_ok=True)
        
        # 出力ファイルパスを生成
        output_csv = os.path.join(output_dir, f"{video_name}_analysis.csv")
        debug_print(f"出力CSVファイル: {output_csv}")
        
        # 動画分析を実行
        show_preview = get_show_preview()
        debug_print(f"プレビュー表示: {show_preview}")
        success = analyze_video(video_path, output_csv, show_preview=show_preview)
        
        if success:
            debug_print(f"動画分析完了: {output_csv}")
            print(f"✓ 分析完了: {output_csv}")
            
            # 分析結果をデータベースに保存（パスをLinux互換に変換）
            debug_print("分析結果のデータベース保存開始")
            # Windowsのパス区切り文字をLinux互換の/に変換
            db_output_dir = os.path.normpath(output_dir).replace('\\', '/')
            debug_print(f"データベース保存用パス正規化: {output_dir} -> {db_output_dir}")
            output_id = save_analysis_result_to_database(
                video_id, core_lib_id, db_output_dir, db_path
            )
            
            if output_id:
                debug_print(f"分析結果のデータベース保存完了: {output_id}")
                print(f"✓ データベース保存完了: core_lib_output_ID={output_id}")
            else:
                debug_error("分析結果のデータベース保存失敗")
                print("✗ データベース保存失敗")
        else:
            debug_error(f"動画分析失敗: {video_path}")
            print(f"✗ 分析失敗: {video_path}")
    
    debug_print("全ての分析完了")
    print(f"\n全ての分析が完了しました。結果は {version_dir} ディレクトリに保存されています。")

def run_legacy_analysis():
    """従来のファイルベースの分析処理"""
    debug_print("=== 従来のファイルベース処理開始 ===")
    print("\n=== 従来のファイルベース処理を実行 ===")
    
    # 動画ファイルのリストを直接指定（従来の処理用）
    mov_files_path = "mov_files.txt"
    debug_print(f"動画ファイルリストパス: {mov_files_path}")
    
    if not os.path.exists(mov_files_path):
        debug_error(f"動画ファイルリストが見つかりません: {mov_files_path}")
        print(f"エラー: {mov_files_path} が見つかりません")
        return
    
    debug_print("動画ファイルリストの存在確認完了")
    
    # 設定から出力ディレクトリを作成
    output_dir = get_output_dir()
    debug_print(f"出力ディレクトリ: {output_dir}")
    os.makedirs(output_dir, exist_ok=True)
    
    # 動画ファイルのリストを読み込み
    with open(mov_files_path, 'r', encoding='utf-8') as f:
        video_files = [line.strip() for line in f if line.strip()]
    
    debug_print(f"読み込んだ動画ファイル数: {len(video_files)}")
    print(f"分析対象動画ファイル数: {len(video_files)}")
    
    # 各動画ファイルを分析
    for i, video_path in enumerate(video_files, 1):
        if not os.path.exists(video_path):
            debug_error(f"動画ファイルが見つかりません: {video_path}")
            print(f"警告: 動画ファイルが見つかりません: {video_path}")
            continue
        
        debug_print(f"動画分析開始 [{i}/{len(video_files)}]: {video_path}")
        print(f"\n[{i}/{len(video_files)}] 分析中: {video_path}")
        
        # 出力CSVファイル名を生成
        video_name = Path(video_path).stem
        output_csv = os.path.join(output_dir, f"{video_name}_analysis.csv")
        debug_print(f"出力CSVファイル: {output_csv}")
        
        # 動画分析を実行
        show_preview = get_show_preview()
        debug_print(f"プレビュー表示: {show_preview}")
        success = analyze_video(video_path, output_csv, show_preview=show_preview)
        
        if success:
            debug_print(f"動画分析完了: {output_csv}")
            print(f"✓ 分析完了: {output_csv}")
        else:
            debug_error(f"動画分析失敗: {video_path}")
            print(f"✗ 分析失敗: {video_path}")
    
    debug_print("従来のファイルベース処理完了")
    print(f"\n全ての分析が完了しました。結果は {output_dir} ディレクトリに保存されています。")

if __name__ == "__main__":
    main() 