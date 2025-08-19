#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
動画ファイルから顔検出結果をCSV出力するスクリプト
作成日: 2025-07-31
作成者: AI Assistant
"""

import cv2
import csv
import os
import sys
from pathlib import Path
import numpy as np
from datetime import datetime

# eye_detectionモジュールをインポート
from eye_detection import (
    face_mesh, 
    calculate_eye_opening, 
    estimate_pose_simple,
    draw_landmarks_with_colors,
    draw_pose_arrow
)

# 設定ファイルをインポート
from config import (
    get_output_dir,
    get_mov_files_path,
    get_show_preview,
    get_progress_interval
)

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
    
    # 設定から動画ファイルのリストを読み込み
    mov_files_path = get_mov_files_path()
    
    if not os.path.exists(mov_files_path):
        print(f"エラー: {mov_files_path} が見つかりません")
        return
    
    # 設定から出力ディレクトリを作成
    output_dir = get_output_dir()
    os.makedirs(output_dir, exist_ok=True)
    
    # 動画ファイルのリストを読み込み
    with open(mov_files_path, 'r', encoding='utf-8') as f:
        video_files = [line.strip() for line in f if line.strip()]
    
    print(f"分析対象動画ファイル数: {len(video_files)}")
    
    # 各動画ファイルを分析
    for i, video_path in enumerate(video_files, 1):
        if not os.path.exists(video_path):
            print(f"警告: 動画ファイルが見つかりません: {video_path}")
            continue
        
        print(f"\n[{i}/{len(video_files)}] 分析中: {video_path}")
        
        # 出力CSVファイル名を生成
        video_name = Path(video_path).stem
        output_csv = os.path.join(output_dir, f"{video_name}_analysis.csv")
        
        # 動画分析を実行
        show_preview = get_show_preview()
        success = analyze_video(video_path, output_csv, show_preview=show_preview)
        
        if success:
            print(f"✓ 分析完了: {output_csv}")
        else:
            print(f"✗ 分析失敗: {video_path}")
    
    print(f"\n全ての分析が完了しました。結果は {output_dir} ディレクトリに保存されています。")

if __name__ == "__main__":
    main() 