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
    estimate_pose_simple
)

# 設定ファイルをインポート
from config import (
    get_output_dir,
    get_mov_files_path,
    get_show_preview,
    get_progress_interval
)

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
    
    # CSVファイルを開く
    with open(output_csv_path, 'w', newline='', encoding='utf-8') as csvfile:
        # CSVヘッダーを定義
        fieldnames = [
            'frame', 'leye_openness', 'reye_openness', 'confidence', 
            'face_yaw', 'face_pitch', 'nose_length', 'normal_vector',
            'distance_x', 'distance_y', 'arrow_length', 'pitch_arrow_length',
            'triangle_area', 'eye_distance', 'area_confidence', 'visibility_confidence',
            'basic_visibility', 'z_anomaly_score', 'eye_opening_score', 'eye_difference_score',
            'eye_symmetry', 'distance_confidence', 'shape_confidence', 'eye_distance_ratio',
            'weighted_confidence', 'final_confidence'
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
                'arrow_length': 0.0,
                'pitch_arrow_length': 0.0,
                'triangle_area': 0.0,
                'eye_distance': 0.0,
                'area_confidence': 0.0,
                'visibility_confidence': 0.0,
                'basic_visibility': 0.0,
                'z_anomaly_score': 0.0,
                'eye_opening_score': 0.0,
                'eye_difference_score': 0.0,
                'eye_symmetry': 0.0,
                'distance_confidence': 0.0,
                'shape_confidence': 0.0,
                'eye_distance_ratio': 0.0,
                'weighted_confidence': 0.0,
                'final_confidence': 0.0
            }
            
            # 検出結果の処理
            if results.multi_face_landmarks:
                # 最初の顔のみを処理（複数顔がある場合）
                face_landmarks = results.multi_face_landmarks[0]
                
                # 開瞼度計算
                left_eye_opening, right_eye_opening, nose_length = calculate_eye_opening(face_landmarks, frame.shape[:2])
                
                # 顔の向き推定
                yaw, pitch, confidence, landmarks_info = estimate_pose_simple(face_landmarks, frame.shape[:2])
                
                # データを更新
                confidence_details = landmarks_info.get('confidence_details', {})
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
                    'arrow_length': landmarks_info.get('arrow_length', 0.0),
                    'pitch_arrow_length': landmarks_info.get('pitch_arrow_length', 0.0),
                    'triangle_area': landmarks_info.get('triangle_area', 0.0),
                    'eye_distance': landmarks_info.get('eye_distance', 0.0),
                                            'area_confidence': confidence_details.get('area_confidence', 0.0),
                        'visibility_confidence': confidence_details.get('visibility_confidence', 0.0),
                        'basic_visibility': confidence_details.get('basic_visibility', 0.0),
                        'z_anomaly_score': confidence_details.get('z_anomaly_score', 0.0),
                        'eye_opening_score': confidence_details.get('eye_opening_score', 0.0),
                        'eye_difference_score': confidence_details.get('eye_difference_score', 0.0),
                        'eye_symmetry': confidence_details.get('eye_symmetry', 0.0),
                        'distance_confidence': confidence_details.get('distance_confidence', 0.0),
                        'shape_confidence': confidence_details.get('shape_confidence', 0.0),
                        'eye_distance_ratio': confidence_details.get('eye_distance_ratio', 0.0),
                        'weighted_confidence': confidence_details.get('weighted_confidence', 0.0),
                        'final_confidence': confidence_details.get('final_confidence', 0.0)
                })
                
                # プレビュー表示
                if show_preview:
                    # 結果をフレームに描画
                    cv2.putText(frame, f"Frame: {frame_count}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    cv2.putText(frame, f"Yaw: {yaw:.1f}deg, Pitch: {pitch:.1f}deg", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    cv2.putText(frame, f"Confidence: {confidence:.2f}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    
                    cv2.imshow("Video Analysis", frame)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break
            
            # CSVに書き込み
            writer.writerow(row_data)
    
    # リソースを解放
    cap.release()
    if show_preview:
        cv2.destroyAllWindows()
    
    print(f"分析完了: {output_csv_path}")
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