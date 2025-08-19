import warnings
warnings.filterwarnings("ignore", category=UserWarning)

import cv2
import mediapipe as mp
import numpy as np
from scipy.spatial import distance as dist
import math

# 設定ファイルをインポート
from config import (
    get_sunglasses_brightness_threshold,
    get_eye_roi_scale_factor,
    get_eye_roi_target_size,
    get_multi_frame_count
)

# MediaPipeの初期化
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True, 
                                 min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils
drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)

# ランドマーク描画用の色定義（BGR形式）
EAR_COLOR = (0, 255, 0)      # 緑色 - EAR計算用
POSE_COLOR = (255, 0, 0)     # 青色 - 顔の向き推定用
TRIANGLE_COLOR = (0, 255, 255)  # 黄色 - 三角形描画用

# ランドマーク描画関数
def draw_landmarks_with_colors(frame, face_landmarks, image_size):
    # 開瞼度計算用ランドマーク（緑色）
    eye_opening_indices = [159, 145, 386, 374, 168, 1]  # 左右上まぶた、左右下まぶた、鼻根、鼻先
    
    for idx in eye_opening_indices:
        landmark = face_landmarks.landmark[idx]
        x = int(landmark.x * image_size[1])
        y = int(landmark.y * image_size[0])
        cv2.circle(frame, (x, y), 3, EAR_COLOR, -1)
    
    # 顔の向き推定用ランドマーク（青色）
    pose_indices = [1, 33, 263, 168]  # 鼻先、左目、右目、鼻根
    for idx in pose_indices:
        landmark = face_landmarks.landmark[idx]
        x = int(landmark.x * image_size[1])
        y = int(landmark.y * image_size[0])
        cv2.circle(frame, (x, y), 5, POSE_COLOR, -1)

# 動的ROI抽出関数（修正案.mdベース）
def extract_eye_roi_dynamic(image, face_landmarks, eye_indices, scale_factor=None):
    """
    目のランドマークの中心を基準に、目と目の距離や目の幅に基づいて動的にROIサイズを設定
    """
    if not face_landmarks or not eye_indices:
        print("DEBUG: face_landmarks or eye_indices is None")
        return None
    
    # スケールファクターを設定から取得（デフォルト値も設定）
    if scale_factor is None:
        scale_factor = get_eye_roi_scale_factor()
    
    # 目のランドマーク座標を取得
    eye_points = []
    for i in eye_indices:
        if i < len(face_landmarks.landmark):
            landmark = face_landmarks.landmark[i]
            eye_points.append([landmark.x, landmark.y])
    
    if len(eye_points) < 3:
        print(f"DEBUG: Not enough eye points: {len(eye_points)}")
        return None
    
    # 目の幅を計算
    x_coords = [p[0] for p in eye_points]
    y_coords = [p[1] for p in eye_points]
    eye_width = max(x_coords) - min(x_coords)
    
    # 顔のスケール（目と目の距離）を計算（MediaPipeの例: 左目中心33, 右目中心263）
    if 33 < len(face_landmarks.landmark) and 263 < len(face_landmarks.landmark):
        left_eye_center = face_landmarks.landmark[33]
        right_eye_center = face_landmarks.landmark[263]
        # 正規化座標をピクセル座標に変換してから距離を計算
        left_eye_px = np.array([left_eye_center.x * image.shape[1], left_eye_center.y * image.shape[0]])
        right_eye_px = np.array([right_eye_center.x * image.shape[1], right_eye_center.y * image.shape[0]])
        inter_eye_distance = np.linalg.norm(left_eye_px - right_eye_px)
    else:
        # ランドマークが不足している場合は目の幅を使用
        inter_eye_distance = eye_width * image.shape[1] * 2  # ピクセル座標に変換
    
    # ROIサイズを動的に決定（目と目の距離を基準にスケール）
    roi_width = int(inter_eye_distance * scale_factor * 0.5)  # 目と目の距離の半分程度
    roi_height = int(roi_width * 0.6)  # 縦は幅の0.6倍（目の形状を考慮）
    
    # 目の中心を計算
    x_center = int(np.mean(x_coords) * image.shape[1])
    y_center = int(np.mean(y_coords) * image.shape[0])
    
    # ROIを計算（画像範囲内に収まるよう調整）
    x_min = max(0, x_center - roi_width // 2)
    x_max = min(image.shape[1], x_center + roi_width // 2)
    y_min = max(0, y_center - roi_height // 2)
    y_max = min(image.shape[0], y_center + roi_height // 2)
    
    # デバッグ情報を出力
    print(f"DEBUG: ROI calculation - inter_eye_distance: {inter_eye_distance:.3f}, scale_factor: {scale_factor}")
    print(f"DEBUG: ROI size - width: {roi_width}, height: {roi_height}")
    print(f"DEBUG: ROI center - x: {x_center}, y: {y_center}")
    print(f"DEBUG: ROI bounds - x_min: {x_min}, x_max: {x_max}, y_min: {y_min}, y_max: {y_max}")
    print(f"DEBUG: Image shape: {image.shape}")
    
    # ROIサイズが0でないことを確認
    if roi_width <= 0 or roi_height <= 0:
        print(f"DEBUG: Invalid ROI size - width: {roi_width}, height: {roi_height}")
        return None
    
    # ROIを切り出し
    eye_roi = image[y_min:y_max, x_min:x_max]
    
    if eye_roi.size == 0:
        print(f"DEBUG: Extracted ROI is empty - size: {eye_roi.size}")
        return None
    
    print(f"DEBUG: Successfully extracted ROI - size: {eye_roi.shape}")
    return eye_roi

# 輝度分析によるサングラス検出（リサイズで正規化）
def check_sunglasses(eye_roi, target_size=None):
    """
    ROIを固定サイズにリサイズして輝度を計算し、サングラスを検出
    """
    if eye_roi is None or eye_roi.size == 0:
        print("DEBUG: check_sunglasses - eye_roi is None or empty")
        return False, 0
    
    print(f"DEBUG: check_sunglasses - eye_roi shape: {eye_roi.shape}, size: {eye_roi.size}")
    
    # ターゲットサイズを設定から取得
    if target_size is None:
        target_size = get_eye_roi_target_size()
    
    print(f"DEBUG: check_sunglasses - target_size: {target_size}")
    
    # ROIを固定サイズにリサイズ
    resized_roi = cv2.resize(eye_roi, target_size, interpolation=cv2.INTER_AREA)
    print(f"DEBUG: check_sunglasses - resized_roi shape: {resized_roi.shape}")
    
    gray = cv2.cvtColor(resized_roi, cv2.COLOR_BGR2GRAY)
    brightness = np.mean(gray)
    
    print(f"DEBUG: check_sunglasses - calculated brightness: {brightness:.1f}")
    
    # 閾値を設定から取得
    threshold = get_sunglasses_brightness_threshold()
    is_sunglasses = brightness < threshold
    
    print(f"DEBUG: check_sunglasses - threshold: {threshold}, is_sunglasses: {is_sunglasses}")
    
    return is_sunglasses, brightness

# 複数フレームでの輝度分析（オプション）
def check_sunglasses_multi_frame(image, face_landmarks, eye_indices, num_frames=None):
    """
    複数フレームの平均輝度を使用してノイズを軽減
    """
    if num_frames is None:
        num_frames = get_multi_frame_count()
    
    brightness_values = []
    for _ in range(num_frames):
        eye_roi = extract_eye_roi_dynamic(image, face_landmarks, eye_indices)
        if eye_roi is not None and eye_roi.size > 0:
            resized_roi = cv2.resize(eye_roi, get_eye_roi_target_size(), interpolation=cv2.INTER_AREA)
            gray = cv2.cvtColor(resized_roi, cv2.COLOR_BGR2GRAY)
            brightness_values.append(np.mean(gray))
    
    if not brightness_values:
        return False, 0
    
    average_brightness = np.mean(brightness_values)
    threshold = get_sunglasses_brightness_threshold()
    return average_brightness < threshold, average_brightness

# 信頼度調整関数（サングラス装着時のみ信頼度を下げる）
def adjust_confidence(original_confidence, eye_roi):
    """
    サングラス装着時のみ信頼度を30%下げる
    """
    print(f"DEBUG: adjust_confidence - original_confidence: {original_confidence}")
    
    is_sunglasses, brightness = check_sunglasses(eye_roi)
    
    print(f"DEBUG: adjust_confidence - is_sunglasses: {is_sunglasses}, brightness: {brightness}")
    
    if is_sunglasses:
        adjusted_confidence = original_confidence * 0.7
        print(f"DEBUG: adjust_confidence - adjusted_confidence: {adjusted_confidence}")
        return adjusted_confidence, brightness  # サングラス装着時に信頼度を30%下げる
    
    print(f"DEBUG: adjust_confidence - no adjustment, returning original")
    return original_confidence, brightness

# 開瞼度計算関数（瞼間距離と鼻の長さによる再定義）
def calculate_eye_opening(face_landmarks, image_size):
    """
    瞼間の距離と鼻の長さを使用して開瞼度を計算
    鼻の長さで正規化することで、顔の大きさや距離に依存しない指標を算出
    """
    # 瞼間の距離を計算（上まぶたから下まぶた）
    # 左目の瞼間距離
    left_upper_eyelid = face_landmarks.landmark[159]  # 左目上まぶた
    left_lower_eyelid = face_landmarks.landmark[145]  # 左目下まぶた
    left_eye_opening = dist.euclidean(
        (left_upper_eyelid.x * image_size[1], left_upper_eyelid.y * image_size[0]),
        (left_lower_eyelid.x * image_size[1], left_lower_eyelid.y * image_size[0])
    )
    
    # 右目の瞼間距離
    right_upper_eyelid = face_landmarks.landmark[386]  # 右目上まぶた
    right_lower_eyelid = face_landmarks.landmark[374]  # 右目下まぶた
    right_eye_opening = dist.euclidean(
        (right_upper_eyelid.x * image_size[1], right_upper_eyelid.y * image_size[0]),
        (right_lower_eyelid.x * image_size[1], right_lower_eyelid.y * image_size[0])
    )
    
    # 鼻の長さを計算（鼻根から鼻先まで）
    nose_bridge = face_landmarks.landmark[168]  # 鼻根
    nose_tip = face_landmarks.landmark[1]       # 鼻先
    nose_length = dist.euclidean(
        (nose_bridge.x * image_size[1], nose_bridge.y * image_size[0]),
        (nose_tip.x * image_size[1], nose_tip.y * image_size[0])
    )
    
    # 鼻の長さで正規化して開瞼度を計算
    left_eye_opening_ratio = left_eye_opening / nose_length if nose_length > 0 else 0
    right_eye_opening_ratio = right_eye_opening / nose_length if nose_length > 0 else 0
    
    return left_eye_opening_ratio, right_eye_opening_ratio, nose_length

# 簡易的な顔の向き推定関数
def estimate_pose_simple(face_landmarks, image_size, frame=None):
    """
    参考サイトの手法に基づく簡易的な顔の向き推定
    行列計算を使わずに、ランドマークの幾何学的関係から推定
    """
    # 必要なランドマークの座標を取得
    nose_tip = face_landmarks.landmark[1]      # 鼻先
    left_eye = face_landmarks.landmark[33]     # 左目
    right_eye = face_landmarks.landmark[263]   # 右目
    nose_bridge = face_landmarks.landmark[168] # 鼻根
    
    # 座標をピクセル座標に変換
    nose_tip_px = (int(nose_tip.x * image_size[1]), int(nose_tip.y * image_size[0]))
    left_eye_px = (int(left_eye.x * image_size[1]), int(left_eye.y * image_size[0]))
    right_eye_px = (int(right_eye.x * image_size[1]), int(right_eye.y * image_size[0]))
    nose_bridge_px = (int(nose_bridge.x * image_size[1]), int(nose_bridge.y * image_size[0]))
    
    # 両目を結ぶ直線の傾きから左右の向き（Yaw）を推定
    eye_line_angle = math.atan2(right_eye_px[1] - left_eye_px[1], right_eye_px[0] - left_eye_px[0])
    eye_line_angle_deg = math.degrees(eye_line_angle)
    
    # 両目の中心点を計算
    eye_center_x = (left_eye_px[0] + right_eye_px[0]) / 2
    eye_center_y = (left_eye_px[1] + right_eye_px[1]) / 2
    eye_center = (int(eye_center_x), int(eye_center_y))
    
    # 三角形の分析（左目、右目、鼻先）
    # 三角形の重心を計算
    triangle_center_x = (left_eye_px[0] + right_eye_px[0] + nose_tip_px[0]) / 3
    triangle_center_y = (left_eye_px[1] + right_eye_px[1] + nose_tip_px[1]) / 3
    triangle_center = (int(triangle_center_x), int(triangle_center_y))
    
    # 鼻先から三角形の重心へのベクトル
    nose_to_center_x = triangle_center[0] - nose_tip_px[0]
    nose_to_center_y = triangle_center[1] - nose_tip_px[1]
    
    # 両目を結ぶ線分の長さ
    eye_distance = dist.euclidean(left_eye_px, right_eye_px)

    # 三角形面積を計算
    triangle_area = abs((left_eye_px[0] * (right_eye_px[1] - nose_tip_px[1]) +
                         right_eye_px[0] * (nose_tip_px[1] - left_eye_px[1]) +
                         nose_tip_px[0] * (left_eye_px[1] - right_eye_px[1])) / 2)

    # distance_x / distance_y for pose arrows
    distance_x = nose_tip_px[0] - triangle_center[0]
    distance_y = triangle_center[1] - nose_tip_px[1]  # +: up

    # 開瞼度を計算
    left_eye_opening, right_eye_opening, nose_length = calculate_eye_opening(face_landmarks, image_size)

    # 開瞼度比率（鼻長で正規化済みなのでそのまま使用）
    left_eye_opening_ratio = left_eye_opening
    right_eye_opening_ratio = right_eye_opening

    # サングラス検出（修正案.mdベース）
    left_eye_indices = [33, 160, 158, 133, 153, 144]  # MediaPipeの左目インデックス
    eye_roi = extract_eye_roi_dynamic(frame, face_landmarks, left_eye_indices)
    
    # 基本的な信頼度（サングラス検出前）
    basic_confidence = 1.0  # 基本信頼度は1.0
    
    # サングラス装着時のみ信頼度を調整
    adjusted_confidence, brightness = adjust_confidence(basic_confidence, eye_roi)
    
    print(f"DEBUG: estimate_pose_simple - adjusted_confidence: {adjusted_confidence}, brightness: {brightness}")

    # Yaw（左右の向き）とPitch（上下の向き）を計算
    yaw = np.clip(distance_x * 0.5, -45, 45)  # ±45度
    pitch = np.clip(distance_y * 0.5, -45, 45)  # ±45度

    # 三角形の法線ベクトルを計算
    edge1_x = right_eye_px[0] - left_eye_px[0]
    edge1_y = right_eye_px[1] - left_eye_px[1]
    edge2_x = nose_tip_px[0] - left_eye_px[0]
    edge2_y = nose_tip_px[1] - left_eye_px[1]
    
    # 外積で法線ベクトルを計算（2Dなのでz成分のみ）
    normal_z = edge1_x * edge2_y - edge1_y * edge2_x
    
    # 法線ベクトルの向きを正規化
    normal_magnitude = math.sqrt(edge1_x**2 + edge1_y**2) * math.sqrt(edge2_x**2 + edge2_y**2)
    if normal_magnitude > 0:
        normalized_normal = normal_z / normal_magnitude
    else:
        normalized_normal = 0

    return yaw, pitch, adjusted_confidence, {
        'nose_tip': nose_tip_px,
        'left_eye': left_eye_px,
        'right_eye': right_eye_px,
        'nose_bridge': nose_bridge_px,
        'eye_center': eye_center,
        'triangle_center': triangle_center,
        'distance_x': distance_x,
        'distance_y': distance_y,
        'triangle_area': triangle_area,
        'eye_distance': eye_distance,
        'normal_vector': normalized_normal,
        'eye_roi': eye_roi,
        'sunglasses_detected': check_sunglasses(eye_roi)[0] if eye_roi is not None else False,
        'brightness': brightness,
        'threshold': get_sunglasses_brightness_threshold(),
        'scale_factor': get_eye_roi_scale_factor(),
        'target_size': get_eye_roi_target_size()
    }

# 描画関数（顔の向きを矢印で表示）
def draw_pose_arrow(frame, yaw, pitch, confidence, landmarks_info):
    """
    顔の向きを矢印で描画
    """
    # 基準点（鼻先）
    base_point = landmarks_info['nose_tip']
    
    # 矢印の長さ（信頼度に応じて変化、最大100px）
    arrow_length = int(confidence * 100)
    if arrow_length < 10:
        arrow_length = 10
    
    # Yaw（左右）の矢印
    yaw_arrow_length = min(abs(landmarks_info['distance_x']), 100) * np.sign(landmarks_info['distance_x'])
    yaw_direction = np.array([yaw_arrow_length, 0])
    yaw_end = tuple(map(int, np.array(base_point) + yaw_direction))
    
    # Pitch（上下）の矢印（Y軸方向に制限）
    pitch_arrow_length = min(abs(landmarks_info['distance_y']), 100) * np.sign(landmarks_info['distance_y'])
    pitch_direction = np.array([0, -pitch_arrow_length])  # Y軸の符号を反転
    pitch_end = tuple(map(int, np.array(base_point) + pitch_direction))
    
    # 矢印を描画
    cv2.arrowedLine(frame, base_point, yaw_end, (255, 0, 0), 3, tipLength=0.3)  # 青色でYaw
    cv2.arrowedLine(frame, base_point, pitch_end, (0, 255, 255), 3, tipLength=0.3)  # 黄色でPitch
    
    # 三角形を描画
    triangle_points = np.array([landmarks_info['left_eye'], 
                               landmarks_info['right_eye'], 
                               landmarks_info['nose_tip']], np.int32)
    cv2.polylines(frame, [triangle_points], True, TRIANGLE_COLOR, 2)
    
    # 法線ベクトルを描画（三角形の重心から）
    if 'normal_vector' in landmarks_info:
        normal = landmarks_info['normal_vector']
        triangle_center = landmarks_info['triangle_center']
        
        # 法線ベクトルの方向を矢印で表示
        normal_length = 30
        normal_end_x = int(triangle_center[0] + normal * normal_length)
        normal_end_y = triangle_center[1]
        normal_end = (normal_end_x, normal_end_y)
        
        # 法線ベクトルを赤色で描画
        cv2.arrowedLine(frame, triangle_center, normal_end, (0, 0, 255), 2, tipLength=0.3)

# メイン処理（モジュール化のため削除）
if __name__ == "__main__":
    cap = cv2.VideoCapture(0)  # ウェブカメラ（動画の場合はファイルパス）
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # RGB変換
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb_frame)
        
        # 検出結果の処理
        if results.multi_face_landmarks:
            # 複数の顔が検出された場合は最初の顔を使用
            face_landmarks = results.multi_face_landmarks[0]
            
            # 開瞼度計算（瞼間距離と鼻の長さによる）
            left_eye_opening, right_eye_opening, nose_length = calculate_eye_opening(face_landmarks, frame.shape[:2])
            
            # 簡易的な顔の向き推定
            yaw, pitch, confidence, landmarks_info = estimate_pose_simple(face_landmarks, frame.shape[:2], frame)
            
            # 結果表示
            cv2.putText(frame, f"Left Eye Opening: {left_eye_opening:.3f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame, f"Right Eye Opening: {right_eye_opening:.3f}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame, f"Nose Length: {nose_length:.1f}px", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame, f"Confidence: {confidence:.2f}", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame, f"Yaw: {yaw:.1f}deg, Pitch: {pitch:.1f}deg", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame, f"Normal Vector: {landmarks_info['normal_vector']:.3f}", (10, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame, f"Distance X: {landmarks_info.get('distance_x', 0):.1f}, Y: {landmarks_info.get('distance_y', 0):.1f}", (10, 210), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame, f"Sunglasses: {'Yes' if landmarks_info.get('sunglasses_detected', False) else 'No'}", (10, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # 輝度値と閾値を表示
            brightness = landmarks_info.get('brightness', 0)
            threshold = landmarks_info.get('threshold', 50)
            cv2.putText(frame, f"Brightness: {brightness:.1f}", (10, 270), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame, f"Threshold: {threshold}", (10, 300), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # 設定値を表示
            scale_factor = landmarks_info.get('scale_factor', 1.5)
            target_size = landmarks_info.get('target_size', (50, 50))
            cv2.putText(frame, f"Scale Factor: {scale_factor}", (10, 330), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame, f"Target Size: {target_size[0]}x{target_size[1]}", (10, 360), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # 色分けされたランドマーク描画
            draw_landmarks_with_colors(frame, face_landmarks, frame.shape[:2])
            
            # 顔の向きを矢印で描画
            draw_pose_arrow(frame, yaw, pitch, confidence, landmarks_info)
            
            # 色の説明を表示
            cv2.putText(frame, "Green: Eye Opening", (10, 390), cv2.FONT_HERSHEY_SIMPLEX, 0.5, EAR_COLOR, 1)
            cv2.putText(frame, "Blue: Pose", (10, 410), cv2.FONT_HERSHEY_SIMPLEX, 0.5, POSE_COLOR, 1)
            cv2.putText(frame, "Yellow: Triangle", (10, 430), cv2.FONT_HERSHEY_SIMPLEX, 0.5, TRIANGLE_COLOR, 1)
            cv2.putText(frame, "Blue Arrow: Yaw, Yellow Arrow: Pitch", (10, 450), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.putText(frame, "Red Arrow: Normal Vector", (10, 470), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        
        cv2.imshow("Eye Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()