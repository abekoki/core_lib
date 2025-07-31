import cv2
import mediapipe as mp
import numpy as np
from scipy.spatial import distance as dist
import math

# MediaPipeの初期化
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True, 
                                 min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils
drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)

# ランドマーク描画用の色定義（BGR形式）
EAR_COLOR = (0, 255, 0)      # 緑色 - EAR計算用
POSE_COLOR = (255, 0, 0)     # 青色 - 顔の向き推定用
CONFIDENCE_COLOR = (0, 0, 255)  # 赤色 - 信頼度計算用
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
    
    # 信頼度計算用ランドマーク（赤色）
    confidence_indices = [1, 33, 263, 168]  # 鼻先、左右目、鼻根
    for idx in confidence_indices:
        landmark = face_landmarks.landmark[idx]
        x = int(landmark.x * image_size[1])
        y = int(landmark.y * image_size[0])
        cv2.circle(frame, (x, y), 4, CONFIDENCE_COLOR, 2)  # 輪郭のみ

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

# 簡易的な顔の向き推定関数（参考サイトの手法に基づく）
def estimate_pose_simple(face_landmarks, image_size):
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
    
    # 1. 両目を結ぶ直線の傾きから左右の向き（Yaw）を推定
    # 水平線との角度を計算
    eye_line_angle = math.atan2(right_eye_px[1] - left_eye_px[1], right_eye_px[0] - left_eye_px[0])
    eye_line_angle_deg = math.degrees(eye_line_angle)
    
    # 両目の中心点を計算
    eye_center_x = (left_eye_px[0] + right_eye_px[0]) / 2
    eye_center_y = (left_eye_px[1] + right_eye_px[1]) / 2
    eye_center = (int(eye_center_x), int(eye_center_y))
    
    # 2. 三角形の分析（左目、右目、鼻先）
    # 三角形の重心を計算
    triangle_center_x = (left_eye_px[0] + right_eye_px[0] + nose_tip_px[0]) / 3
    triangle_center_y = (left_eye_px[1] + right_eye_px[1] + nose_tip_px[1]) / 3
    triangle_center = (int(triangle_center_x), int(triangle_center_y))
    
    # 鼻先から三角形の重心へのベクトル
    nose_to_center_x = triangle_center[0] - nose_tip_px[0]
    nose_to_center_y = triangle_center[1] - nose_tip_px[1]
    
    # 3. 鼻筋の直線の分析
    # 鼻根から鼻先へのベクトル
    nose_bridge_to_tip_x = nose_tip_px[0] - nose_bridge_px[0]
    nose_bridge_to_tip_y = nose_tip_px[1] - nose_bridge_px[1]
    
    # 両目を結ぶ線分の長さ
    eye_distance = dist.euclidean(left_eye_px, right_eye_px)
    
    # Yaw（左右の向き）の計算（サンプルコードの手法に合わせる）
    # 三角形の重心と鼻の頂点とのX軸上の距離を計算
    distance_x = nose_tip_px[0] - triangle_center[0]
    
    # 矢印の長さを計算（上限を100pxとする）
    arrow_length = min(abs(distance_x), 100) * np.sign(distance_x)
    
    # 距離を角度に変換（サンプルコードの手法）
    yaw = np.clip(distance_x * 0.5, -45, 45)  # ±45度に制限
    
    # 三角形の法線ベクトルを計算
    # 三角形の3つの頂点から2つの辺ベクトルを計算
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
    
    # Pitch（上下の向き）の計算（サンプルコードの手法に合わせる）
    # 三角形の重心と鼻の頂点とのY軸上の距離を計算
    # 画像座標系ではY軸が下向きが正なので、符号を反転して上向きを正にする
    distance_y = triangle_center[1] - nose_tip_px[1]  # 符号を反転
    
    # Y軸方向の矢印の長さを計算（上限を100pxとする）
    pitch_arrow_length = min(abs(distance_y), 100) * np.sign(distance_y)
    
    # 距離を角度に変換（上向き=正、下向き=負）
    pitch = np.clip(distance_y * 0.5, -45, 45)  # ±45度に制限
    
    # 信頼度の計算（目のランドマークとその周辺の情報から）
    # 1. 三角形の面積による基本信頼度
    triangle_area = abs((left_eye_px[0] * (right_eye_px[1] - nose_tip_px[1]) + 
                        right_eye_px[0] * (nose_tip_px[1] - left_eye_px[1]) + 
                        nose_tip_px[0] * (left_eye_px[1] - right_eye_px[1])) / 2)
    
    # 面積を正規化（顔の大きさに依存しないように）
    normalized_area = triangle_area / (eye_distance ** 2) if eye_distance > 0 else 0
    area_confidence = min(normalized_area * 10, 1.0)  # 0-1の範囲に正規化
    
    # 2. 目のランドマークの可視性チェック（遮蔽検出強化）
    # 目の周辺の重要なランドマークを取得
    eye_landmarks = [
        33,   # 左目中心
        263,  # 右目中心
        159,  # 左上まぶた
        145,  # 左下まぶた
        386,  # 右上まぶた
        374,  # 右下まぶた
        133,  # 左目外角
        362,  # 右目外角
        157,  # 左目内角
        158,  # 左目内角
        387,  # 右目内角
        388,  # 右目内角
        160,  # 左目上部
        161,  # 左目上部
        163,  # 左目上部
        246,  # 左目下部
        247,  # 左目下部
        249,  # 左目下部
        373,  # 右目上部
        374,  # 右目上部
        375,  # 右目上部
        380,  # 右目下部
        381,  # 右目下部
        382   # 右目下部
    ]
    
    # 各ランドマークの可視性をチェック
    eye_visibility_scores = []
    z_coordinates = []
    
    for landmark_id in eye_landmarks:
        landmark = face_landmarks.landmark[landmark_id]
        
        # z座標を記録（遮蔽検出用）
        z_coordinates.append(landmark.z)
        
        # 基本的な可視性チェック
        if landmark.z > 0 and 0 <= landmark.x <= 1 and 0 <= landmark.y <= 1:
            eye_visibility_scores.append(1.0)
        else:
            eye_visibility_scores.append(0.0)
    
    # 基本的な可視性スコア
    basic_visibility = sum(eye_visibility_scores) / len(eye_visibility_scores)
    
    # 遮蔽検出の強化
    # 1. z座標の異常値検出（遮蔽時はz座標が異常な値になることが多い）
    z_coordinates = np.array(z_coordinates)
    z_mean = np.mean(z_coordinates)
    z_std = np.std(z_coordinates)
    
    # z座標が異常に分散している場合は遮蔽の可能性
    z_anomaly_score = 1.0 - min(1.0, z_std / 0.1)  # 標準偏差が0.1以上で異常
    
    # 2. 目の開瞼度による遮蔽検出
    left_eye_opening, right_eye_opening, _ = calculate_eye_opening(face_landmarks, image_size)
    
    # 開瞼度が異常に低い場合は遮蔽の可能性
    min_eye_opening = min(left_eye_opening, right_eye_opening)
    eye_opening_score = max(0.0, min_eye_opening / 0.1)  # 0.1以下で遮蔽の可能性
    
    # 3. 左右の目の開瞼度の差が異常に大きい場合も遮蔽の可能性
    eye_difference = abs(left_eye_opening - right_eye_opening)
    eye_difference_score = max(0.0, 1.0 - eye_difference / 0.05)  # 差が0.05以上で異常
    
    # 総合的な可視性スコア（遮蔽検出強化版）
    visibility_confidence = (
        basic_visibility * 0.4 +
        z_anomaly_score * 0.3 +
        eye_opening_score * 0.2 +
        eye_difference_score * 0.1
    )
    
    # 遮蔽が検出された場合は大幅に信頼度を下げる
    if basic_visibility < 0.5 or z_anomaly_score < 0.3 or eye_opening_score < 0.3:
        visibility_confidence *= 0.2  # 遮蔽時は大幅に信頼度を下げる
    
    # 3. 目の形状の妥当性チェック
    # 左右の目の開瞼度が極端に異なる場合は信頼度を下げる
    left_eye_opening, right_eye_opening, _ = calculate_eye_opening(face_landmarks, image_size)
    eye_symmetry = 1.0 - abs(left_eye_opening - right_eye_opening) / max(left_eye_opening + right_eye_opening, 0.001)
    eye_symmetry = max(0.0, min(1.0, eye_symmetry))  # 0-1の範囲に制限
    
    # 4. 目の位置関係の妥当性チェック
    # 両目の距離が極端に近い、または遠い場合は信頼度を下げる
    eye_distance_ratio = eye_distance / image_size[1]  # 画像幅に対する比率
    if 0.05 <= eye_distance_ratio <= 0.25:  # 適切な範囲
        distance_confidence = 1.0
    else:
        # 範囲外の場合は距離に応じて信頼度を下げる
        distance_confidence = max(0.0, 1.0 - abs(eye_distance_ratio - 0.15) / 0.15)
    
    # 5. 三角形の形状の妥当性チェック
    # 三角形が極端に歪んでいる場合は信頼度を下げる
    # 各辺の長さを計算
    left_eye_to_right_eye = dist.euclidean(left_eye_px, right_eye_px)
    left_eye_to_nose = dist.euclidean(left_eye_px, nose_tip_px)
    right_eye_to_nose = dist.euclidean(right_eye_px, nose_tip_px)
    
    # 三角形の辺の長さの比率をチェック
    sides = [left_eye_to_right_eye, left_eye_to_nose, right_eye_to_nose]
    sides.sort()
    if sides[2] > 0:  # 最長辺が0でない場合
        triangle_ratio = sides[0] / sides[2]  # 最短辺と最長辺の比率
        shape_confidence = max(0.0, triangle_ratio)  # 比率が1に近いほど良い
    else:
        shape_confidence = 0.0
    
    # 6. 総合信頼度の計算（改善された集約方法）
    # 各信頼度要素の重み
    confidence_weights = {
        'area': 0.15,          # 三角形の面積
        'visibility': 0.35,    # ランドマークの可視性（最重要）
        'symmetry': 0.25,      # 目の対称性
        'distance': 0.15,      # 目の距離の妥当性
        'shape': 0.10          # 三角形の形状
    }
    
    # 基本の重み付き平均
    weighted_confidence = (
        area_confidence * confidence_weights['area'] +
        visibility_confidence * confidence_weights['visibility'] +
        eye_symmetry * confidence_weights['symmetry'] +
        distance_confidence * confidence_weights['distance'] +
        shape_confidence * confidence_weights['shape']
    )
    
    # 信頼度の補正と集約（遮蔽検出強化版）
    # 1. 可視性が低い場合は全体の信頼度を大幅に下げる
    if visibility_confidence < 0.5:
        weighted_confidence *= visibility_confidence * 0.3  # より厳しく削減
    
    # 2. 遮蔽が強く検出された場合はさらに大幅に削減
    if basic_visibility < 0.3 or z_anomaly_score < 0.2 or eye_opening_score < 0.2:
        weighted_confidence *= 0.1  # 遮蔽時は90%削減
    
    # 2. 目の対称性が極端に悪い場合は信頼度を下げる
    if eye_symmetry < 0.3:
        weighted_confidence *= eye_symmetry * 0.7
    
    # 3. 三角形の形状が極端に歪んでいる場合は信頼度を下げる
    if shape_confidence < 0.2:
        weighted_confidence *= shape_confidence * 0.6
    
    # 4. 距離が不適切な場合は信頼度を下げる
    if distance_confidence < 0.3:
        weighted_confidence *= distance_confidence * 0.8
    
    # 5. 最終的な信頼度を計算（非線形補正）
    # 低信頼度の場合はより厳しく、高信頼度の場合は緩やかに評価
    if weighted_confidence < 0.3:
        # 低信頼度の場合はさらに下げる
        confidence = weighted_confidence * 0.8
    elif weighted_confidence > 0.8:
        # 高信頼度の場合は少し上げる
        confidence = weighted_confidence * 1.1
    else:
        # 中程度の信頼度はそのまま
        confidence = weighted_confidence
    
    # 信頼度を0-1の範囲に制限
    confidence = max(0.0, min(1.0, confidence))
    
    return yaw, pitch, confidence, {
        'nose_tip': nose_tip_px,
        'left_eye': left_eye_px,
        'right_eye': right_eye_px,
        'nose_bridge': nose_bridge_px,
        'eye_center': eye_center,
        'triangle_center': triangle_center,
        'normal_vector': normalized_normal,
        'edge1': (edge1_x, edge1_y),
        'edge2': (edge2_x, edge2_y),
        'distance_x': distance_x,
        'distance_y': distance_y,
        'arrow_length': arrow_length,
        'pitch_arrow_length': pitch_arrow_length,
        'triangle_area': triangle_area,
        'eye_distance': eye_distance,
        'confidence_details': {
            'area_confidence': area_confidence,
            'visibility_confidence': visibility_confidence,
            'basic_visibility': basic_visibility,
            'z_anomaly_score': z_anomaly_score,
            'eye_opening_score': eye_opening_score,
            'eye_difference_score': eye_difference_score,
            'eye_symmetry': eye_symmetry,
            'distance_confidence': distance_confidence,
            'shape_confidence': shape_confidence,
            'eye_distance_ratio': eye_distance_ratio,
            'weighted_confidence': weighted_confidence,
            'final_confidence': confidence
        }
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
    
    # Yaw（左右）の矢印（サンプルコードの手法に合わせる）
    # 矢印の方向ベクトルをX軸に制限
    yaw_direction = np.array([landmarks_info['arrow_length'], 0])
    yaw_end = tuple(map(int, np.array(base_point) + yaw_direction))
    
    # Pitch（上下）の矢印（Y軸方向に制限）
    # 上向きを正、下向きを負にするため、Y軸の符号を反転
    pitch_direction = np.array([0, -landmarks_info['pitch_arrow_length']])  # Y軸の符号を反転
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
            for face_landmarks in results.multi_face_landmarks:
                # 開瞼度計算（瞼間距離と鼻の長さによる）
                left_eye_opening, right_eye_opening, nose_length = calculate_eye_opening(face_landmarks, frame.shape[:2])
                
                # 簡易的な顔の向き推定
                yaw, pitch, confidence, landmarks_info = estimate_pose_simple(face_landmarks, frame.shape[:2])
                
                # 結果表示
                cv2.putText(frame, f"Left Eye Opening: {left_eye_opening:.3f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(frame, f"Right Eye Opening: {right_eye_opening:.3f}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(frame, f"Nose Length: {nose_length:.1f}px", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(frame, f"Confidence: {confidence:.2f}", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(frame, f"Yaw: {yaw:.1f}deg, Pitch: {pitch:.1f}deg", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(frame, f"Normal Vector: {landmarks_info['normal_vector']:.3f}", (10, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(frame, f"Distance X: {landmarks_info.get('distance_x', 0):.1f}, Y: {landmarks_info.get('distance_y', 0):.1f}", (10, 210), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                # 色分けされたランドマーク描画
                draw_landmarks_with_colors(frame, face_landmarks, frame.shape[:2])
                
                # 顔の向きを矢印で描画
                draw_pose_arrow(frame, yaw, pitch, confidence, landmarks_info)
                
                # 色の説明を表示
                cv2.putText(frame, "Green: Eye Opening", (10, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.5, EAR_COLOR, 1)
                cv2.putText(frame, "Blue: Pose", (10, 260), cv2.FONT_HERSHEY_SIMPLEX, 0.5, POSE_COLOR, 1)
                cv2.putText(frame, "Red: Confidence", (10, 280), cv2.FONT_HERSHEY_SIMPLEX, 0.5, CONFIDENCE_COLOR, 1)
                cv2.putText(frame, "Yellow: Triangle", (10, 300), cv2.FONT_HERSHEY_SIMPLEX, 0.5, TRIANGLE_COLOR, 1)
                cv2.putText(frame, "Blue Arrow: Yaw, Yellow Arrow: Pitch", (10, 320), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                cv2.putText(frame, "Red Arrow: Normal Vector", (10, 340), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        
        cv2.imshow("Eye Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()