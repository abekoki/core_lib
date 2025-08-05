import warnings
warnings.filterwarnings("ignore", category=UserWarning)

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

def analyze_eye_brightness(frame, face_landmarks, image_size):
    """
    目領域の輝度ヒストグラムを分析して、黒目と白目の存在を検出
    """
    try:
        # グレースケールに変換
        if len(frame.shape) == 3:
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            gray_frame = frame
        
        # 左右の目の領域を定義
        left_eye_landmarks = [
            33,   # 左目中心
            159,  # 左上まぶた
            145,  # 左下まぶた
            133,  # 左目外角
            157,  # 左目内角
            158   # 左目内角
        ]
        
        right_eye_landmarks = [
            263,  # 右目中心
            386,  # 右上まぶた
            374,  # 右下まぶた
            362,  # 右目外角
            387,  # 右目内角
            388   # 右目内角
        ]
        
        # 左右の目の輝度スコアを計算
        left_eye_score = analyze_single_eye_brightness(gray_frame, face_landmarks, left_eye_landmarks, image_size)
        right_eye_score = analyze_single_eye_brightness(gray_frame, face_landmarks, right_eye_landmarks, image_size)
        
        # 両目の平均スコアを返す
        return (left_eye_score + right_eye_score) / 2.0
        
    except Exception as e:
        # エラーが発生した場合は低い信頼度を返す
        return 0.1

def analyze_single_eye_brightness(gray_frame, face_landmarks, eye_landmarks, image_size):
    """
    単一の目の輝度ヒストグラムを分析
    """
    try:
        # 目のランドマークから目の領域を定義
        eye_points = []
        for landmark_id in eye_landmarks:
            landmark = face_landmarks.landmark[landmark_id]
            x = int(landmark.x * image_size[1])
            y = int(landmark.y * image_size[0])
            eye_points.append([x, y])
        
        if len(eye_points) < 3:
            return 0.1
        
        # 目の領域を囲む矩形を計算
        eye_points = np.array(eye_points, dtype=np.int32)
        x, y, w, h = cv2.boundingRect(eye_points)
        
        # 領域を少し拡張
        margin = 5
        x = max(0, x - margin)
        y = max(0, y - margin)
        w = min(gray_frame.shape[1] - x, w + 2 * margin)
        h = min(gray_frame.shape[0] - y, h + 2 * margin)
        
        # 目の領域を抽出
        eye_region = gray_frame[y:y+h, x:x+w]
        
        if eye_region.size == 0:
            return 0.1
        
        # 輝度ヒストグラムを計算
        hist = cv2.calcHist([eye_region], [0], None, [256], [0, 256])
        hist = hist.flatten()
        
        # ヒストグラムを正規化
        hist = hist / np.sum(hist)
        
        # 黒目領域（低輝度）と白目領域（高輝度）の存在を検出
        # 黒目領域: 輝度0-50
        # 白目領域: 輝度150-255
        # 中間領域: 輝度51-149
        
        dark_region = np.sum(hist[0:51])  # 黒目領域
        light_region = np.sum(hist[150:256])  # 白目領域
        mid_region = np.sum(hist[51:150])  # 中間領域
        
        # 理想的な目の輝度分布を定義
        # 黒目と白目の両方が存在し、適度なコントラストがある場合に高スコア
        score = 0.0
        
        # 黒目領域の評価（適度な黒目がある場合に高スコア）
        if 0.15 <= dark_region <= 0.35:
            score += 0.4
        elif 0.08 <= dark_region <= 0.45:
            score += 0.2
        else:
            score += 0.0
        
        # 白目領域の評価（適度な白目がある場合に高スコア）
        if 0.25 <= light_region <= 0.55:
            score += 0.4
        elif 0.15 <= light_region <= 0.65:
            score += 0.2
        else:
            score += 0.0
        
        # コントラストの評価（黒目と白目の差が大きい場合に高スコア）
        contrast = abs(dark_region - light_region)
        if contrast > 0.25:
            score += 0.2
        elif contrast > 0.15:
            score += 0.1
        else:
            score += 0.0
        
        return min(1.0, score)
        
    except Exception as e:
        return 0.1

def analyze_eye_periphery_contrast(gray_frame, face_landmarks, image_size):
    """
    目領域の外側のコントラスト差を分析して、サングラスなどの遮蔽を検出
    """
    try:
        # 左右の目の領域を定義
        left_eye_landmarks = [
            33,   # 左目中心
            159,  # 左上まぶた
            145,  # 左下まぶた
            133,  # 左目外角
            157,  # 左目内角
            158   # 左目内角
        ]
        
        right_eye_landmarks = [
            263,  # 右目中心
            386,  # 右上まぶた
            374,  # 右下まぶた
            362,  # 右目外角
            387,  # 右目内角
            388   # 右目内角
        ]
        
        # 左右の目の外側コントラストスコアを計算
        left_contrast_score = analyze_single_eye_periphery_contrast(gray_frame, face_landmarks, left_eye_landmarks, image_size)
        right_contrast_score = analyze_single_eye_periphery_contrast(gray_frame, face_landmarks, right_eye_landmarks, image_size)
        
        # 両目の平均スコアを返す
        return (left_contrast_score + right_contrast_score) / 2.0
        
    except Exception as e:
        # エラーが発生した場合は低い信頼度を返す
        return 0.3

def analyze_single_eye_periphery_contrast(gray_frame, face_landmarks, eye_landmarks, image_size):
    """
    単一の目の外側のコントラスト差を分析
    """
    try:
        # 目のランドマークから目の領域を定義
        eye_points = []
        for landmark_id in eye_landmarks:
            landmark = face_landmarks.landmark[landmark_id]
            x = int(landmark.x * image_size[1])
            y = int(landmark.y * image_size[0])
            eye_points.append([x, y])
        
        if len(eye_points) < 3:
            return 0.3
        
        # 目の領域を囲む矩形を計算
        eye_points = np.array(eye_points, dtype=np.int32)
        x, y, w, h = cv2.boundingRect(eye_points)
        
        # 目の領域を少し拡張
        margin = 3
        x = max(0, x - margin)
        y = max(0, y - margin)
        w = min(gray_frame.shape[1] - x, w + 2 * margin)
        h = min(gray_frame.shape[0] - y, h + 2 * margin)
        
        # 目の領域を抽出
        eye_region = gray_frame[y:y+h, x:x+w]
        
        if eye_region.size == 0:
            return 0.3
        
        # 目の外側の領域を定義（目の領域の周囲）
        outer_margin = 15
        outer_x = max(0, x - outer_margin)
        outer_y = max(0, y - outer_margin)
        outer_w = min(gray_frame.shape[1] - outer_x, w + 2 * outer_margin)
        outer_h = min(gray_frame.shape[0] - outer_y, h + 2 * outer_margin)
        
        # 外側の領域を抽出
        outer_region = gray_frame[outer_y:outer_y+outer_h, outer_x:outer_x+outer_w]
        
        if outer_region.size == 0:
            return 0.3
        
        # 目の領域と外側の領域の輝度統計を計算
        eye_mean = np.mean(eye_region)
        eye_std = np.std(eye_region)
        outer_mean = np.mean(outer_region)
        outer_std = np.std(outer_region)
        
        # コントラスト差を計算
        contrast_diff = abs(eye_mean - outer_mean)
        
        # サングラスなどの遮蔽を検出
        # コントラスト差が小さい場合（サングラスなどで遮蔽されている場合）は低スコア
        score = 1.0
        
        # コントラスト差が非常に小さい場合（サングラスの可能性）
        if contrast_diff < 10:
            score = 0.2
        elif contrast_diff < 20:
            score = 0.4
        elif contrast_diff < 30:
            score = 0.6
        elif contrast_diff < 40:
            score = 0.8
        else:
            score = 1.0
        
        # 目の領域の標準偏差が小さい場合（均一な色で遮蔽されている可能性）
        if eye_std < 5:
            score *= 0.5
        elif eye_std < 10:
            score *= 0.7
        elif eye_std < 15:
            score *= 0.9
        
        return max(0.1, min(1.0, score))
        
    except Exception as e:
        return 0.3

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
# ==============================================
# New confidence calculation helper (v2)
# ==============================================

def compute_confidence_v2(face_landmarks, image_size, frame,
                         left_eye_opening_ratio, right_eye_opening_ratio,
                         left_eye_px, right_eye_px, nose_tip_px, triangle_center,
                         eye_distance):
    """Compute confidence using revised deduction-based model with improved sunglasses detection."""
    import math
    import numpy as np

    # --- 1. Landmark visibility (緩和) --------------------------
    eye_landmarks_ids = [
        33, 263, 159, 145, 386, 374, 133, 362, 157, 158,
        387, 388, 160, 161, 163, 246, 247, 249,
        373, 374, 375, 380, 381, 382
    ]
    visible = 0
    z_list = []
    for lid in eye_landmarks_ids:
        lm = face_landmarks.landmark[lid]
        z_list.append(lm.z)
        if 0 <= lm.x <= 1 and 0 <= lm.y <= 1 and lm.z > 0:
            visible += 1
    missing = len(eye_landmarks_ids) - visible
    basic_visibility = visible / len(eye_landmarks_ids)

    # 緩和: 欠損ランドマークの減点を軽減
    deductions = missing * 0.03  # 0.07 → 0.03 に緩和

    # --- 2. Eye opening (緩和) ----------------------------------
    if left_eye_opening_ratio < 0.04 and right_eye_opening_ratio < 0.04:  # 0.06 → 0.04 に緩和
        deductions += 0.25  # 0.40 → 0.25 に緩和
    elif left_eye_opening_ratio < 0.04 or right_eye_opening_ratio < 0.04:
        deductions += 0.15  # 0.25 → 0.15 に緩和

    # --- 3. Eye angle asymmetry (緩和) --------------------------
    def _eye_angle(up_id, low_id):
        up = face_landmarks.landmark[up_id]
        lo = face_landmarks.landmark[low_id]
        dy = (lo.y - up.y) * image_size[0]
        dx = (lo.x - up.x) * image_size[1]
        return math.degrees(math.atan2(dy, dx))

    angle_left = _eye_angle(159, 145)
    angle_right = _eye_angle(386, 374)
    eye_angle_diff = abs(angle_left - angle_right)
    if eye_angle_diff > 20:  # 15 → 20 に緩和
        deductions += 0.10  # 0.20 → 0.10 に緩和

    # --- 4. Left / Right eye–nose distance ratio (緩和) --------
    left_nose = math.hypot(left_eye_px[0] - nose_tip_px[0], left_eye_px[1] - nose_tip_px[1])
    right_nose = math.hypot(right_eye_px[0] - nose_tip_px[0], right_eye_px[1] - nose_tip_px[1])
    lr_ratio = left_nose / right_nose if right_nose > 0 else 0
    if lr_ratio < 0.7 or lr_ratio > 1.3:  # 0.8/1.2 → 0.7/1.3 に緩和
        deductions += 0.08  # 0.15 → 0.08 に緩和

    # --- 5. Triangle shape ratio (緩和) -------------------------
    sides = sorted([eye_distance, left_nose, right_nose])
    triangle_shape_ratio = sides[0] / sides[2] if sides[2] > 0 else 0
    if triangle_shape_ratio < 0.20:  # 0.25 → 0.20 に緩和
        deductions += 0.08  # 0.15 → 0.08 に緩和

    # --- 6. Brightness histogram score (緩和) ------------------
    brightness_score = 1.0
    if frame is not None:
        brightness_score = analyze_eye_brightness(frame, face_landmarks, image_size)
    if brightness_score < 0.15:  # 0.25 → 0.15 に緩和
        deductions += 0.10  # 0.15 → 0.10 に緩和

    # --- 7. Z-coordinate dispersion (緩和) ---------------------
    z_std = float(np.std(z_list)) if z_list else 0.0
    if z_std > 0.15:  # 0.10 → 0.15 に緩和
        deductions += 0.08  # 0.15 → 0.08 に緩和

    # --- 8. NEW: Eye periphery contrast analysis --------------
    periphery_contrast_score = 1.0
    if frame is not None:
        periphery_contrast_score = analyze_eye_periphery_contrast(frame, face_landmarks, image_size)
    
    # サングラスなどの遮蔽検出（新しい要素）
    if periphery_contrast_score < 0.3:  # コントラスト差が非常に小さい場合
        deductions += 0.20  # サングラス検出時は大幅減点
    elif periphery_contrast_score < 0.5:
        deductions += 0.10
    elif periphery_contrast_score < 0.7:
        deductions += 0.05

    # --- Final confidence (緩和) --------------------------------
    confidence = 1.0 - deductions
    confidence = max(0.15, min(1.0, confidence))  # 最低信頼度を0.05 → 0.15 に上げる

    # 最終的な信頼度補正（緩和）
    if confidence < 0.3:
        confidence *= 0.9  # 0.7 → 0.9 に緩和
    elif confidence > 0.7:
        confidence *= 1.1  # 高信頼度の場合はさらに上げる

    details = {
        'basic_visibility': basic_visibility,
        'missing_landmarks': missing,
        'left_eye_opening_ratio': left_eye_opening_ratio,
        'right_eye_opening_ratio': right_eye_opening_ratio,
        'eye_angle_diff': eye_angle_diff,
        'lr_distance_ratio': lr_ratio,
        'triangle_shape_ratio': triangle_shape_ratio,
        'brightness_score': brightness_score,
        'periphery_contrast_score': periphery_contrast_score,
        'z_std': z_std,
        'deductions': deductions,
        'final_confidence': confidence
    }
    return confidence, details

# ==============================================

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

    # ---------------- New Confidence Logic (v2) ----------------
    confidence, confidence_details = compute_confidence_v2(
        face_landmarks=face_landmarks,
        image_size=image_size,
        frame=frame,
        left_eye_opening_ratio=left_eye_opening_ratio,
        right_eye_opening_ratio=right_eye_opening_ratio,
        left_eye_px=left_eye_px,
        right_eye_px=right_eye_px,
        nose_tip_px=nose_tip_px,
        triangle_center=triangle_center,
        eye_distance=eye_distance
    )

    # Calculate yaw (left/right) and pitch (up/down) based on nose-to-center distances before returning.
    yaw = np.clip(distance_x * 0.5, -45, 45)  # ±45 degrees
    pitch = np.clip(distance_y * 0.5, -45, 45)  # ±45 degrees

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

    return yaw, pitch, confidence, {
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
        'confidence_details': confidence_details
    }
    
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
    
    # 開瞼度を計算（信頼度計算に必要）
    left_eye_opening, right_eye_opening, nose_length = calculate_eye_opening(face_landmarks, image_size)
    left_eye_opening_ratio = left_eye_opening
    right_eye_opening_ratio = right_eye_opening
    
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
        
        # 基本的な可視性チェック（より厳密な条件に変更）
        if landmark.z > 0 and 0 <= landmark.x <= 1 and 0 <= landmark.y <= 1:
            eye_visibility_scores.append(1.0)
        else:
            eye_visibility_scores.append(0.0)
    
    # 基本的な可視性スコア
    basic_visibility = sum(eye_visibility_scores) / len(eye_visibility_scores)
    
    # 遮蔽検出の強化（より厳密に）
    # 1. z座標の異常値検出（遮蔽時はz座標が異常な値になることが多い）
    z_coordinates = np.array(z_coordinates)
    z_mean = np.mean(z_coordinates)
    z_std = np.std(z_coordinates)
    
    # z座標が異常に分散している場合は遮蔽の可能性（より厳密に）
    z_anomaly_score = 1.0 - min(1.0, z_std / 0.1)  # 標準偏差が0.1以上で異常（0.15から0.1に厳密化）
    
    # 2. 開瞼度による遮蔽検出（新しい要素）
    # 開瞼度が異常に低い場合は遮蔽の可能性
    eye_opening_score = 1.0
    if left_eye_opening_ratio < 0.03 or right_eye_opening_ratio < 0.03:  # 開瞼度が極端に低い場合
        eye_opening_score = 0.05  # 大幅に信頼度を下げる
    elif left_eye_opening_ratio < 0.08 or right_eye_opening_ratio < 0.08:  # 開瞼度が低い場合
        eye_opening_score = 0.2
    elif left_eye_opening_ratio < 0.12 or right_eye_opening_ratio < 0.12:  # 開瞼度がやや低い場合
        eye_opening_score = 0.5
    
    # 3. 左右の開瞼度の差による遮蔽検出（新しい要素）
    # 左右の開瞼度に大きな差がある場合は遮蔽の可能性
    eye_difference_score = 1.0
    eye_opening_diff = abs(left_eye_opening_ratio - right_eye_opening_ratio)
    if eye_opening_diff > 0.08:  # 左右の差が大きい場合
        eye_difference_score = 0.1  # 大幅に信頼度を下げる
    elif eye_opening_diff > 0.04:  # 左右の差がやや大きい場合
        eye_difference_score = 0.3
    
    # 4. 目の立体感検出（新しい要素）
    # 目の周辺のz座標の変化から立体感を検出
    eye_depth_score = 1.0
    
    # 目の中心と周辺のz座標の差を計算
    left_eye_center_z = face_landmarks.landmark[33].z
    right_eye_center_z = face_landmarks.landmark[263].z
    
    # 目の周辺のz座標の平均
    eye_periphery_z = np.mean([face_landmarks.landmark[i].z for i in [159, 145, 386, 374]])
    
    # 目の中心と周辺のz座標の差（立体感の指標）
    left_eye_depth = abs(left_eye_center_z - eye_periphery_z)
    right_eye_depth = abs(right_eye_center_z - eye_periphery_z)
    
    # 立体感が少ない場合は遮蔽の可能性
    if left_eye_depth < 0.015 and right_eye_depth < 0.015:  # 立体感が極端に少ない場合
        eye_depth_score = 0.05  # 大幅に信頼度を下げる
    elif left_eye_depth < 0.025 or right_eye_depth < 0.025:  # 立体感が少ない場合
        eye_depth_score = 0.2
    elif left_eye_depth < 0.035 or right_eye_depth < 0.035:  # 立体感がやや少ない場合
        eye_depth_score = 0.5
    
    # 5. 目の位置関係による遮蔽検出
    # 左右の目の位置関係が正常かチェック
    left_eye_landmark = face_landmarks.landmark[33]  # 左目
    right_eye_landmark = face_landmarks.landmark[263]  # 右目
    
    # 目の位置関係の妥当性（左右の目の距離が適切か）
    eye_position_score = 1.0
    if (0 <= left_eye_landmark.x <= 1 and 0 <= left_eye_landmark.y <= 1 and
        0 <= right_eye_landmark.x <= 1 and 0 <= right_eye_landmark.y <= 1):
        # 左右の目のx座標の差が適切な範囲内かチェック
        eye_x_diff = abs(left_eye_landmark.x - right_eye_landmark.x)
        if eye_x_diff < 0.1 or eye_x_diff > 0.4:  # 範囲を元に戻す
            eye_position_score = 0.5
    else:
        eye_position_score = 0.0
    
    # 6. 目付近の色変化・輝度変化の検出
    # 目の周辺のランドマークのz座標の変化から色変化・輝度変化を推定
    eye_variation_score = 1.0
    
    # z座標の変化量を計算（目付近の立体感・色変化の指標）
    z_variations = []
    for i in range(len(z_coordinates) - 1):
        z_variations.append(abs(z_coordinates[i] - z_coordinates[i+1]))
    
    if z_variations:
        avg_z_variation = np.mean(z_variations)
        # 適度な変化がある場合は高信頼度、変化が少ない場合は信頼度を下げる
        if avg_z_variation < 0.003:  # 変化が極端に少ない場合（閾値を厳密化）
            eye_variation_score = 0.05  # 大幅に信頼度を下げる
        elif avg_z_variation < 0.008:  # 変化が少ない場合
            eye_variation_score = 0.2
        elif avg_z_variation > 0.04:  # 変化が大きすぎる場合（ノイズの可能性）
            eye_variation_score = 0.6
        else:  # 適度な変化
            eye_variation_score = 1.0
    
    # 7. 輝度ヒストグラム分析による遮蔽検出（新しい要素）
    eye_brightness_score = 1.0
    if frame is not None:
        eye_brightness_score = analyze_eye_brightness(frame, face_landmarks, image_size)
    
    # 総合的な可視性スコア（新しい要素を追加）
    visibility_confidence = (
        basic_visibility * 0.15 +
        z_anomaly_score * 0.15 +
        eye_opening_score * 0.20 +  # 開瞼度の重みを増加
        eye_difference_score * 0.15 +  # 左右差の重みを増加
        eye_depth_score * 0.10 +
        eye_position_score * 0.05 +
        eye_brightness_score * 0.20  # 輝度ヒストグラム分析
    )
    
    # 遮蔽が検出された場合のみ大幅に信頼度を下げる（より厳密に）
    if (basic_visibility < 0.4 or z_anomaly_score < 0.3 or 
        eye_opening_score < 0.3 or eye_difference_score < 0.3 or 
        eye_depth_score < 0.3 or eye_brightness_score < 0.3):  # より厳密な条件
        visibility_confidence *= 0.1  # 削減率を厳密化（0.3から0.1に）
    
    # 4. 目の位置の対称性チェック（開瞼度は除外）
    # 左右の目の位置が対称かチェック
    left_eye_x = left_eye_landmark.x
    right_eye_x = right_eye_landmark.x
    left_eye_y = left_eye_landmark.y
    right_eye_y = right_eye_landmark.y
    
    # 目の位置の対称性（x座標の差とy座標の差から計算）
    eye_x_symmetry = 1.0 - abs(left_eye_x - right_eye_x) / max(abs(left_eye_x - right_eye_x), 0.001)
    eye_y_symmetry = 1.0 - abs(left_eye_y - right_eye_y) / max(abs(left_eye_y - right_eye_y), 0.001)
    
    # 対称性スコア（位置の対称性のみ）
    eye_symmetry = (eye_x_symmetry + eye_y_symmetry) / 2.0
    eye_symmetry = max(0.0, min(1.0, eye_symmetry))  # 0-1の範囲に制限
    
    # 5. 目の位置関係の妥当性チェック
    # 両目の距離が極端に近い、または遠い場合は信頼度を下げる
    eye_distance_ratio = eye_distance / image_size[1]  # 画像幅に対する比率
    if 0.05 <= eye_distance_ratio <= 0.25:  # 適切な範囲
        distance_confidence = 1.0
    else:
        # 範囲外の場合は距離に応じて信頼度を下げる
        distance_confidence = max(0.0, 1.0 - abs(eye_distance_ratio - 0.15) / 0.15)
    
    # 6. 三角形の形状の妥当性チェック
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
    
    # 7. 総合信頼度の計算（改善された集約方法）
    # 各信頼度要素の重み
    confidence_weights = {
        'area': 0.15,          # 三角形の面積
        'visibility': 0.35,    # ランドマークの可視性（重みを調整）
        'symmetry': 0.20,      # 目の対称性
        'distance': 0.15,      # 目の距離の妥当性
        'shape': 0.10,         # 三角形の形状
        'variation': 0.05      # 目付近の色変化・輝度変化
    }
    
    # 基本の重み付き平均
    weighted_confidence = (
        area_confidence * confidence_weights['area'] +
        visibility_confidence * confidence_weights['visibility'] +
        eye_symmetry * confidence_weights['symmetry'] +
        distance_confidence * confidence_weights['distance'] +
        shape_confidence * confidence_weights['shape'] +
        eye_variation_score * confidence_weights['variation']
    )
    
    # 信頼度の補正と集約（より厳密な条件）
    # 1. 可視性が低い場合のみ信頼度を下げる
    if visibility_confidence < 0.3:  # 閾値を0.4から0.3に調整
        weighted_confidence *= 0.5  # 削減率を調整（0.3から0.5に）
    
    # 2. 遮蔽が強く検出された場合のみ大幅に信頼度を下げる
    if (basic_visibility < 0.2 or z_anomaly_score < 0.1 or 
        eye_opening_score < 0.1 or eye_difference_score < 0.1 or 
        eye_depth_score < 0.1):  # より緩やかな条件
        weighted_confidence *= 0.2  # 削減率を調整（0.1から0.2に）
    
    # 3. 目の対称性が極端に悪い場合のみ信頼度を下げる
    if eye_symmetry < 0.2:  # 閾値を0.3から0.2に調整
        weighted_confidence *= 0.7  # 削減率を調整（0.5から0.7に）
    
    # 4. 三角形の形状が極端に歪んでいる場合のみ信頼度を下げる
    if shape_confidence < 0.1:  # 閾値を0.2から0.1に調整
        weighted_confidence *= 0.7  # 削減率を調整（0.5から0.7に）
    
    # 5. 距離が不適切な場合のみ信頼度を下げる
    if distance_confidence < 0.2:  # 閾値を0.3から0.2に調整
        weighted_confidence *= 0.8  # 削減率を調整（0.7から0.8に）
    
    # 6. 目付近の変化が少ない場合のみ信頼度を下げる
    if eye_variation_score < 0.1:  # 閾値を0.2から0.1に調整
        weighted_confidence *= 0.7  # 削減率を調整（0.5から0.7に）
    
    # 7. 輝度ヒストグラムが異常な場合のみ信頼度を下げる
    if eye_brightness_score < 0.1:  # 閾値を0.2から0.1に調整
        weighted_confidence *= 0.5  # 削減率を調整（0.3から0.5に）
    
    # 8. 最終的な信頼度を計算（より適切な補正）
    # 低信頼度の場合は適度に下げる、高信頼度の場合はそのまま評価
    if weighted_confidence < 0.2:  # 閾値を0.3から0.2に調整
        # 低信頼度の場合は適度に下げる
        confidence = weighted_confidence * 0.8  # 削減率を調整（0.5から0.8に）
    elif weighted_confidence > 0.7:  # 閾値を0.8から0.7に調整
        # 高信頼度の場合は少し上げる
        confidence = weighted_confidence * 1.1  # 増加率を調整
    else:
        # 中程度の信頼度はそのまま
        confidence = weighted_confidence
    
    # 信頼度を0-1の範囲に制限し、最低値を設定
    confidence = max(0.2, min(1.0, confidence))  # 最低値を0.1から0.2に調整
    
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
            'eye_position_score': eye_position_score,
            'eye_opening_score': eye_opening_score,  # 新しい要素を追加
            'eye_difference_score': eye_difference_score,  # 新しい要素を追加
            'eye_depth_score': eye_depth_score,  # 新しい要素を追加
            'eye_variation_score': eye_variation_score,
            'eye_brightness_score': eye_brightness_score,  # 輝度ヒストグラム分析
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
    yaw_arrow_length = min(abs(landmarks_info['distance_x']), 100) * np.sign(landmarks_info['distance_x'])
    yaw_direction = np.array([yaw_arrow_length, 0])
    yaw_end = tuple(map(int, np.array(base_point) + yaw_direction))
    
    # Pitch（上下）の矢印（Y軸方向に制限）
    # 上向きを正、下向きを負にするため、Y軸の符号を反転
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
            # 顔検出の信頼度を取得（MediaPipeの内部信頼度）
            face_detection_confidence = 1.0  # デフォルト値
            
            # 複数の顔が検出された場合は最も信頼度の高い顔を選択
            best_face_landmarks = None
            best_confidence = 0.0
            
            for face_landmarks in results.multi_face_landmarks:
                # 簡易的な顔検出信頼度を計算（ランドマークの安定性から推定）
                key_landmarks = [1, 33, 263, 61, 291]  # 鼻先、左右目、左右口角
                landmark_confidence = 0.0
                valid_landmarks = 0
                
                for landmark_id in key_landmarks:
                    landmark = face_landmarks.landmark[landmark_id]
                    # 座標が有効範囲内でz座標が適切な場合
                    if (0 <= landmark.x <= 1 and 0 <= landmark.y <= 1 and 
                        landmark.z > 0 and landmark.z < 1):
                        valid_landmarks += 1
                
                # 有効なランドマークの割合で信頼度を計算
                landmark_confidence = valid_landmarks / len(key_landmarks)
                
                if landmark_confidence > best_confidence:
                    best_confidence = landmark_confidence
                    best_face_landmarks = face_landmarks
            
            # 最も信頼度の高い顔を使用
            if best_face_landmarks is not None:
                face_landmarks = best_face_landmarks
                face_detection_confidence = best_confidence
                # 開瞼度計算（瞼間距離と鼻の長さによる）
                left_eye_opening, right_eye_opening, nose_length = calculate_eye_opening(face_landmarks, frame.shape[:2])
                
                # 簡易的な顔の向き推定
                yaw, pitch, confidence, landmarks_info = estimate_pose_simple(face_landmarks, frame.shape[:2], frame)
                
                # 顔検出の信頼度を最終的な信頼度に組み込み
                final_confidence = confidence * face_detection_confidence
                
                # 結果表示
                cv2.putText(frame, f"Left Eye Opening: {left_eye_opening:.3f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(frame, f"Right Eye Opening: {right_eye_opening:.3f}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(frame, f"Nose Length: {nose_length:.1f}px", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(frame, f"Face Detection: {face_detection_confidence:.2f}", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(frame, f"Final Confidence: {final_confidence:.2f}", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(frame, f"Yaw: {yaw:.1f}deg, Pitch: {pitch:.1f}deg", (10, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(frame, f"Normal Vector: {landmarks_info['normal_vector']:.3f}", (10, 210), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(frame, f"Distance X: {landmarks_info.get('distance_x', 0):.1f}, Y: {landmarks_info.get('distance_y', 0):.1f}", (10, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(frame, f"Periphery Contrast: {landmarks_info.get('confidence_details', {}).get('periphery_contrast_score', 0):.2f}", (10, 270), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                # 色分けされたランドマーク描画
                draw_landmarks_with_colors(frame, face_landmarks, frame.shape[:2])
                
                # 顔の向きを矢印で描画
                draw_pose_arrow(frame, yaw, pitch, final_confidence, landmarks_info)
                
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