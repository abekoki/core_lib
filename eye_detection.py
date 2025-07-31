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
    
    # 信頼度の計算
    # ランドマークの安定性と三角形の形状から計算
    # 三角形の面積が大きいほど信頼度が高い
    triangle_area = abs((left_eye_px[0] * (right_eye_px[1] - nose_tip_px[1]) + 
                        right_eye_px[0] * (nose_tip_px[1] - left_eye_px[1]) + 
                        nose_tip_px[0] * (left_eye_px[1] - right_eye_px[1])) / 2)
    
    # 面積を正規化（顔の大きさに依存しないように）
    normalized_area = triangle_area / (eye_distance ** 2) if eye_distance > 0 else 0
    confidence = min(normalized_area * 10, 1.0)  # 0-1の範囲に正規化
    
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
        'pitch_arrow_length': pitch_arrow_length
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

# メイン処理
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