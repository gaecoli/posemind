import cv2
import mediapipe as mp
import numpy as np
from typing import List, Tuple
import os
from PIL import Image, ImageDraw, ImageFont
from ultralytics import YOLO

class PoseAnalyzer:
    def __init__(self):
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
            model_complexity=2
        )
        self.mp_draw = mp.solutions.drawing_utils
        self.font_path = "/System/Library/Fonts/PingFang.ttc"
        if not os.path.exists(self.font_path):
            self.font_path = "/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf"
        self.font_size = 20
        self.font = ImageFont.truetype(self.font_path, self.font_size)
        self.min_person_size = 0.2
        self.max_person_size = 0.8
        self.yolo_model = YOLO('models/yolov8n.pt')
        self.conf_threshold = 0.5

    def calculate_angle_3d(self, a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
        v1 = b - a
        v2 = c - b
        dot_product = np.dot(v1, v2)
        norm_v1 = np.linalg.norm(v1)
        norm_v2 = np.linalg.norm(v2)
        if norm_v1 == 0 or norm_v2 == 0:
            return 0
        cos_angle = dot_product / (norm_v1 * norm_v2)
        cos_angle = np.clip(cos_angle, -1.0, 1.0)
        return np.degrees(np.arccos(cos_angle))

    def detect_body_facing(self, landmarks) -> str:
        """改进的身体朝向检测：front（正面）、back（背面）或side（侧面）"""
        left_shoulder = landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER.value]
        right_shoulder = landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
        left_hip = landmarks[self.mp_pose.PoseLandmark.LEFT_HIP.value]
        right_hip = landmarks[self.mp_pose.PoseLandmark.RIGHT_HIP.value]
        nose = landmarks[self.mp_pose.PoseLandmark.NOSE.value]
        
        # 计算左右肩膀和髋部的Z值差异
        z_diff_shoulders = left_shoulder.z - right_shoulder.z
        z_diff_hips = left_hip.z - right_hip.z
        
        # 计算左右肩膀的可见性比率
        shoulder_visibility_ratio = left_shoulder.visibility / (right_shoulder.visibility + 0.001)
        
        # 计算左右肩膀的水平距离
        shoulder_distance = abs(left_shoulder.x - right_shoulder.x)
        
        # 计算左右肩膀的x坐标差异（用于侧面检测）
        x_diff_shoulders = abs(left_shoulder.x - right_shoulder.x)
        
        # 计算鼻子与肩膀中点的x坐标差异（用于侧面检测）
        shoulder_midpoint_x = (left_shoulder.x + right_shoulder.x) / 2
        nose_offset = abs(nose.x - shoulder_midpoint_x)
        
        # 侧面判断：肩膀水平距离明显缩短，且一侧肩膀可见性明显高于另一侧
        if (shoulder_distance < 0.15 or # 肩膀水平距离小
            (shoulder_visibility_ratio > 2.0 or shoulder_visibility_ratio < 0.5) or # 一侧肩膀可见性明显高于另一侧
            nose_offset > 0.15): # 鼻子不在肩膀中点附近
            return "side"
        
        # 背面判断：Z值差异较大
        facing_score = (z_diff_shoulders + z_diff_hips) / 2
        if facing_score > 0.05:
            return "back"
            
        # 默认为正面
        return "front"

    def detect_side_direction(self, landmarks) -> str:
        """检测侧面时身体朝向的方向：left或right"""
        nose = landmarks[self.mp_pose.PoseLandmark.NOSE.value]
        left_shoulder = landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER.value]
        right_shoulder = landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
        
        # 计算鼻子相对于肩膀中点的位置
        shoulder_midpoint_x = (left_shoulder.x + right_shoulder.x) / 2
        
        # 如果鼻子在肩膀中点左侧，则身体朝向右侧；否则朝向左侧
        return "right" if nose.x < shoulder_midpoint_x else "left"
    
    def detect_shooting_arm_side(self, landmarks, side_direction: str) -> str:
        """侧面拍摄时的投篮手检测"""
        left_wrist = landmarks[self.mp_pose.PoseLandmark.LEFT_WRIST.value]
        right_wrist = landmarks[self.mp_pose.PoseLandmark.RIGHT_WRIST.value]
        left_elbow = landmarks[self.mp_pose.PoseLandmark.LEFT_ELBOW.value]
        right_elbow = landmarks[self.mp_pose.PoseLandmark.RIGHT_ELBOW.value]
        left_shoulder = landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER.value]
        right_shoulder = landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
        
        # 转换为3D坐标
        left_wrist_pos = np.array([left_wrist.x, left_wrist.y, left_wrist.z])
        right_wrist_pos = np.array([right_wrist.x, right_wrist.y, right_wrist.z])
        left_elbow_pos = np.array([left_elbow.x, left_elbow.y, left_elbow.z])
        right_elbow_pos = np.array([right_elbow.x, right_elbow.y, right_elbow.z])
        left_shoulder_pos = np.array([left_shoulder.x, left_shoulder.y, left_shoulder.z])
        right_shoulder_pos = np.array([right_shoulder.x, right_shoulder.y, right_shoulder.z])
        
        # 计算肘部角度
        left_elbow_angle = self.calculate_angle_3d(
            left_shoulder_pos, 
            left_elbow_pos, 
            left_wrist_pos)
        right_elbow_angle = self.calculate_angle_3d(
            right_shoulder_pos, 
            right_elbow_pos, 
            right_wrist_pos)
        
        # 计算可见性评分
        left_visibility = (left_wrist.visibility + left_elbow.visibility) / 2
        right_visibility = (right_wrist.visibility + right_elbow.visibility) / 2
        
        # 根据侧面朝向判断
        if side_direction == "left":  # 身体朝向左侧
            # 朝向左侧时，右手更可能是投篮手（因为更靠近相机）
            if right_visibility > 0.5:
                return "right"
            else:
                return "left"
        else:  # 身体朝向右侧
            # 朝向右侧时，左手更可能是投篮手
            if left_visibility > 0.5:
                return "left"
            else:
                return "right"

    def detect_shooting_arm(self, landmarks) -> str:
        """改进的投篮手臂检测，支持正面、侧面和背面"""
        # 获取身体朝向
        body_facing = self.detect_body_facing(landmarks)
        
        # 根据不同朝向使用不同的检测逻辑
        if body_facing == "side":
            # 侧面朝向，先确定侧面方向
            side_direction = self.detect_side_direction(landmarks)
            return self.detect_shooting_arm_side(landmarks, side_direction)
        elif body_facing == "back":
            # 背对镜头时的逻辑
            left_wrist = landmarks[self.mp_pose.PoseLandmark.LEFT_WRIST.value]
            right_wrist = landmarks[self.mp_pose.PoseLandmark.RIGHT_WRIST.value]
            left_elbow = landmarks[self.mp_pose.PoseLandmark.LEFT_ELBOW.value]
            right_elbow = landmarks[self.mp_pose.PoseLandmark.RIGHT_ELBOW.value]
            
            # 计算肘部角度
            left_shoulder = landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER.value]
            right_shoulder = landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
            left_shoulder_pos = np.array([left_shoulder.x, left_shoulder.y, left_shoulder.z])
            right_shoulder_pos = np.array([right_shoulder.x, right_shoulder.y, right_shoulder.z])
            left_elbow_pos = np.array([left_elbow.x, left_elbow.y, left_elbow.z])
            right_elbow_pos = np.array([right_elbow.x, right_elbow.y, right_elbow.z])
            left_wrist_pos = np.array([left_wrist.x, left_wrist.y, left_wrist.z])
            right_wrist_pos = np.array([right_wrist.x, right_wrist.y, right_wrist.z])
            
            left_elbow_angle = self.calculate_angle_3d(left_shoulder_pos, left_elbow_pos, left_wrist_pos)
            right_elbow_angle = self.calculate_angle_3d(right_shoulder_pos, right_elbow_pos, right_wrist_pos)
            
            # 考虑肘部角度，投篮手臂肘部角度通常更小
            if (left_elbow_angle < right_elbow_angle - 20):
                return "left"
            if (right_elbow_angle < left_elbow_angle - 20):
                return "right"
            
            # 通过水平位置判断：背面投篮时，右手在图像的左侧，左手在图像的右侧
            if left_wrist.x < right_wrist.x:
                return "right"  # 可能是右手投篮（因为背对镜头，位置反转）
            else:
                return "left"
        else:
            # 正面朝向时的逻辑
            left_wrist = landmarks[self.mp_pose.PoseLandmark.LEFT_WRIST.value]
            right_wrist = landmarks[self.mp_pose.PoseLandmark.RIGHT_WRIST.value]
            left_elbow = landmarks[self.mp_pose.PoseLandmark.LEFT_ELBOW.value]
            right_elbow = landmarks[self.mp_pose.PoseLandmark.RIGHT_ELBOW.value]
            
            # 计算可见性与肘部角度综合评分
            left_shoulder = landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER.value]
            right_shoulder = landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
            left_shoulder_pos = np.array([left_shoulder.x, left_shoulder.y, left_shoulder.z])
            right_shoulder_pos = np.array([right_shoulder.x, right_shoulder.y, right_shoulder.z])
            left_elbow_pos = np.array([left_elbow.x, left_elbow.y, left_elbow.z])
            right_elbow_pos = np.array([right_elbow.x, right_elbow.y, right_elbow.z])
            left_wrist_pos = np.array([left_wrist.x, left_wrist.y, left_wrist.z])
            right_wrist_pos = np.array([right_wrist.x, right_wrist.y, right_wrist.z])
            
            left_elbow_angle = self.calculate_angle_3d(left_shoulder_pos, left_elbow_pos, left_wrist_pos)
            right_elbow_angle = self.calculate_angle_3d(right_shoulder_pos, right_elbow_pos, right_wrist_pos)
            
            if left_wrist.visibility > right_wrist.visibility and left_elbow_angle < right_elbow_angle:
                return "left"
            return "right"

    def get_landmark_points_3d(self, landmarks, side: str) -> dict:
        if side == "left":
            return {
                "shoulder": np.array([landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                                      landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER.value].y,
                                      landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER.value].z]),
                "elbow": np.array([landmarks[self.mp_pose.PoseLandmark.LEFT_ELBOW.value].x,
                                   landmarks[self.mp_pose.PoseLandmark.LEFT_ELBOW.value].y,
                                   landmarks[self.mp_pose.PoseLandmark.LEFT_ELBOW.value].z]),
                "wrist": np.array([landmarks[self.mp_pose.PoseLandmark.LEFT_WRIST.value].x,
                                   landmarks[self.mp_pose.PoseLandmark.LEFT_WRIST.value].y,
                                   landmarks[self.mp_pose.PoseLandmark.LEFT_WRIST.value].z])
            }
        else:
            return {
                "shoulder": np.array([landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,
                                      landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y,
                                      landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value].z]),
                "elbow": np.array([landmarks[self.mp_pose.PoseLandmark.RIGHT_ELBOW.value].x,
                                   landmarks[self.mp_pose.PoseLandmark.RIGHT_ELBOW.value].y,
                                   landmarks[self.mp_pose.PoseLandmark.RIGHT_ELBOW.value].z]),
                "wrist": np.array([landmarks[self.mp_pose.PoseLandmark.RIGHT_WRIST.value].x,
                                   landmarks[self.mp_pose.PoseLandmark.RIGHT_WRIST.value].y,
                                   landmarks[self.mp_pose.PoseLandmark.RIGHT_WRIST.value].z])
            }

    def is_landmark_visible(self, landmark) -> bool:
        return landmark.visibility > 0.5

    def get_knee_angle(self, landmarks) -> float:
        left_hip = landmarks[self.mp_pose.PoseLandmark.LEFT_HIP.value]
        left_knee = landmarks[self.mp_pose.PoseLandmark.LEFT_KNEE.value]
        left_ankle = landmarks[self.mp_pose.PoseLandmark.LEFT_ANKLE.value]
        right_hip = landmarks[self.mp_pose.PoseLandmark.RIGHT_HIP.value]
        right_knee = landmarks[self.mp_pose.PoseLandmark.RIGHT_KNEE.value]
        right_ankle = landmarks[self.mp_pose.PoseLandmark.RIGHT_ANKLE.value]

        if self.is_landmark_visible(left_hip) and self.is_landmark_visible(left_knee) and self.is_landmark_visible(left_ankle):
            hip = np.array([left_hip.x, left_hip.y, left_hip.z])
            knee = np.array([left_knee.x, left_knee.y, left_knee.z])
            ankle = np.array([left_ankle.x, left_ankle.y, left_ankle.z])
        elif self.is_landmark_visible(right_hip) and self.is_landmark_visible(right_knee) and self.is_landmark_visible(right_ankle):
            hip = np.array([right_hip.x, right_hip.y, right_hip.z])
            knee = np.array([right_knee.x, right_knee.y, right_knee.z])
            ankle = np.array([right_ankle.x, right_ankle.y, right_ankle.z])
        else:
            return 0
        return self.calculate_angle_3d(hip, knee, ankle)

    def save_keyframe(self, frame: np.ndarray, output_dir: str, frame_count: int, description: str):
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        output_path = os.path.join(output_dir, f"keyframe_{frame_count:04d}_{description}.jpg")
        cv2.imwrite(output_path, frame)

    def put_chinese_text(self, img: np.ndarray, text: str, position: Tuple[int, int], color: Tuple[int, int, int]) -> np.ndarray:
        img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(img_pil)
        draw.text(position, text, font=self.font, fill=color)
        return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

    def detect_shooting_start(self, landmarks, shooting_arm: str) -> bool:
        arm_points = self.get_landmark_points_3d(landmarks, shooting_arm)
        elbow_angle = self.calculate_angle_3d(arm_points["shoulder"], arm_points["elbow"], arm_points["wrist"])
        return (elbow_angle < 100 and 
                arm_points["wrist"][1] < arm_points["shoulder"][1])

    def detect_shooting_end(self, landmarks, shooting_arm: str) -> bool:
        arm_points = self.get_landmark_points_3d(landmarks, shooting_arm)
        elbow_angle = self.calculate_angle_3d(arm_points["shoulder"], arm_points["elbow"], arm_points["wrist"])
        return (elbow_angle > 150 and 
                arm_points["wrist"][1] > arm_points["elbow"][1])

    def detect_person_yolo(self, frame: np.ndarray) -> List[Tuple[int, int, int, int]]:
        results = self.yolo_model(frame, conf=self.conf_threshold)
        person_boxes = []
        for result in results:
            boxes = result.boxes
            for box in boxes:
                if box.cls == 0:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    person_boxes.append((int(x1), int(y1), int(x2), int(y2)))
        return person_boxes

    def get_person_size(self, landmarks) -> float:
        if not landmarks:
            return 0
        nose = landmarks[self.mp_pose.PoseLandmark.NOSE.value]
        left_ankle = landmarks[self.mp_pose.PoseLandmark.LEFT_ANKLE.value]
        right_ankle = landmarks[self.mp_pose.PoseLandmark.RIGHT_ANKLE.value]
        ankle_y = (left_ankle.y + right_ankle.y) / 2
        height = abs(nose.y - ankle_y)
        return height

    def is_person_valid(self, landmarks, frame_height: int) -> bool:
        if not landmarks:
            return False
        person_size = self.get_person_size(landmarks)
        if person_size < 0.1 or person_size > 0.9:
            return False
        required_landmarks = [
            self.mp_pose.PoseLandmark.NOSE,
            self.mp_pose.PoseLandmark.LEFT_SHOULDER,
            self.mp_pose.PoseLandmark.RIGHT_SHOULDER,
            self.mp_pose.PoseLandmark.LEFT_ELBOW,
            self.mp_pose.PoseLandmark.RIGHT_ELBOW,
            self.mp_pose.PoseLandmark.LEFT_WRIST,
            self.mp_pose.PoseLandmark.RIGHT_WRIST
        ]
        visible_count = sum(1 for landmark in required_landmarks if self.is_landmark_visible(landmarks[landmark.value]))
        return visible_count >= len(required_landmarks) * 0.7

    def analyze_shooting(self, video_path: str) -> Tuple[str, List[float]]:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return "无法打开视频文件", []

        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # 调整视频尺寸为更大的尺寸
        target_width = 1280
        target_height = 720
        scale = min(target_width / frame_width, target_height / frame_height)
        frame_width = int(frame_width * scale)
        frame_height = int(frame_height * scale)
        
        output_fps = max(1, fps // 3)
        output_dir = "output_videos"
        keyframes_dir = "keyframes"
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(keyframes_dir, exist_ok=True)

        output_path = os.path.join(output_dir, "analyzed_shooting.mp4")
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, output_fps, (frame_width, frame_height))

        scores = []
        is_shooting = False
        frame_count = 0
        last_keyframe = 0
        keyframe_interval = 30
        shooting_arm = None
        facing_history = []  # 记录朝向历史
        side_direction_history = []  # 记录侧面方向历史
        shooting_start_frame = 0
        shooting_end_frame = 0
        min_shooting_duration = 5  # 减少最小持续时间
        max_shooting_duration = 90  # 增加最大持续时间
        last_valid_frame = 0
        max_invalid_frames = 20  # 增加允许的无效帧数

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.resize(frame, (frame_width, frame_height))
            
            # 使用YOLOv7检测人物
            person_boxes = self.detect_person_yolo(frame)
            
            # 如果检测到多个人物，选择最大的人物框
            if person_boxes:
                # 计算每个人物框的面积
                areas = [(box[2] - box[0]) * (box[3] - box[1]) for box in person_boxes]
                max_area_idx = np.argmax(areas)
                target_box = person_boxes[max_area_idx]
                
                # 裁剪目标人物区域，并扩大裁剪范围
                x1, y1, x2, y2 = target_box
                margin = 50  # 添加边距
                x1 = max(0, x1 - margin)
                y1 = max(0, y1 - margin)
                x2 = min(frame_width, x2 + margin)
                y2 = min(frame_height, y2 + margin)
                person_frame = frame[y1:y2, x1:x2]
                
                # 确保裁剪区域有效
                if person_frame.size == 0 or person_frame.shape[0] == 0 or person_frame.shape[1] == 0:
                    frame_count += 1
                    continue
                
                # 在原始帧上绘制人物框
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
                # 处理裁剪后的人物帧
                rgb_frame = cv2.cvtColor(person_frame, cv2.COLOR_BGR2RGB)
                results = self.pose.process(rgb_frame)

                if results.pose_landmarks:
                    landmarks = results.pose_landmarks.landmark
                    
                    # 检查人物是否在有效范围内
                    if not self.is_person_valid(landmarks, frame_height):
                        frame_count += 1
                        if frame_count - last_valid_frame > max_invalid_frames:
                            is_shooting = False
                            shooting_start_frame = 0
                            shooting_end_frame = 0
                            shooting_arm = None
                        continue
                    
                    last_valid_frame = frame_count
                    annotated_frame = frame.copy()
                    
                    # 将姿态关键点绘制在原始帧上
                    self.mp_draw.draw_landmarks(
                        annotated_frame,
                        results.pose_landmarks,
                        self.mp_pose.POSE_CONNECTIONS,
                        landmark_drawing_spec=self.mp_draw.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=3),
                        connection_drawing_spec=self.mp_draw.DrawingSpec(color=(0, 255, 0), thickness=2)
                    )

                    # 检测身体朝向，并添加到历史记录
                    current_facing = self.detect_body_facing(landmarks)
                    facing_history.append(current_facing)
                    if len(facing_history) > 10:  # 保持历史记录的长度
                        facing_history.pop(0)
                    
                    # 使用历史记录中出现次数最多的朝向作为当前朝向
                    stable_facing = max(set(facing_history), key=facing_history.count)
                    
                    # 如果是侧面，还需要检测侧面方向
                    side_direction = ""
                    if stable_facing == "side":
                        current_side_direction = self.detect_side_direction(landmarks)
                        side_direction_history.append(current_side_direction)
                        if len(side_direction_history) > 10:
                            side_direction_history.pop(0)
                        side_direction = max(set(side_direction_history), key=side_direction_history.count)
                    
                    # 如果投篮手未确定或需要重新确定
                    if shooting_arm is None or frame_count % 30 == 0:  # 每30帧重新评估
                        shooting_arm = self.detect_shooting_arm(landmarks)

                    # 检测投篮动作的开始和结束
                    if not is_shooting and self.detect_shooting_start(landmarks, shooting_arm):
                        is_shooting = True
                        shooting_start_frame = frame_count
                        self.save_keyframe(annotated_frame, keyframes_dir, frame_count, "准备姿势")
                    elif is_shooting and self.detect_shooting_end(landmarks, shooting_arm):
                        shooting_duration = frame_count - shooting_start_frame
                        if min_shooting_duration <= shooting_duration <= max_shooting_duration:
                            is_shooting = False
                            shooting_end_frame = frame_count
                            self.save_keyframe(annotated_frame, keyframes_dir, frame_count, "出手姿势")
                        else:
                            is_shooting = False
                            shooting_start_frame = 0
                            shooting_end_frame = 0

                    if is_shooting:
                        arm_points = self.get_landmark_points_3d(landmarks, shooting_arm)
                        elbow_angle = self.calculate_angle_3d(arm_points["shoulder"], arm_points["elbow"], arm_points["wrist"])
                        knee_angle = self.get_knee_angle(landmarks)

                        head = np.array([landmarks[self.mp_pose.PoseLandmark.NOSE.value].x,
                                         landmarks[self.mp_pose.PoseLandmark.NOSE.value].y,
                                         landmarks[self.mp_pose.PoseLandmark.NOSE.value].z])
                        release_height = np.linalg.norm(arm_points["wrist"] - head)

                        wrist_elbow_vector = arm_points["wrist"] - arm_points["elbow"]
                        horizontal_vector = np.array([wrist_elbow_vector[0], wrist_elbow_vector[1], 0])
                        release_angle = self.calculate_angle_3d(wrist_elbow_vector, horizontal_vector, np.array([0, 0, 1]))

                        elbow_score = 1 - abs(elbow_angle - 90) / 90
                        height_score = min(1.0, release_height * 2)
                        angle_score = 1 - abs(release_angle - 52.5) / 52.5
                        if knee_angle > 0:
                            knee_score = 1 - abs(knee_angle - 135) / 135
                            score = (elbow_score + knee_score + height_score + angle_score) / 4
                        else:
                            score = (elbow_score + height_score + angle_score) / 3
                        scores.append(score)

                        if frame_count - last_keyframe >= keyframe_interval:
                            self.save_keyframe(annotated_frame, keyframes_dir, frame_count, "动作过程")
                            last_keyframe = frame_count

                    # 添加标签
                    label_texts = [
                        f"状态: {'投篮中' if is_shooting else '等待投篮'}",
                        f"投篮手: {'左手' if shooting_arm == 'left' else '右手'}",
                        f"朝向: {stable_facing}" + (f" ({side_direction})" if stable_facing == "side" else ""),
                        f"进度: {frame_count}/{total_frames}",
                        f"已检测投篮: {len(scores)}次"
                    ]
                    if is_shooting:
                        label_texts.extend([
                            f"肘角: {elbow_angle:.1f}°",
                            f"膝盖: {knee_angle:.1f}°" if knee_angle > 0 else "膝盖: 不可见",
                            f"出手角: {release_angle:.1f}°"
                        ])

                    for i, text in enumerate(label_texts):
                        annotated_frame = self.put_chinese_text(
                            annotated_frame,
                            text,
                            (10, 10 + i * 25),
                            (255, 255, 255)
                        )

                    out.write(annotated_frame)

            frame_count += 1

        cap.release()
        out.release()

        if not scores:
            return "未检测到有效的投篮动作", []

        avg_score = np.mean(scores)
        quality = "标准" if avg_score > 0.5 else "需要改进" if avg_score > 0.3 else "不规范"
        feedback = []
        
        # 记录最后一帧的数据用于生成反馈
        if is_shooting and landmarks:
            # 肘部角度反馈
            if elbow_angle < 80:
                feedback.append("手肘弯曲度不够，投篮时手肘需要稍微弯曲一些")
            elif elbow_angle > 100:
                feedback.append("手肘弯曲过度，投篮时手肘需要稍微打开一些")
            
            # 膝盖角度反馈
            if knee_angle > 0:
                if knee_angle < 120:
                    feedback.append("膝盖弯曲过度，需要稍微伸直一些")
                elif knee_angle > 150:
                    feedback.append("膝盖弯曲不够，起跳前需要稍微下蹲一些")
            
            # 出手角度反馈
            if release_angle < 45:
                feedback.append("出手角度太平，球需要抛得更高一些")
            elif release_angle > 60:
                feedback.append("出手角度太陡，球需要抛得平一些")
            
            # 手腕位置反馈
            if wrist_elbow_vector[1] > 0:  # 手腕在肘部下方
                feedback.append("手腕抬得不够高，出手点应在额头上方")
        elif len(scores) > 0:
            # 如果没有当前帧的数据，但有分数记录，给出一般性建议
            if avg_score <= 0.5:
                feedback.append("投篮姿势需要改进，注意手肘弯曲度和出手点位置")
                feedback.append("保持身体平衡，投篮时上半身挺直")
                feedback.append("出手时手腕应充分伸展，保持流畅的跟随动作")
        
        return f"检测到{len(scores)}次投篮，平均质量：{quality}{'，建议：' + '；'.join(feedback) if feedback else ''}", scores