import cv2
import mediapipe as mp
import numpy as np
from typing import List, Tuple
import os
from PIL import Image, ImageDraw, ImageFont
from ultralytics import YOLO
import torch

class PoseAnalyzer:
    def __init__(self):
        # 检测可用的设备
        self.device = os.environ.get('TORCH_DEVICE', 'cpu')
        if self.device == 'cuda' and torch.cuda.is_available():
            self.device = 'cuda'
        else:
            self.device = 'cpu'
            
        print(f"使用设备: {self.device}")
        
        # 初始化MediaPipe
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
            model_complexity=2
        )
        self.mp_draw = mp.solutions.drawing_utils
        
        # 加载字体
        self.font_path = "/System/Library/Fonts/PingFang.ttc"
        if not os.path.exists(self.font_path):
            self.font_path = "/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf"
        self.font_size = 20
        self.font = ImageFont.truetype(self.font_path, self.font_size)
        
        # 初始化YOLO模型
        self.yolo_model = YOLO('models/yolov8n-pose.pt')  # 使用YOLOv8姿态估计模型
        self.ball_model = YOLO('models/yolov8n.pt')  # 使用YOLOv8物体检测模型
        if self.device == 'cuda':
            self.yolo_model.to('cuda')
            self.ball_model.to('cuda')
        self.conf_threshold = 0.5
        
        # 性能优化参数
        self.min_person_size = 0.2
        self.max_person_size = 0.8
        self.frame_skip = 2  # 每处理2帧跳过1帧
        
        # 篮球检测参数
        self.ball_tracking_history = []  # 存储球的运动轨迹
        self.max_ball_history = 10  # 最大轨迹长度
        self.ball_detection_threshold = 0.2  # 降低篮球检测置信度阈值
        self.min_ball_movement = 0.05  # 降低最小球移动距离阈值
        self.shooting_ball_speed = 0.1  # 降低投篮时球的移动速度阈值
        self.ball_distance_threshold = 0.3  # 增加球与投篮手的距离阈值

    def calculate_angle_3d(self, a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
        """计算3D空间中的角度"""
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

    def detect_body_facing(self, keypoints) -> Tuple[str, str]:
        """检测身体朝向和侧面方向"""
        if keypoints.shape[0] < 11:
            return "unknown", "unknown"
            
        # 获取关键点
        left_shoulder = keypoints[5]  # 左肩
        right_shoulder = keypoints[6]  # 右肩
        left_hip = keypoints[11]  # 左髋
        right_hip = keypoints[12]  # 右髋
        nose = keypoints[0]  # 鼻子
        
        # 计算肩膀和髋部的Z值差异
        z_diff_shoulders = left_shoulder[2] - right_shoulder[2]
        z_diff_hips = left_hip[2] - right_hip[2]
        
        # 计算肩膀的水平距离
        shoulder_distance = abs(left_shoulder[0] - right_shoulder[0])
        
        # 计算鼻子与肩膀中点的x坐标差异
        shoulder_midpoint_x = (left_shoulder[0] + right_shoulder[0]) / 2
        nose_offset = abs(nose[0] - shoulder_midpoint_x)
        
        # 侧面判断
        if (shoulder_distance < 0.15 or  # 肩膀水平距离小
            nose_offset > 0.15):  # 鼻子不在肩膀中点附近
            # 判断侧面方向
            side_direction = "left" if nose[0] < shoulder_midpoint_x else "right"
            return "side", side_direction
        
        # 背面判断
        facing_score = (z_diff_shoulders + z_diff_hips) / 2
        if facing_score > 0.05:
            return "back", "unknown"
            
        # 默认为正面
        return "front", "unknown"

    def detect_shooting_arm(self, keypoints, body_facing: str, side_direction: str) -> str:
        """检测投篮手臂"""
        if keypoints.shape[0] < 11:
            return None
            
        # 获取关键点
        left_shoulder = keypoints[5]
        right_shoulder = keypoints[6]
        left_elbow = keypoints[7]
        right_elbow = keypoints[8]
        left_wrist = keypoints[9]
        right_wrist = keypoints[10]
        
        # 计算肘部角度
        left_angle = self.calculate_angle_3d(
            left_shoulder.cpu().numpy(),
            left_elbow.cpu().numpy(),
            left_wrist.cpu().numpy()
        )
        
        right_angle = self.calculate_angle_3d(
            right_shoulder.cpu().numpy(),
            right_elbow.cpu().numpy(),
            right_wrist.cpu().numpy()
        )
        
        # 根据身体朝向判断投篮手
        if body_facing == "side":
            # 侧面时，根据侧面方向判断
            if side_direction == "left":
                # 身体朝向左侧，右手更可能是投篮手
                if right_angle < 100 and right_angle > 60:
                    return "right"
                elif left_angle < 100 and left_angle > 60:
                    return "left"
            else:
                # 身体朝向右侧，左手更可能是投篮手
                if left_angle < 100 and left_angle > 60:
                    return "left"
                elif right_angle < 100 and right_angle > 60:
                    return "right"
        elif body_facing == "back":
            # 背面时，通过肘部角度和手腕位置判断
            if left_angle < right_angle - 20:
                return "left"
            elif right_angle < left_angle - 20:
                return "right"
            # 通过水平位置判断
            if left_wrist[0] < right_wrist[0]:
                return "right"
            else:
                return "left"
        else:
            # 正面时，通过肘部角度和可见性判断
            if left_angle < right_angle and left_wrist[2] > 0.5:
                return "left"
            elif right_angle < left_angle and right_wrist[2] > 0.5:
                return "right"
        
        return None

    def detect_ball(self, frame: np.ndarray) -> Tuple[bool, List[float]]:
        """检测篮球位置和运动"""
        results = self.ball_model(frame, conf=self.ball_detection_threshold)
        ball_detected = False
        ball_position = None
        
        for result in results:
            boxes = result.boxes
            for box in boxes:
                if box.cls == 32:  # YOLO中篮球的类别ID
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    conf = box.conf[0].cpu().numpy()
                    ball_position = [(x1 + x2) / 2, (y1 + y2) / 2, conf]
                    ball_detected = True
                    break
            if ball_detected:
                break
        
        # 更新球的运动轨迹
        if ball_detected:
            self.ball_tracking_history.append(ball_position)
            if len(self.ball_tracking_history) > self.max_ball_history:
                self.ball_tracking_history.pop(0)
        
        return ball_detected, ball_position

    def is_ball_moving_upward(self) -> bool:
        """判断球是否向上运动"""
        if len(self.ball_tracking_history) < 2:
            return False
            
        # 计算球的垂直移动速度
        recent_positions = self.ball_tracking_history[-2:]
        y_diff = recent_positions[1][1] - recent_positions[0][1]
        return y_diff < -self.min_ball_movement  # 负值表示向上运动

    def detect_shooting_action(self, results, ball_position: List[float]) -> Tuple[bool, str, float]:
        """使用YOLO检测投篮动作，结合篮球位置"""
        if not results or not results[0].keypoints or len(results[0].keypoints.data) == 0:
            return False, None, 0.0
            
        try:
            keypoints = results[0].keypoints.data[0]  # 获取第一个人的关键点
            
            # 检查关键点是否有效
            if keypoints.shape[0] < 11:  # 确保有足够的关键点
                return False, None, 0.0
                
            # 检查关键点是否可见（置信度大于0）
            if (keypoints[5][2] < 0.1 or keypoints[6][2] < 0.1 or
                keypoints[7][2] < 0.1 or keypoints[8][2] < 0.1 or
                keypoints[9][2] < 0.1 or keypoints[10][2] < 0.1):
                return False, None, 0.0
            
            # 检测身体朝向
            body_facing, side_direction = self.detect_body_facing(keypoints)
            
            # 检测投篮手臂
            shooting_arm = self.detect_shooting_arm(keypoints, body_facing, side_direction)
            if not shooting_arm:
                return False, None, 0.0
            
            # 获取对应手臂的关键点
            if shooting_arm == "left":
                shoulder = keypoints[5]
                elbow = keypoints[7]
                wrist = keypoints[9]
            else:
                shoulder = keypoints[6]
                elbow = keypoints[8]
                wrist = keypoints[10]
            
            # 计算肘部角度
            angle = self.calculate_angle_3d(
                shoulder.cpu().numpy(),
                elbow.cpu().numpy(),
                wrist.cpu().numpy()
            )
            
            # 判断投篮动作
            is_shooting = False
            confidence = 0.0
            
            # 检查肘部角度
            if angle < 120 and angle > 40:  # 放宽角度范围
                # 如果检测到篮球，检查球的位置
                if ball_position:
                    wrist_pos = wrist.cpu().numpy()
                    ball_distance = np.sqrt(
                        (wrist_pos[0] - ball_position[0])**2 +
                        (wrist_pos[1] - ball_position[1])**2
                    )
                    
                    # 如果球在投篮手附近或向上运动，认为是投篮动作
                    if (ball_distance < self.ball_distance_threshold or 
                        self.is_ball_moving_upward()):
                        is_shooting = True
                        confidence = 1 - abs(angle - 90) / 90
                else:
                    # 即使没有检测到篮球，只要肘部角度合适也认为是投篮动作
                    is_shooting = True
                    confidence = 1 - abs(angle - 90) / 90
                
            return is_shooting, shooting_arm, confidence
            
        except Exception as e:
            print(f"检测投篮动作时出错: {str(e)}")
            return False, None, 0.0

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
        """保存关键帧"""
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        output_path = os.path.join(output_dir, f"keyframe_{frame_count:04d}_{description}.jpg")
        cv2.imwrite(output_path, frame)

    def put_chinese_text(self, img: np.ndarray, text: str, position: Tuple[int, int], color: Tuple[int, int, int]) -> np.ndarray:
        """在图像上绘制中文文本"""
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
        """分析投篮视频"""
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return "无法打开视频文件", []
        
        # 获取视频信息
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # 调整视频尺寸
        target_width = 1280
        target_height = 720
        scale = min(target_width / frame_width, target_height / frame_height)
        frame_width = int(frame_width * scale)
        frame_height = int(frame_height * scale)
        
        # 创建输出目录
        output_dir = "output_videos"
        keyframes_dir = "keyframes"
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(keyframes_dir, exist_ok=True)
        
        # 创建输出视频
        output_path = os.path.join(output_dir, "analyzed_shooting.mp4")
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))
        
        scores = []
        is_shooting = False
        frame_count = 0
        last_keyframe = 0
        keyframe_interval = 30
        shooting_arm = None
        shooting_start_frame = 0
        shooting_end_frame = 0
        min_shooting_duration = 5
        max_shooting_duration = 90
        body_facing = "unknown"
        side_direction = "unknown"
        
        try:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                    
                # 调整帧大小
                frame = cv2.resize(frame, (frame_width, frame_height))
                
                # 跳过部分帧
                if frame_count % (self.frame_skip + 1) != 0:
                    frame_count += 1
                    continue
                
                try:
                    # 使用YOLO进行姿态估计
                    results = self.yolo_model(frame, conf=self.conf_threshold)
                    
                    # 检测篮球
                    ball_detected, ball_position = self.detect_ball(frame)
                    
                    # 检测投篮动作
                    current_is_shooting, current_shooting_arm, confidence = self.detect_shooting_action(results, ball_position)
                    
                    # 更新身体朝向信息
                    if results and results[0].keypoints and len(results[0].keypoints.data) > 0:
                        keypoints = results[0].keypoints.data[0]
                        body_facing, side_direction = self.detect_body_facing(keypoints)
                    
                    # 处理投篮动作
                    if current_is_shooting:
                        if not is_shooting:
                            is_shooting = True
                            shooting_start_frame = frame_count
                            shooting_arm = current_shooting_arm
                            self.save_keyframe(frame, keyframes_dir, frame_count, "准备姿势")
                        else:
                            shooting_duration = frame_count - shooting_start_frame
                            if shooting_duration > max_shooting_duration:
                                is_shooting = False
                                shooting_start_frame = 0
                                shooting_end_frame = 0
                    else:
                        if is_shooting:
                            shooting_duration = frame_count - shooting_start_frame
                            if min_shooting_duration <= shooting_duration <= max_shooting_duration:
                                is_shooting = False
                                shooting_end_frame = frame_count
                                self.save_keyframe(frame, keyframes_dir, frame_count, "出手姿势")
                                scores.append(confidence)
                            else:
                                is_shooting = False
                                shooting_start_frame = 0
                                shooting_end_frame = 0
                    
                    # 绘制结果
                    annotated_frame = frame.copy()
                    
                    # 绘制篮球位置
                    if ball_detected:
                        x, y, conf = ball_position
                        cv2.circle(annotated_frame, (int(x), int(y)), 10, (0, 255, 0), -1)
                        cv2.putText(annotated_frame, f"篮球 {conf:.2f}", (int(x), int(y) - 10),
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                    
                    if results and results[0].keypoints and len(results[0].keypoints.data) > 0:
                        # 绘制关键点
                        for kp in results[0].keypoints.data[0]:
                            x, y = int(kp[0]), int(kp[1])
                            cv2.circle(annotated_frame, (x, y), 3, (0, 255, 0), -1)
                    
                    # 添加标签
                    label_texts = [
                        f"状态: {'投篮中' if is_shooting else '等待投篮'}",
                        f"投篮手: {'左手' if shooting_arm == 'left' else '右手' if shooting_arm else '未知'}",
                        f"朝向: {body_facing}" + (f" ({side_direction})" if body_facing == "side" else ""),
                        f"篮球: {'已检测' if ball_detected else '未检测'}",
                        f"进度: {frame_count}/{total_frames}",
                        f"已检测投篮: {len(scores)}次"
                    ]
                    
                    if is_shooting:
                        label_texts.extend([
                            f"置信度: {confidence:.2f}"
                        ])
                    
                    for i, text in enumerate(label_texts):
                        annotated_frame = self.put_chinese_text(
                            annotated_frame,
                            text,
                            (10, 10 + i * 25),
                            (255, 255, 255)
                        )
                    
                    # 保存关键帧
                    if is_shooting and frame_count - last_keyframe >= keyframe_interval:
                        self.save_keyframe(annotated_frame, keyframes_dir, frame_count, "动作过程")
                        last_keyframe = frame_count
                    
                    out.write(annotated_frame)
                    
                except Exception as e:
                    print(f"处理帧 {frame_count} 时出错: {str(e)}")
                    out.write(frame)  # 如果处理出错，保存原始帧
                
                frame_count += 1
                
        except Exception as e:
            print(f"分析视频时出错: {str(e)}")
            return f"分析过程中出现错误: {str(e)}", []
            
        finally:
            cap.release()
            out.release()
        
        if not scores:
            return "未检测到有效的投篮动作", []
        
        avg_score = np.mean(scores)
        quality = "标准" if avg_score > 0.7 else "需要改进" if avg_score > 0.5 else "不规范"
        
        return f"检测到{len(scores)}次投篮，平均质量：{quality}", scores

    def __del__(self):
        """清理资源"""
        if hasattr(self, 'pose'):
            self.pose.close()