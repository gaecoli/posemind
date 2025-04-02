import cv2
import mediapipe as mp
import numpy as np
from typing import List, Tuple

class PoseAnalyzer:
    def __init__(self):
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.mp_draw = mp.solutions.drawing_utils

    def analyze_pushup(self, video_path: str) -> Tuple[str, List[float]]:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return "无法打开视频文件", []

        pushup_count = 0
        scores = []
        is_down = False
        min_shoulder_elbow_angle = 90  # 俯卧撑最低点角度阈值

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # 转换颜色空间
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.pose.process(rgb_frame)

            if results.pose_landmarks:
                # 获取关键点
                landmarks = results.pose_landmarks.landmark
                
                # 计算肩部和肘部的角度
                shoulder = np.array([
                    landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                    landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER.value].y
                ])
                elbow = np.array([
                    landmarks[self.mp_pose.PoseLandmark.LEFT_ELBOW.value].x,
                    landmarks[self.mp_pose.PoseLandmark.LEFT_ELBOW.value].y
                ])
                wrist = np.array([
                    landmarks[self.mp_pose.PoseLandmark.LEFT_WRIST.value].x,
                    landmarks[self.mp_pose.PoseLandmark.LEFT_WRIST.value].y
                ])

                # 计算角度
                v1 = elbow - shoulder
                v2 = wrist - elbow
                angle = np.degrees(np.arctan2(np.cross(v1, v2), np.dot(v1, v2)))

                # 判断俯卧撑状态
                if angle < min_shoulder_elbow_angle and not is_down:
                    is_down = True
                elif angle > 160 and is_down:  # 俯卧撑最高点角度阈值
                    is_down = False
                    pushup_count += 1

                # 计算动作质量分数
                if is_down:
                    score = 1 - (angle - 90) / 90  # 归一化分数
                    scores.append(score)

        cap.release()

        # 计算平均分数
        avg_score = np.mean(scores) if scores else 0
        quality = "标准" if avg_score > 0.7 else "需要改进" if avg_score > 0.5 else "不规范"

        return f"完成 {pushup_count} 个俯卧撑，动作质量：{quality}", scores 