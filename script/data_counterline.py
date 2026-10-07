import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import Rbf, splprep, splev, griddata
from scipy.ndimage import gaussian_filter, distance_transform_edt
import os
import argparse
import mediapipe as mp
import dlib
import json
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from PIL import Image, ImageTk
import pickle
import math
from matplotlib.colors import LinearSegmentedColormap
from matplotlib import font_manager

class FaceContourEditor:
    """交互式面部轮廓编辑器类"""
    def __init__(self, image_path, initial_detector='mediapipe', contour_path=None, center_x=None):
        """初始化编辑器"""
        self.image_path = image_path
        self.original_image = cv2.imread(image_path)
        if self.original_image is None:
            raise ValueError(f"无法读取图像: {image_path}")
            
        self.h, self.w = self.original_image.shape[:2]
        self.display_image = self.original_image.copy()
        self.detector = initial_detector
        
        # 计算缩放比例以适应屏幕
        screen_width = 1400
        screen_height = 900
        self.scale_factor = min(screen_width / self.w, screen_height / self.h, 1.0)
        if self.scale_factor < 1.0:
            self.display_w = int(self.w * self.scale_factor)
            self.display_h = int(self.h * self.scale_factor)
        else:
            self.display_w = self.w
            self.display_h = self.h
            self.scale_factor = 1.0
        
        # 如果未指定中心点，使用图像宽度的一半
        self.center_x = center_x if center_x is not None else self.w // 2
        
        # 初始化UI元素
        self.root = tk.Tk()
        self.root.title("面部轮廓编辑器")
        self.root.geometry(f"{min(self.display_w+400, 1800)}x{min(self.display_h+200, 1000)}")
        
        # 设置主界面
        self.setup_ui()
        
        # 初始化轮廓
        self.contour_points = []
        self.dragging_point = None
        self.selected_point = None
        self.face_mask = None
        
        # 如果提供了轮廓路径，加载轮廓
        if contour_path and os.path.exists(contour_path):
            self.load_contour(contour_path)
        else:
            self.generate_initial_contour()
        
        # 更新显示
        self.update_display()
    
    def image_to_display_coords(self, x, y):
        """将图像坐标转换为显示坐标"""
        return int(x * self.scale_factor), int(y * self.scale_factor)
    
    def display_to_image_coords(self, x, y):
        """将显示坐标转换为图像坐标"""
        return int(x / self.scale_factor), int(y / self.scale_factor)
    
    def setup_ui(self):
        """设置用户界面"""
        # 创建主框架
        main_frame = tk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # 创建左侧控制面板
        control_frame = tk.Frame(main_frame, width=350)
        control_frame.pack(side=tk.LEFT, fill=tk.Y, padx=10, pady=10)
        control_frame.pack_propagate(False)
        
        # 创建图像显示区域
        self.canvas_frame = tk.Frame(main_frame)
        self.canvas_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # 创建画布（使用显示尺寸）
        self.canvas = tk.Canvas(self.canvas_frame, 
                               width=self.display_w, 
                               height=self.display_h,
                               bg='white')
        self.canvas.pack(fill=tk.BOTH, expand=True)
        
        # 绑定鼠标事件
        self.canvas.bind("<Button-1>", self.on_canvas_click)
        self.canvas.bind("<B1-Motion>", self.on_canvas_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_canvas_release)
        self.canvas.bind("<Button-3>", self.on_right_click)
        self.canvas.bind("<Double-Button-1>", self.on_double_click)
        
        # 控制面板标题
        title_label = tk.Label(control_frame, text="轮廓调整控制", font=("Arial", 14, "bold"))
        title_label.pack(pady=10)
        
        # 图像信息
        info_frame = tk.LabelFrame(control_frame, text="图像信息", padx=5, pady=5)
        info_frame.pack(fill=tk.X, pady=5)
        
        tk.Label(info_frame, text=f"原始尺寸: {self.w} x {self.h}").pack(anchor=tk.W)
        tk.Label(info_frame, text=f"显示尺寸: {self.display_w} x {self.display_h}").pack(anchor=tk.W)
        tk.Label(info_frame, text=f"缩放比例: {self.scale_factor:.2f}").pack(anchor=tk.W)
        
        # 轮廓操作
        contour_frame = tk.LabelFrame(control_frame, text="轮廓操作", padx=5, pady=5)
        contour_frame.pack(fill=tk.X, pady=5)
        
        tk.Button(contour_frame, text="重新生成轮廓", command=self.generate_initial_contour).pack(fill=tk.X, pady=1)
        tk.Button(contour_frame, text="平滑轮廓", command=self.smooth_contour).pack(fill=tk.X, pady=1)
        tk.Button(contour_frame, text="添加插值点", command=self.add_interpolated_points).pack(fill=tk.X, pady=1)
        
        # 文件操作
        file_frame = tk.LabelFrame(control_frame, text="文件操作", padx=5, pady=5)
        file_frame.pack(fill=tk.X, pady=5)
        
        tk.Button(file_frame, text="保存轮廓", command=self.save_contour_dialog).pack(fill=tk.X, pady=1)
        tk.Button(file_frame, text="加载轮廓", command=self.load_contour_dialog).pack(fill=tk.X, pady=1)
        
        # 操作说明
        help_frame = tk.LabelFrame(control_frame, text="操作说明", padx=5, pady=5)
        help_frame.pack(fill=tk.X, pady=5)
        
        instructions = (
            "• 左键点击: 选择点\n"
            "• 左键拖动: 移动点\n"
            "• 右键点击: 删除点\n"
            "• 双击空白: 添加新点"
        )
        tk.Label(help_frame, text=instructions, justify=tk.LEFT, font=("Arial", 9)).pack()
        
        # 当前状态
        self.status_var = tk.StringVar(value="就绪")
        status_frame = tk.Frame(control_frame)
        status_frame.pack(side=tk.BOTTOM, fill=tk.X, pady=5)
        tk.Label(status_frame, text="状态:", font=("Arial", 10, "bold")).pack(side=tk.LEFT)
        status_label = tk.Label(status_frame, textvariable=self.status_var, font=("Arial", 9))
        status_label.pack(side=tk.LEFT, padx=5)
        
        # 完成按钮
        tk.Button(
            control_frame, 
            text="完成并生成等高线图", 
            command=self.finish_editing,
            bg="#4CAF50", 
            fg="white", 
            font=("Arial", 12, "bold"),
            height=2
        ).pack(fill=tk.X, pady=10)
    
    def update_display(self):
        """更新显示图像"""
        # 复制原图并缩放
        if self.scale_factor != 1.0:
            display_img = cv2.resize(self.original_image, (self.display_w, self.display_h), 
                                   interpolation=cv2.INTER_AREA)
        else:
            display_img = self.original_image.copy()
        
        # 转换中心线到显示坐标
        display_center_x = int(self.center_x * self.scale_factor)
        
        # 绘制中心线
        cv2.line(display_img, 
                (display_center_x, 0), 
                (display_center_x, self.display_h), 
                (0, 255, 0), 2)
        
        # 绘制当前轮廓
        if len(self.contour_points) > 2:
            # 转换轮廓点到显示坐标
            display_contour_points = []
            for point in self.contour_points:
                display_x, display_y = self.image_to_display_coords(point[0], point[1])
                display_contour_points.append((display_x, display_y))
            
            # 绘制闭合区域
            contour_array = np.array(display_contour_points, dtype=np.int32)
            
            # 创建半透明填充
            overlay = display_img.copy()
            cv2.fillPoly(overlay, [contour_array], (0, 0, 255))
            cv2.addWeighted(overlay, 0.3, display_img, 0.7, 0, display_img)
            
            # 绘制轮廓线
            for i in range(len(display_contour_points)):
                p1 = display_contour_points[i]
                p2 = display_contour_points[(i + 1) % len(display_contour_points)]
                cv2.line(display_img, p1, p2, (0, 255, 0), 2, lineType=cv2.LINE_AA)
            
            # 绘制点
            for i, point in enumerate(display_contour_points):
                # 高亮选中的点
                if i == self.selected_point:
                    cv2.circle(display_img, point, 8, (255, 255, 0), -1, lineType=cv2.LINE_AA)
                    cv2.circle(display_img, point, 8, (0, 0, 0), 2, lineType=cv2.LINE_AA)
                else:
                    cv2.circle(display_img, point, 5, (0, 0, 255), -1, lineType=cv2.LINE_AA)
                    cv2.circle(display_img, point, 5, (255, 255, 255), 1, lineType=cv2.LINE_AA)
        
        # 转换为RGB用于Tkinter显示
        display_img_rgb = cv2.cvtColor(display_img, cv2.COLOR_BGR2RGB)
        self.display_image_pil = Image.fromarray(display_img_rgb)
        self.display_image_tk = ImageTk.PhotoImage(image=self.display_image_pil)
        
        # 更新画布
        self.canvas.delete("all")
        self.canvas.create_image(0, 0, image=self.display_image_tk, anchor=tk.NW)
    
    def generate_initial_contour(self):
        """生成初始轮廓"""
        self.status_var.set("生成初始轮廓...")
        self.root.update()
        
        # 使用手动生成的轮廓
        mask, contour = self.create_manual_face_mask(self.original_image, self.center_x)
        
        if contour is not None and len(contour) > 0:
            self.contour_points = contour
            self.selected_point = None
            self.status_var.set(f"生成了{len(self.contour_points)}个轮廓点")
        else:
            self.status_var.set("生成轮廓失败")
        
        self.update_display()
    
    def create_manual_face_mask(self, image, center_x=None):
        """手动创建面部掩码基于面部关键点"""
        h, w = image.shape[:2]
        
        if center_x is None:
            center_x = w // 2
        
        # 改进的脸型轮廓点
        contour_offsets = [
            # 前额部分 (从左到右)
            (-0.35, 0.15), (-0.25, 0.08), (-0.15, 0.05), (-0.05, 0.03), (0, 0.02),
            (0.05, 0.03), (0.15, 0.05), (0.25, 0.08), (0.35, 0.15),
            # 右脸部分 (从上到下)
            (0.42, 0.25), (0.45, 0.35), (0.47, 0.45), (0.46, 0.55), (0.43, 0.65),
            (0.38, 0.75), (0.30, 0.83), (0.20, 0.88), (0.10, 0.92),
            # 下巴部分
            (0, 0.94),
            # 左脸部分 (从下到上，对称)
            (-0.10, 0.92), (-0.20, 0.88), (-0.30, 0.83), (-0.38, 0.75), (-0.43, 0.65),
            (-0.46, 0.55), (-0.47, 0.45), (-0.45, 0.35), (-0.42, 0.25),
        ]
        
        # 估计面部大小
        face_height = h * 0.8
        face_width = w * 0.35
        face_top = h * 0.05
        
        # 计算实际坐标
        contour_points = []
        for dx, dy in contour_offsets:
            x = center_x + int(dx * face_width)
            y = int(face_top + dy * face_height)
            
            x = max(0, min(w-1, x))
            y = max(0, min(h-1, y))
            
            contour_points.append((x, y))
            
        return None, contour_points
    
    def on_canvas_click(self, event):
        """处理画布点击事件"""
        canvas_x = event.x
        canvas_y = event.y
        
        # 检查是否点击现有点
        for i, point in enumerate(self.contour_points):
            display_x, display_y = self.image_to_display_coords(point[0], point[1])
            if abs(display_x - canvas_x) < 15 and abs(display_y - canvas_y) < 15:
                self.dragging_point = i
                self.selected_point = i
                self.update_display()
                self.status_var.set(f"选择点 #{i}: ({point[0]}, {point[1]})")
                return
        
        # 清除选择
        self.selected_point = None
        self.update_display()
    
    def on_canvas_drag(self, event):
        """处理画布拖动事件"""
        if self.dragging_point is not None:
            image_x, image_y = self.display_to_image_coords(event.x, event.y)
            
            # 限制坐标在图像范围内
            image_x = max(0, min(self.w - 1, image_x))
            image_y = max(0, min(self.h - 1, image_y))
            
            # 更新点位置
            self.contour_points[self.dragging_point] = (image_x, image_y)
            self.update_display()
            self.status_var.set(f"移动点 #{self.dragging_point}: ({image_x}, {image_y})")
    
    def on_canvas_release(self, event):
        """处理鼠标释放事件"""
        self.dragging_point = None
    
    def on_right_click(self, event):
        """处理右键点击（删除点）"""
        # 查找最近的点
        closest_dist = float('inf')
        closest_idx = -1
        
        for i, point in enumerate(self.contour_points):
            display_x, display_y = self.image_to_display_coords(point[0], point[1])
            dist = ((display_x - event.x) ** 2 + (display_y - event.y) ** 2) ** 0.5
            if dist < closest_dist and dist < 20:
                closest_dist = dist
                closest_idx = i
        
        # 删除点
        if closest_idx != -1 and len(self.contour_points) > 3:
            removed = self.contour_points.pop(closest_idx)
            self.selected_point = None
            self.update_display()
            self.status_var.set(f"删除点 #{closest_idx}: ({removed[0]}, {removed[1]})")
    
    def on_double_click(self, event):
        """处理双击事件添加新点"""
        image_x, image_y = self.display_to_image_coords(event.x, event.y)
        
        # 添加新点
        self.contour_points.append((image_x, image_y))
        self.selected_point = len(self.contour_points) - 1
        self.update_display()
        self.status_var.set(f"添加点 #{len(self.contour_points) - 1}: ({image_x}, {image_y})")
    
    def smooth_contour(self):
        """平滑轮廓"""
        if len(self.contour_points) < 4:
            messagebox.showinfo("提示", "需要至少4个点才能平滑轮廓")
            return
            
        # 使用样条插值平滑轮廓
        points = np.array(self.contour_points)
        points = np.vstack([points, points[0]])
        
        x = points[:, 0]
        y = points[:, 1]
        
        try:
            t = np.zeros(len(points))
            for i in range(1, len(points)):
                t[i] = t[i-1] + np.sqrt((x[i] - x[i-1])**2 + (y[i] - y[i-1])**2)
            
            t = t / t[-1]
            
            tck, u = splprep([x, y], u=t, s=0.0, per=1)
            
            num_new_points = len(self.contour_points)
            new_points = splev(np.linspace(0, 1, num_new_points), tck)
            
            self.contour_points = [(int(new_points[0][i]), int(new_points[1][i])) 
                                  for i in range(num_new_points)]
            
            self.selected_point = None
            self.update_display()
            self.status_var.set(f"轮廓已平滑，包含{len(self.contour_points)}个点")
        except Exception as e:
            messagebox.showerror("错误", f"平滑轮廓失败: {str(e)}")
    
    def add_interpolated_points(self):
        """在轮廓线上添加插值点"""
        if len(self.contour_points) < 3:
            messagebox.showinfo("提示", "需要至少3个点才能添加插值点")
            return
        
        new_points = []
        
        for i in range(len(self.contour_points)):
            p1 = self.contour_points[i]
            p2 = self.contour_points[(i + 1) % len(self.contour_points)]
            
            new_points.append(p1)
            
            mid_x = (p1[0] + p2[0]) // 2
            mid_y = (p1[1] + p2[1]) // 2
            
            new_points.append((mid_x, mid_y))
        
        self.contour_points = new_points
        self.selected_point = None
        self.update_display()
        self.status_var.set(f"添加了插值点，现在有{len(self.contour_points)}个点")
    
    def save_contour_dialog(self):
        """打开保存轮廓对话框"""
        file_path = filedialog.asksaveasfilename(
            defaultextension=".json",
            filetypes=[("JSON文件", "*.json"), ("所有文件", "*.*")],
            title="保存面部轮廓"
        )
        
        if file_path:
            self.save_contour(file_path)
    
    def save_contour(self, file_path):
        """保存轮廓到文件"""
        try:
            data = {
                "contour_points": self.contour_points,
                "center_x": self.center_x,
                "image_path": self.image_path,
                "image_size": (self.w, self.h)
            }
            
            with open(file_path, 'w') as f:
                json.dump(data, f, indent=2)
                
            self.status_var.set(f"轮廓已保存到: {file_path}")
        except Exception as e:
            messagebox.showerror("错误", f"保存轮廓失败: {str(e)}")
    
    def load_contour_dialog(self):
        """打开加载轮廓对话框"""
        file_path = filedialog.askopenfilename(
            filetypes=[("JSON文件", "*.json"), ("所有文件", "*.*")],
            title="加载面部轮廓"
        )
        
        if file_path:
            self.load_contour(file_path)
    
    def load_contour(self, file_path):
        """从文件加载轮廓"""
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
            
            if "contour_points" not in data:
                messagebox.showerror("错误", "无效的轮廓文件格式")
                return
            
            self.contour_points = data["contour_points"]
            
            if "center_x" in data:
                self.center_x = data["center_x"]
            
            self.selected_point = None
            self.update_display()
            self.status_var.set(f"从{file_path}加载了{len(self.contour_points)}个轮廓点")
            
        except Exception as e:
            messagebox.showerror("错误", f"加载轮廓失败: {str(e)}")
    
    def finish_editing(self):
        """完成编辑并进入等高线图生成"""
        if len(self.contour_points) < 3:
            if not messagebox.askyesno("警告", "轮廓不完整，确定要继续吗？"):
                return
        
        # 保存轮廓到临时文件
        save_dir = os.path.dirname(self.image_path)
        base_name = os.path.splitext(os.path.basename(self.image_path))[0]
        auto_save_path = os.path.join(save_dir, f"{base_name}_contour.json")
        
        try:
            self.save_contour(auto_save_path)
        except:
            pass
        
        # 关闭编辑器窗口
        self.root.destroy()


class FacialFeatureDetector:
    """面部五官检测器"""
    def __init__(self):
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
    
    def detect_facial_features(self, image):
        """检测面部五官关键点"""
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb_image)
        
        if not results.multi_face_landmarks:
            return None
        
        landmarks = results.multi_face_landmarks[0]
        h, w = image.shape[:2]
        
        # 关键点索引定义
        feature_indices = {
            'left_eye': [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246],
            'right_eye': [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398],
            'nose': [19, 20, 1, 2, 5, 4, 6, 168, 8, 9, 10, 151, 195, 197, 196, 3, 51, 48, 115, 131, 134],
            'mouth': [61, 84, 17, 314, 405, 320, 307, 375, 321, 308, 324, 318, 402, 317, 14, 87, 178, 88, 95],
            'left_eyebrow': [70, 63, 105, 66, 107, 55, 65, 52, 53, 46],
            'right_eyebrow': [296, 334, 293, 300, 276, 283, 282, 295, 285, 336]
        }
        
        features = {}
        for feature_name, indices in feature_indices.items():
            points = []
            for idx in indices:
                if idx < len(landmarks.landmark):
                    landmark = landmarks.landmark[idx]
                    x = int(landmark.x * w)
                    y = int(landmark.y * h)
                    points.append((x, y))
            features[feature_name] = points
        
        return features


class ContourFittingGenerator:
    """贴合轮廓和五官的等高线图生成器"""
    def __init__(self, original_img, contour_points, face_center, output_dir):
        self.original_img = original_img
        self.contour_points = contour_points
        self.face_center = face_center
        self.output_dir = output_dir
        self.img_h, self.img_w = original_img.shape[:2]
        
        # 初始化五官检测器
        self.feature_detector = FacialFeatureDetector()
        
        # 检测五官
        self.facial_features = self.feature_detector.detect_facial_features(original_img)
        
        # 创建面部掩码
        self.face_mask = self.create_face_mask()

    def create_face_mask(self):
        """创建面部掩码"""
        mask = np.zeros((self.img_h, self.img_w), dtype=np.float32)
        
        if len(self.contour_points) < 3:
            return mask
        
        contour_array = np.array(self.contour_points, dtype=np.int32)
        cv2.fillPoly(mask, [contour_array], 1.0)
        
        return mask
    
    def create_distance_field(self):
        """创建基于轮廓和五官的距离场"""
        # 创建轮廓距离场
        contour_mask = np.zeros((self.img_h, self.img_w), dtype=np.uint8)
        if len(self.contour_points) > 2:
            contour_array = np.array(self.contour_points, dtype=np.int32)
            cv2.fillPoly(contour_mask, [contour_array], 255)
        
        # 计算到轮廓边界的距离
        distance_to_boundary = distance_transform_edt(contour_mask)
        
        # 创建五官影响区域
        feature_influence = np.zeros((self.img_h, self.img_w), dtype=np.float32)
        
        if self.facial_features:
            for feature_name, points in self.facial_features.items():
                if len(points) > 0:
                    # 为不同五官设置不同的影响强度
                    influence_strength = {
                        'left_eye': 0.8,
                        'right_eye': 0.8,
                        'nose': 0.6,
                        'mouth': 0.7,
                        'left_eyebrow': 0.4,
                        'right_eyebrow': 0.4
                    }.get(feature_name, 0.5)
                    
                    # 创建五官区域掩码
                    feature_mask = np.zeros((self.img_h, self.img_w), dtype=np.uint8)
                    if len(points) > 2:
                        points_array = np.array(points, dtype=np.int32)
                        cv2.fillPoly(feature_mask, [points_array], 255)
                    
                    # 扩展五官影响区域
                    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (30, 30))
                    expanded_mask = cv2.dilate(feature_mask, kernel, iterations=1)
                    
                    # 添加到特征影响图
                    feature_influence += (expanded_mask / 255.0) * influence_strength
        
        # 归一化距离场
        if distance_to_boundary.max() > 0:
            normalized_distance = distance_to_boundary / distance_to_boundary.max()
        else:
            normalized_distance = np.zeros_like(distance_to_boundary)
        
        # 结合轮廓距离和五官影响
        combined_field = normalized_distance * 0.7 + feature_influence * 0.3
        
        # 确保值在合理范围内
        combined_field = np.clip(combined_field, 0, 1)
        
        return combined_field
    
    def create_contour_fitted_values(self, pattern_type='distance_based'):
        """创建贴合轮廓的数值分布"""
        if pattern_type == 'distance_based':
            # 基于到轮廓边界的距离
            values = self.create_distance_field()
            
        elif pattern_type == 'concentric_fitted':
            # 贴合轮廓的同心圆模式
            # 计算轮廓中心
            contour_array = np.array(self.contour_points)
            center_y, center_x = np.mean(contour_array, axis=0).astype(int)
            
            # 创建到中心的距离场
            y_coords, x_coords = np.ogrid[:self.img_h, :self.img_w]
            distances = np.sqrt((x_coords - center_x)**2 + (y_coords - center_y)**2)
            
            # 计算轮廓内最大距离来归一化
            face_mask_bool = self.face_mask > 0
            if np.any(face_mask_bool):
                max_dist_in_face = distances[face_mask_bool].max()
                if max_dist_in_face > 0:
                    values = 1.0 - (distances / max_dist_in_face)
                else:
                    values = np.ones_like(distances) * 0.5
            else:
                values = np.ones_like(distances) * 0.5
                
        elif pattern_type == 'facial_topology':
            # 基于面部拓扑结构
            values = np.zeros((self.img_h, self.img_w), dtype=np.float32)
            
            # 根据面部区域设置不同值
            contour_array = np.array(self.contour_points)
            if len(contour_array) > 0:
                # 上部区域（额头、眉毛）
                top_y = contour_array[:, 1].min()
                mid_y = np.mean(contour_array[:, 1])
                bottom_y = contour_array[:, 1].max()
                
                y_coords, x_coords = np.ogrid[:self.img_h, :self.img_w]
                
                # 创建渐变
                forehead_region = (y_coords <= top_y + (mid_y - top_y) * 0.4)
                cheek_region = ((y_coords > top_y + (mid_y - top_y) * 0.4) & 
                               (y_coords <= mid_y + (bottom_y - mid_y) * 0.6))
                chin_region = (y_coords > mid_y + (bottom_y - mid_y) * 0.6)
                
                values[forehead_region] = 0.3
                values[cheek_region] = 0.8
                values[chin_region] = 0.5
            
        elif pattern_type == 'feature_enhanced':
            # 增强五官特征的模式
            values = self.create_distance_field()
            
            # 增强五官区域
            if self.facial_features:
                for feature_name, points in self.facial_features.items():
                    if len(points) > 2:
                        feature_mask = np.zeros((self.img_h, self.img_w), dtype=np.uint8)
                        points_array = np.array(points, dtype=np.int32)
                        cv2.fillPoly(feature_mask, [points_array], 255)
                        
                        # 不同五官不同的增强强度
                        enhancement = {
                            'left_eye': 0.9,
                            'right_eye': 0.9,
                            'nose': 0.7,
                            'mouth': 0.8,
                            'left_eyebrow': 0.6,
                            'right_eyebrow': 0.6
                        }.get(feature_name, 0.5)
                        
                        values[feature_mask > 0] = enhancement
        
        # 应用面部掩码
        values = values * self.face_mask
        
        # 平滑处理以减少边缘锯齿
        values = gaussian_filter(values, sigma=2.0)
        
        return values
    
    def open_adjustment_interface(self):
        """打开调整界面"""
        self.adjustment_window = tk.Tk()
        self.adjustment_window.title("贴合轮廓等高线图生成器")
        self.adjustment_window.geometry("650x800")
        
        # 创建主框架
        main_frame = tk.Frame(self.adjustment_window)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # 标题
        title_label = tk.Label(main_frame, text="贴合轮廓和五官的等高线图生成器", font=("Arial", 16, "bold"))
        title_label.pack(pady=10)
        
        # 五官检测状态
        feature_frame = tk.LabelFrame(main_frame, text="五官检测状态", padx=5, pady=5)
        feature_frame.pack(fill=tk.X, pady=5)
        
        if self.facial_features:
            detected_features = []
            for feature_name, points in self.facial_features.items():
                if points:
                    detected_features.append(f"{feature_name}: {len(points)}个点")
            status_text = "✓ 已检测到五官：" + ", ".join(detected_features)
        else:
            status_text = "✗ 未检测到五官，将仅使用轮廓"
        
        tk.Label(feature_frame, text=status_text, wraplength=600, justify=tk.LEFT).pack(anchor=tk.W)
        
        # 等高线模式选择
        pattern_frame = tk.LabelFrame(main_frame, text="等高线贴合模式", padx=5, pady=5)
        pattern_frame.pack(fill=tk.X, pady=5)
        
        self.pattern_var = tk.StringVar(value='distance_based')
        patterns = [
            ('distance_based', '距离贴合 - 基于到轮廓边界的距离'),
            ('concentric_fitted', '同心圆贴合 - 从面部中心向轮廓递减'),
            ('facial_topology', '面部拓扑 - 基于面部区域结构'),
            ('feature_enhanced', '五官增强 - 突出显示五官特征')
        ]
        
        for value, desc in patterns:
            tk.Radiobutton(pattern_frame, text=desc, variable=self.pattern_var, 
                          value=value, wraplength=500, justify=tk.LEFT).pack(anchor=tk.W, pady=2)
        
        # 等高线参数
        contour_frame = tk.LabelFrame(main_frame, text="等高线参数", padx=5, pady=5)
        contour_frame.pack(fill=tk.X, pady=5)
        
        # 等高线数量
        levels_frame = tk.Frame(contour_frame)
        levels_frame.pack(fill=tk.X, pady=2)
        tk.Label(levels_frame, text="等高线数量:", width=15).pack(side=tk.LEFT)
        self.levels_var = tk.IntVar(value=20)
        levels_scale = tk.Scale(levels_frame, from_=8, to=40, resolution=1,
                               orient=tk.HORIZONTAL, variable=self.levels_var, length=200)
        levels_scale.pack(side=tk.RIGHT, fill=tk.X, expand=True)
        
        # 等高线颜色
        color_frame = tk.Frame(contour_frame)
        color_frame.pack(fill=tk.X, pady=2)
        tk.Label(color_frame, text="等高线颜色:", width=15).pack(side=tk.LEFT)
        self.contour_color_var = tk.StringVar(value='white')
        color_combo = ttk.Combobox(color_frame, textvariable=self.contour_color_var,
                                  values=['white', 'black', 'red', 'blue', 'green', 'yellow', 'cyan', 'magenta'],
                                  state="readonly", width=15)
        color_combo.pack(side=tk.RIGHT, fill=tk.X, expand=True)
        
        # 等高线粗细
        width_frame = tk.Frame(contour_frame)
        width_frame.pack(fill=tk.X, pady=2)
        tk.Label(width_frame, text="线条粗细:", width=15).pack(side=tk.LEFT)
        self.contour_width_var = tk.DoubleVar(value=1.0)
        width_scale = tk.Scale(width_frame, from_=0.3, to=3.0, resolution=0.1,
                              orient=tk.HORIZONTAL, variable=self.contour_width_var, length=200)
        width_scale.pack(side=tk.RIGHT, fill=tk.X, expand=True)
        
        # 等高线透明度
        alpha_frame = tk.Frame(contour_frame)
        alpha_frame.pack(fill=tk.X, pady=2)
        tk.Label(alpha_frame, text="透明度:", width=15).pack(side=tk.LEFT)
        self.contour_alpha_var = tk.DoubleVar(value=0.8)
        alpha_scale = tk.Scale(alpha_frame, from_=0.1, to=1.0, resolution=0.05,
                              orient=tk.HORIZONTAL, variable=self.contour_alpha_var, length=200)
        alpha_scale.pack(side=tk.RIGHT, fill=tk.X, expand=True)
        
        # 线型选择
        style_frame = tk.Frame(contour_frame)
        style_frame.pack(fill=tk.X, pady=2)
        tk.Label(style_frame, text="线型:", width=15).pack(side=tk.LEFT)
        self.line_style_var = tk.StringVar(value='solid')
        style_combo = ttk.Combobox(style_frame, textvariable=self.line_style_var,
                                  values=['solid', 'dashed', 'dashdot', 'dotted'],
                                  state="readonly", width=15)
        style_combo.pack(side=tk.RIGHT, fill=tk.X, expand=True)
        
        # 显示数值标签
        self.show_labels_var = tk.BooleanVar(value=False)
        tk.Checkbutton(contour_frame, text="显示数值标签", 
                      variable=self.show_labels_var).pack(anchor=tk.W, pady=2)
        
        # 高级处理参数
        process_frame = tk.LabelFrame(main_frame, text="高级处理参数", padx=5, pady=5)
        process_frame.pack(fill=tk.X, pady=5)
        
        # 平滑强度
        smooth_frame = tk.Frame(process_frame)
        smooth_frame.pack(fill=tk.X, pady=2)
        tk.Label(smooth_frame, text="平滑强度:", width=15).pack(side=tk.LEFT)
        self.smooth_var = tk.DoubleVar(value=2.0)
        smooth_scale = tk.Scale(smooth_frame, from_=0.5, to=5.0, resolution=0.1,
                               orient=tk.HORIZONTAL, variable=self.smooth_var, length=200)
        smooth_scale.pack(side=tk.RIGHT, fill=tk.X, expand=True)
        
        # 边缘羽化
        feather_frame = tk.Frame(process_frame)
        feather_frame.pack(fill=tk.X, pady=2)
        tk.Label(feather_frame, text="边缘羽化:", width=15).pack(side=tk.LEFT)
        self.feather_var = tk.DoubleVar(value=3.0)
        feather_scale = tk.Scale(feather_frame, from_=0.0, to=8.0, resolution=0.2,
                                orient=tk.HORIZONTAL, variable=self.feather_var, length=200)
        feather_scale.pack(side=tk.RIGHT, fill=tk.X, expand=True)
        
        # 强度调节
        intensity_frame = tk.Frame(process_frame)
        intensity_frame.pack(fill=tk.X, pady=2)
        tk.Label(intensity_frame, text="整体强度:", width=15).pack(side=tk.LEFT)
        self.intensity_var = tk.DoubleVar(value=1.0)
        intensity_scale = tk.Scale(intensity_frame, from_=0.2, to=2.0, resolution=0.1,
                                  orient=tk.HORIZONTAL, variable=self.intensity_var, length=200)
        intensity_scale.pack(side=tk.RIGHT, fill=tk.X, expand=True)
        
        # 预览与生成
        preview_frame = tk.LabelFrame(main_frame, text="预览与生成", padx=5, pady=5)
        preview_frame.pack(fill=tk.X, pady=5)
        
        tk.Button(preview_frame, text="生成预览", command=self.generate_preview,
                 bg="#2196F3", fg="white", font=("Arial", 11, "bold")).pack(fill=tk.X, pady=2)
        
        tk.Button(preview_frame, text="生成贴合等高线图", command=self.generate_fitted_contour,
                 bg="#4CAF50", fg="white", font=("Arial", 12, "bold")).pack(fill=tk.X, pady=2)
        
        tk.Button(preview_frame, text="生成所有模式", command=self.generate_all_patterns,
                 bg="#FF9800", fg="white", font=("Arial", 11, "bold")).pack(fill=tk.X, pady=2)
        
        # 状态显示
        self.status_var = tk.StringVar(value="就绪 - 选择模式后生成贴合等高线图")
        status_label = tk.Label(main_frame, textvariable=self.status_var, 
                               font=("Arial", 9), fg="blue")
        status_label.pack(pady=5)
        
        # 启动界面
        self.adjustment_window.mainloop()
    
    def create_contour_figure(self, preview_mode=False):
        """创建贴合轮廓的等高线图"""
        import matplotlib
        matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
        matplotlib.rcParams['axes.unicode_minus'] = False
        
        # 根据选择的模式创建数值分布
        pattern_type = self.pattern_var.get()
        values = self.create_contour_fitted_values(pattern_type)
        
        # 应用强度调节
        intensity = self.intensity_var.get()
        values = values * intensity
        
        # 应用平滑处理
        smooth_strength = self.smooth_var.get()
        if smooth_strength > 0:
            values = gaussian_filter(values, sigma=smooth_strength)
        
        # 应用边缘羽化
        feather = self.feather_var.get()
        if feather > 0:
            face_mask_feathered = gaussian_filter(self.face_mask.astype(float), sigma=feather/2)
            values = values * face_mask_feathered
        
        # 设置掩码外区域为NaN
        values[self.face_mask < 0.1] = np.nan
        
        # 创建图形
        plt.figure(figsize=(16, 12) if not preview_mode else (12, 9))
        
        # 显示原图
        plt.imshow(cv2.cvtColor(self.original_img, cv2.COLOR_BGR2RGB))
        
        # 绘制等高线
        try:
            levels = self.levels_var.get()
            valid_values = values[~np.isnan(values)]
            if len(valid_values) > 0:
                min_val, max_val = np.min(valid_values), np.max(valid_values)
                if max_val > min_val:
                    contour_levels = np.linspace(min_val, max_val, levels)
                    
                    # 在绘制等高线前插入 meshgrid 生成
                    x = np.linspace(0, self.img_w, values.shape[1])
                    y = np.linspace(0, self.img_h, values.shape[0])
                    X, Y = np.meshgrid(x, y)
                    
                    # 替换原有的 plt.contour 调用为：
                    cs = plt.contour(
                        X, Y, values,
                        levels=contour_levels,
                        colors=self.contour_color_var.get(),
                        linewidths=self.contour_width_var.get(),
                        alpha=self.contour_alpha_var.get(),
                        linestyles=self.line_style_var.get()
                    )

                    
                    # 添加数值标签
                    if self.show_labels_var.get():
                        plt.clabel(cs, inline=True, fontsize=7, fmt="%.2f", 
                                  colors=self.contour_color_var.get())
                
        except Exception as e:
            print(f"等高线绘制失败: {e}")
        
        # 添加面部中心线
        plt.axvline(x=self.face_center, color='lime', linestyle='--', alpha=0.6, linewidth=1.5)
        
        # 可选：显示检测到的五官轮廓
        if hasattr(self, 'show_features_var') and self.show_features_var.get() and self.facial_features:
            feature_colors = {
                'left_eye': 'cyan',
                'right_eye': 'cyan',
                'nose': 'yellow',
                'mouth': 'magenta',
                'left_eyebrow': 'orange',
                'right_eyebrow': 'orange'
            }
            
            for feature_name, points in self.facial_features.items():
                if len(points) > 2:
                    points_array = np.array(points)
                    color = feature_colors.get(feature_name, 'white')
                    plt.plot(points_array[:, 0], points_array[:, 1], 'o-', 
                            color=color, markersize=2, linewidth=1, alpha=0.7)
        
        # 设置标题
        pattern_names = {
            'distance_based': '距离贴合模式',
            'concentric_fitted': '同心圆贴合模式',
            'facial_topology': '面部拓扑模式',
            'feature_enhanced': '五官增强模式'
        }
        
        title = f'贴合轮廓等高线图 - {pattern_names.get(pattern_type, pattern_type)}'
        plt.title(title, fontsize=16, fontweight='bold', pad=20)
        plt.axis('off')

        # 添加图例
        legend_elements = [
            plt.Line2D([0], [0], color='lime', linestyle='--', linewidth=2, label='中心线'),
            plt.Line2D([0], [0], color=self.contour_color_var.get(), 
                      linestyle=self.line_style_var.get(), linewidth=2, label='贴合等高线')
        ]
        
        plt.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(0.98, 0.98))
        plt.tight_layout()
        
        return plt.gcf()
    
    def generate_preview(self):
        """生成预览"""
        self.status_var.set("生成预览中...")
        self.adjustment_window.update()
        
        try:
            fig = self.create_contour_figure(preview_mode=True)
            plt.show()
            self.status_var.set("预览生成完成")
        except Exception as e:
            self.status_var.set(f"预览生成失败: {str(e)}")
    
    def generate_fitted_contour(self):
        """生成贴合等高线图"""
        self.status_var.set("正在生成贴合等高线图...")
        self.adjustment_window.update()
        
        try:
            fig = self.create_contour_figure()
            
            # 保存图像
            pattern_type = self.pattern_var.get()
            output_path = os.path.join(self.output_dir, f'fitted_contour_{pattern_type}.png')
            fig.savefig(output_path, bbox_inches='tight', dpi=300, facecolor='white')
            plt.show()
            
            self.status_var.set(f"贴合等高线图已保存: {output_path}")
            
        except Exception as e:
            self.status_var.set(f"生成失败: {str(e)}")
    
    def generate_all_patterns(self):
        """生成所有模式的等高线图"""
        self.status_var.set("正在生成所有模式...")
        self.adjustment_window.update()
        
        patterns = ['distance_based', 'concentric_fitted', 'facial_topology', 'feature_enhanced']
        original_pattern = self.pattern_var.get()
        
        for i, pattern in enumerate(patterns):
            try:
                self.pattern_var.set(pattern)
                fig = self.create_contour_figure()
                
                output_path = os.path.join(self.output_dir, f'fitted_contour_{pattern}.png')
                fig.savefig(output_path, bbox_inches='tight', dpi=300, facecolor='white')
                plt.close(fig)
                
                self.status_var.set(f"已生成 {i+1}/{len(patterns)}: {pattern}")
                self.adjustment_window.update()
                
            except Exception as e:
                self.status_var.set(f"生成{pattern}失败: {str(e)}")
        
        # 恢复原始选择
        self.pattern_var.set(original_pattern)
        self.status_var.set(f"所有模式已生成完成 (共{len(patterns)}张)")


def generate_fitted_contour_map(img, contour_points, face_center=None, output_dir='output'):
    """生成贴合轮廓和五官的等高线图"""
    h, w = img.shape[:2]
    
    if face_center is None:
        face_center = w // 2
    
    print(f"使用面部中心线X坐标: {face_center}")
    print("正在启动贴合轮廓等高线图生成器...")
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 启动调整工具
    generator = ContourFittingGenerator(img, contour_points, face_center, output_dir)
    generator.open_adjustment_interface()


def run_editor(image_path, initial_detector='mediapipe', contour_path=None, center_x=None):
    """运行交互式编辑器"""
    editor = FaceContourEditor(image_path, initial_detector, contour_path, center_x)
    editor.root.mainloop()
    
    # 返回最终的轮廓点和中心点
    return editor.contour_points, editor.center_x


def parse_arguments():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='贴合轮廓和五官的等高线图生成器')
    parser.add_argument('--image', type=str, default=os.path.join('data', 'modelface.png'),
                        help='输入图像路径')
    parser.add_argument('--output', type=str, default=os.path.join('data', 'output'),
                        help='输出目录')
    parser.add_argument('--center', type=int, default=None,
                        help='面部中心线的X坐标 (默认为图像宽度的一半)')
    parser.add_argument('--contour', type=str, default=None,
                        help='加载已保存的轮廓文件 (.json)')
    parser.add_argument('--mode', type=str, default='interactive',
                        choices=['interactive', 'auto'],
                        help='运行模式: interactive=交互式编辑, auto=自动生成')
    
    return parser.parse_args()


def main():
    """主函数"""
    # 解析命令行参数
    args = parse_arguments()
    
    # 创建输出目录
    os.makedirs(args.output, exist_ok=True)
    
    # 读取底图
    img = cv2.imread(args.image)
    if img is None:
        print(f"无法加载图像，请检查路径: {args.image}")
        return
    
    print(f"已加载图像: {args.image}, 尺寸: {img.shape[1]}x{img.shape[0]}")
    
    # 获取轮廓点
    if args.mode == 'interactive':
        print("启动交互式轮廓编辑器...")
        contour_points, center_x = run_editor(
            args.image, 
            initial_detector='mediapipe', 
            contour_path=args.contour, 
            center_x=args.center
        )
        
        if len(contour_points) < 3:
            print("轮廓编辑已取消或未创建有效轮廓")
            return
            
    elif args.contour is not None:
        # 加载已保存的轮廓
        print(f"加载轮廓文件: {args.contour}")
        try:
            with open(args.contour, 'r') as f:
                data = json.load(f)
                
            contour_points = data["contour_points"]
            center_x = data.get("center_x", args.center)
            if center_x is None:
                center_x = img.shape[1] // 2
                
            print(f"已加载{len(contour_points)}个轮廓点，中心线X坐标: {center_x}")
        except Exception as e:
            print(f"加载轮廓失败: {e}")
            return
    else:
        # 自动生成轮廓
        print("自动生成面部轮廓...")
        h, w = img.shape[:2]
        center_x = args.center if args.center is not None else w // 2
        
        # 使用简单的椭圆轮廓
        contour_points = []
        face_width = w * 0.35
        face_height = h * 0.8
        face_top = h * 0.1
        
        for angle in range(0, 360, 15):
            rad = np.radians(angle)
            x = center_x + int(face_width * 0.5 * np.cos(rad))
            y = int(face_top + face_height * 0.5 + face_height * 0.5 * np.sin(rad))
            x = max(0, min(w-1, x))
            y = max(0, min(h-1, y))
            contour_points.append((x, y))
        
        print(f"自动生成了{len(contour_points)}个轮廓点")
    
    print(f"开始启动贴合等高线图生成器，使用{len(contour_points)}个轮廓点...")
    
    # 启动贴合等高线图生成器
    generate_fitted_contour_map(
        img, 
        contour_points,
        face_center=center_x,
        output_dir=args.output
    )
    
    print("贴合等高线图生成器已完成！")


def create_test_data():
    """创建测试数据"""
    # 创建一个测试图像
    test_img = np.ones((400, 300, 3), dtype=np.uint8) * 200
    
    # 绘制一个简单的面部轮廓
    center_x = 150
    face_points = []
    
    # 生成椭圆形轮廓
    for angle in range(0, 360, 20):
        rad = np.radians(angle)
        x = center_x + int(80 * np.cos(rad))
        y = 200 + int(120 * np.sin(rad))
        face_points.append((x, y))
    
    # 绘制面部轮廓
    contour_array = np.array(face_points, dtype=np.int32)
    cv2.fillPoly(test_img, [contour_array], (220, 200, 180))
    cv2.polylines(test_img, [contour_array], True, (100, 100, 100), 2)
    
    # 添加一些面部特征
    cv2.circle(test_img, (130, 180), 8, (50, 50, 50), -1)  # 左眼
    cv2.circle(test_img, (170, 180), 8, (50, 50, 50), -1)  # 右眼
    cv2.ellipse(test_img, (150, 220), (15, 8), 0, 0, 180, (150, 100, 100), -1)  # 嘴巴
    
    return test_img, face_points, center_x


def demo_fitted_contour():
    """演示贴合轮廓等高线图生成"""
    print("创建测试数据...")
    test_img, contour_points, center_x = create_test_data()
    
    # 保存测试图像
    os.makedirs('data', exist_ok=True)
    cv2.imwrite('data/test_face.jpg', test_img)
    
    print("启动贴合轮廓等高线图生成器演示...")
    generate_fitted_contour_map(
        test_img, 
        contour_points, 
        face_center=center_x, 
        output_dir='data/output'
    )


if __name__ == "__main__":
    import sys
    
    # 检查是否是演示模式
    if len(sys.argv) > 1 and sys.argv[1] == 'demo':
        demo_fitted_contour()
    else:
        try:
            main()
        except Exception as e:
            import traceback
            print(f"发生错误: {e}")
            traceback.print_exc()