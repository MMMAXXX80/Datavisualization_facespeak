import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import Rbf, splprep, splev, griddata
from scipy.ndimage import gaussian_filter
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
            text="完成并生成热力图", 
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
    
        import numpy as np
    from scipy.interpolate import splprep, splev
    import cv2
    
    class ClassicHeatmapGenerator:
        # ...existing code...
    
        def create_face_mask(self):
            """创建面部掩码（平滑轮廓）"""
            mask = np.zeros((self.img_h, self.img_w), dtype=np.float32)
            
            if len(self.contour_points) < 3:
                return mask
            
            # 对轮廓点进行样条插值平滑
            contour_array = np.array(self.contour_points, dtype=np.int32)
            x = contour_array[:, 0]
            y = contour_array[:, 1]
            # 闭合轮廓
            x = np.r_[x, x[0]]
            y = np.r_[y, y[0]]
            try:
                tck, u = splprep([x, y], s=3.0, per=1)
                unew = np.linspace(0, 1.0, max(100, len(self.contour_points)*4))
                out = splev(unew, tck)
                smooth_contour = np.stack([out[0], out[1]], axis=1).astype(np.int32)
                # 不要画线
            except Exception as e:
                # 插值失败也不要画线
                smooth_contour = contour_array
    
            cv2.fillPoly(mask, [smooth_contour], 1.0)
            # 轻微模糊边缘
            mask = cv2.GaussianBlur(mask, (5, 5), 2)
            return mask
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
        """完成编辑并进入热力图生成"""
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


class ClassicHeatmapGenerator:
    """经典配色热力图生成器"""
    def __init__(self, original_img, contour_points, face_center, output_dir):
        self.original_img = original_img
        self.contour_points = contour_points
        self.face_center = face_center
        self.output_dir = output_dir
        self.img_h, self.img_w = original_img.shape[:2]
        
        # 创建面部掩码
        self.face_mask = self.create_face_mask()
        
        # 定义经典配色方案
        self.classic_colormaps = {
            'medical_thermal': {
                'colors': ['#000066', '#000099', '#0000CC', '#0033FF', '#0066FF', 
                          '#0099FF', '#00CCFF', '#00FFFF', '#33FFCC', '#66FF99',
                          '#99FF66', '#CCFF33', '#FFFF00', '#FFCC00', '#FF9900',
                          '#FF6600', '#FF3300', '#FF0000', '#CC0000', '#990000'],
                'name': '医学热成像 (蓝-青-绿-黄-红)',
                'description': '传统医学热成像配色，从低温蓝色到高温红色'
            },
            'rainbow_spectrum': {
                'colors': ['#4B0082', '#0000FF', '#00FF00', '#FFFF00', '#FFA500', '#FF0000'],
                'name': '彩虹光谱 (紫-蓝-绿-黄-橙-红)',
                'description': '基于可见光光谱的经典彩虹配色'
            },
            'infrared_flare': {
                'colors': ['#0A0040', '#1B0070', '#3B008F', '#5C009F', '#800090',
                           '#A00070', '#C00040', '#E04010', '#FFA000', '#FFFF80', '#FFFFFF'],
                'name': '红外火焰 (深蓝→红紫→橙黄→白)',
                'description': '源自红外热成像图像的色彩渐变，高对比度，适合强调温度和活跃度变化'
            },
            'fire_heat': {
                'colors': ['#000000', '#330000', '#660000', '#990000', '#CC0000', 
                          '#FF0000', '#FF3300', '#FF6600', '#FF9900', '#FFCC00', 
                          '#FFFF00', '#FFFFFF'],
                'name': '火焰热度 (黑-红-黄-白)',
                'description': '模拟火焰燃烧过程的经典热力配色'
            },
            'volcano': {
                'colors': ['#120000', '#7c1c13', '#c84630', '#ffb400', '#fffbe0'],
                'name': '火山 (黑-红-橙-黄-米白)',
                'description': '模拟火山熔岩的温度变化'
            },
            'magma_flow': {
                'colors': ['#000000', '#1A0A1A', '#330A33', '#4D0A4D', '#660A66',
                           '#800A80', '#990A99', '#B30AB3', '#CC0ACC', '#E60AE6',
                           '#FF0AFF', '#FF3366', '#FF5533', '#FF7700', '#FF9900',
                           '#FFBB00', '#FFDD00', '#FFFF00', '#FFFF66', '#FFFFCC'],
                'name': '岩浆流动 (黑-紫-红-橙-黄)',
                'description': '模拟岩浆从冷却到炽热的温度变化'
            },
            'northern_lights': {
                'colors': ['#000011', '#001122', '#002233', '#003344', '#004455',
                           '#005566', '#006677', '#007788', '#008899', '#0099AA',
                           '#00AABB', '#00BBCC', '#00CCDD', '#33DDAA', '#66EE77',
                           '#99FF44', '#CCFF11', '#DDFF00', '#EEFF22', '#FFFF44'],
                'name': '北极光 (深蓝-青-绿-黄)',
                'description': '北极光的神秘色彩变化'
            },
            'iron_heat': {
                'colors': ['#000000', '#1A0000', '#330000', '#4D0000', '#660000',
                           '#800000', '#990000', '#B30000', '#CC0000', '#E60000',
                           '#FF0000', '#FF3300', '#FF6600', '#FF9900', '#FFCC00',
                           '#FFFF00', '#FFFF33', '#FFFF66', '#FFFF99', '#FFFFFF'],
                'name': '铁热 (黑-红-橙-黄-白)',
            },
           'in_flare': {
                'colors': ['#0A0040', '#1B0070', '#3B008F', '#5C009F', '#800090',
                           '#A00070', '#C00040', '#E04010', '#FFA000', '#FFFF80', '#FFFFFF'],
                'name': '红外火焰  (白-黄-橙-红紫-蓝)',
                'description': '红外热成像色带，适用于表示低温聚焦、负向趋势或冷热点分布'
            },
            'solar_flare': {
                'colors': ['#000000', '#330000', '#660000', '#990000', '#CC0000',
                           '#FF0000', '#FF2200', '#FF4400', '#FF6600', '#FF8800',
                           '#FFAA00', '#FFCC00', '#FFEE00', '#FFFF00', '#FFFF22',
                           '#FFFF44', '#FFFF66', '#FFFF88', '#FFFFAA', '#FFFFCC'],
                 'name': '太阳耀斑 (黑-红-橙-黄-浅黄)'
            }
        }
        
        # 定义肌肉测量点
        self.muscle_points = self.generate_precise_muscle_points()
        # 你的平均值结果
        channel_means = {
            'channel1': 2.2910,
            'channel2': 147.3712,
            'channel3': 69.6406,
            'channel4': 356.1210,
            'channel5': 14.5586,
            'channel6': 0
        }
        # 2. 用均值覆盖每个点的 value
        for name, data in self.muscle_points.items():
            for ch in channel_means:
                if ch in name:
                    data['value'] = channel_means[ch]
        # 3. 归一化到0~1
        all_values = [data['value'] for data in self.muscle_points.values()]
        min_v = min(all_values)
        max_v = max(all_values)
        if max_v > min_v:
            for data in self.muscle_points.values():
               data['value'] = (data['value'] - min_v) / (max_v - min_v)
        else:
            for data in self.muscle_points.values():
                data['value'] = 0.5   

    def create_face_mask(self):
        """创建面部掩码"""
        mask = np.zeros((self.img_h, self.img_w), dtype=np.float32)
        
        if len(self.contour_points) < 3:
            return mask
        
        contour_array = np.array(self.contour_points, dtype=np.int32)
        cv2.fillPoly(mask, [contour_array], 1.0)
        
        # 轻微模糊边缘
        mask = cv2.GaussianBlur(mask, (5, 5), 2)
        
        return mask
    
    def generate_precise_muscle_points(self):
        """生成严格贴合底图的左右脸点位（左脸为右脸对称）"""
        # 右脸点位
        right_points = {
            'channel1 下巴': (709, 1488),
            'channel2 唇下': (772, 1340),
            'channel3 咬肌': (918, 1266),
            'channel4 唇上': (758, 1153),
            'channel5 颧骨': (953, 965),
            'channel6 额头': (840, 637),
        }
        # 统一参数
        default_value = 0.7
        default_radius = 30
        

        # 左脸点位：以face_center为对称轴
        left_points = {}
        for k, (x, y) in right_points.items():
            # 对称点x = face_center - (x - face_center) = 2*face_center - x
            left_x = 2 * self.face_center - x
            left_points[k.replace('channel', 'L_channel')] = (left_x, y)

        # 组装所有点
        muscle_points = {}
        for k, (x, y) in right_points.items():
            muscle_points['R_' + k] = {'coord': (x, y), 'value': default_value, 'radius': default_radius}
        for k, (x, y) in left_points.items():
            muscle_points[k] = {'coord': (x, y), 'value': default_value, 'radius': default_radius}

        return muscle_points
    
    def open_adjustment_interface(self):
        """打开调整界面"""
        self.adjustment_window = tk.Tk()
        self.adjustment_window.title("热力图生成器 - 贴合度调试")
        self.adjustment_window.geometry("600x900")
        
        # 创建主框架
        main_frame = tk.Frame(self.adjustment_window)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # 标题
        title_label = tk.Label(main_frame, text="热力图生成器", font=("Arial", 16, "bold"))
        title_label.pack(pady=10)
        
        # 配色方案选择
        colormap_frame = tk.LabelFrame(main_frame, text="经典配色方案", padx=5, pady=5)
        colormap_frame.pack(fill=tk.X, pady=5)
        
        self.colormap_var = tk.StringVar(value='medical_thermal')
        colormap_menu = ttk.Combobox(colormap_frame, textvariable=self.colormap_var, 
                                    values=list(self.classic_colormaps.keys()), 
                                    state="readonly", width=30)
        colormap_menu.pack(fill=tk.X, pady=2)
        colormap_menu.bind("<<ComboboxSelected>>", self.update_colormap_description)
        
        # 配色方案描述
        self.colormap_desc_var = tk.StringVar()
        desc_label = tk.Label(colormap_frame, textvariable=self.colormap_desc_var, 
                             wraplength=400, justify=tk.LEFT, font=("Arial", 9))
        desc_label.pack(fill=tk.X, pady=2)
        self.update_colormap_description()
        
        # 贴合度调试参数
        fitting_frame = tk.LabelFrame(main_frame, text="贴合度调试参数", padx=5, pady=5)
        fitting_frame.pack(fill=tk.X, pady=5)
        
        # 插值分辨率
        res_frame = tk.Frame(fitting_frame)
        res_frame.pack(fill=tk.X, pady=2)
        tk.Label(res_frame, text="插值分辨率:", width=15).pack(side=tk.LEFT)
        self.resolution_var = tk.IntVar(value=600)
        res_scale = tk.Scale(res_frame, from_=200, to=1000, resolution=50,
                            orient=tk.HORIZONTAL, variable=self.resolution_var, length=200)
        res_scale.pack(side=tk.RIGHT, fill=tk.X, expand=True)
        
        # 边缘羽化强度
        feather_frame = tk.Frame(fitting_frame)
        feather_frame.pack(fill=tk.X, pady=2)
        tk.Label(feather_frame, text="边缘羽化:", width=15).pack(side=tk.LEFT)
        self.feather_var = tk.DoubleVar(value=3.0)
        feather_scale = tk.Scale(feather_frame, from_=0.0, to=10.0, resolution=0.5,
                                orient=tk.HORIZONTAL, variable=self.feather_var, length=200)
        feather_scale.pack(side=tk.RIGHT, fill=tk.X, expand=True)
        
        # 轮廓收缩/扩张
        contour_frame = tk.Frame(fitting_frame)
        contour_frame.pack(fill=tk.X, pady=2)
        tk.Label(contour_frame, text="轮廓调整:", width=15).pack(side=tk.LEFT)
        self.contour_adjust_var = tk.IntVar(value=0)
        contour_scale = tk.Scale(contour_frame, from_=-20, to=20, resolution=1,
                                orient=tk.HORIZONTAL, variable=self.contour_adjust_var, length=200)
        contour_scale.pack(side=tk.RIGHT, fill=tk.X, expand=True)
        
        # 透明度
        alpha_frame = tk.Frame(fitting_frame)
        alpha_frame.pack(fill=tk.X, pady=2)
        tk.Label(alpha_frame, text="透明度:", width=15).pack(side=tk.LEFT)
        self.alpha_var = tk.DoubleVar(value=0.75)
        alpha_scale = tk.Scale(alpha_frame, from_=0.1, to=1.0, resolution=0.05,
                              orient=tk.HORIZONTAL, variable=self.alpha_var, length=200)
        alpha_scale.pack(side=tk.RIGHT, fill=tk.X, expand=True)

        # 等高线透明度
        contour_alpha_frame = tk.Frame(fitting_frame)
        contour_alpha_frame.pack(fill=tk.X, pady=2)
        tk.Label(contour_alpha_frame, text="等高线透明度:", width=15).pack(side=tk.LEFT)
        self.contour_alpha_var = tk.DoubleVar(value=0.7)
        contour_alpha_scale = tk.Scale(contour_alpha_frame, from_=0.1, to=1.0, resolution=0.05,
                                      orient=tk.HORIZONTAL, variable=self.contour_alpha_var, length=200)
        contour_alpha_scale.pack(side=tk.RIGHT, fill=tk.X, expand=True)
        
        # 肌肉点强度调整
        muscle_frame = tk.LabelFrame(main_frame, text="活动强度调整", padx=5, pady=5)
        muscle_frame.pack(fill=tk.X, pady=5)
        
        # 创建滚动区域
        canvas = tk.Canvas(muscle_frame, height=150)
        scrollbar = tk.Scrollbar(muscle_frame, orient="vertical", command=canvas.yview)
        scrollable_frame = tk.Frame(canvas)
        
        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
        # 肌肉点调整控件
        self.muscle_vars = {}
        for name, data in self.muscle_points.items():
            if name in self.muscle_vars:
               self.muscle_vars[name].set(data['value'])
            point_frame = tk.Frame(scrollable_frame)
            point_frame.pack(fill=tk.X, pady=1)
            
            tk.Label(point_frame, text=name, width=12).pack(side=tk.LEFT)
            
            var = tk.DoubleVar(value=data['value'])
            self.muscle_vars[name] = var
            
            scale = tk.Scale(point_frame, from_=0.0, to=1.0, resolution=0.01,
                            orient=tk.HORIZONTAL, variable=var, length=200)
            scale.pack(side=tk.RIGHT, fill=tk.X, expand=True)
        
        # 快速预设
        preset_frame = tk.LabelFrame(main_frame, text="快速预设", padx=5, pady=5)
        preset_frame.pack(fill=tk.X, pady=5)
        
        preset_row1 = tk.Frame(preset_frame)
        preset_row1.pack(fill=tk.X, pady=2)
        tk.Button(preset_row1, text="轻度活动", command=self.preset_light, width=12).pack(side=tk.LEFT, padx=2)
        tk.Button(preset_row1, text="中度活动", command=self.preset_medium, width=12).pack(side=tk.LEFT, padx=2)
        tk.Button(preset_row1, text="高强度", command=self.preset_high, width=12).pack(side=tk.LEFT, padx=2)
        
        preset_row2 = tk.Frame(preset_frame)
        preset_row2.pack(fill=tk.X, pady=2)
        tk.Button(preset_row2, text="不对称", command=self.preset_asymmetric, width=12).pack(side=tk.LEFT, padx=2)
        tk.Button(preset_row2, text="重置", command=self.reset_values, width=12).pack(side=tk.LEFT, padx=2)
        
        # 实时预览
        preview_frame = tk.LabelFrame(main_frame, text="实时预览与调试", padx=5, pady=5)
        preview_frame.pack(fill=tk.X, pady=5)
        
        tk.Button(preview_frame, text="生成预览", command=self.generate_preview,
                 bg="#2196F3", fg="white", font=("Arial", 11, "bold")).pack(fill=tk.X, pady=2)
        
        tk.Button(preview_frame, text="显示轮廓贴合度", command=self.show_contour_fitting,
                 bg="#FF9800", fg="white", font=("Arial", 11, "bold")).pack(fill=tk.X, pady=2)
        
        tk.Button(preview_frame, text="调试插值网格", command=self.debug_interpolation,
                 bg="#9C27B0", fg="white", font=("Arial", 11, "bold")).pack(fill=tk.X, pady=2)
        
        # 生成最终结果
        generate_frame = tk.LabelFrame(main_frame, text="生成最终结果", padx=5, pady=5)
        generate_frame.pack(fill=tk.X, pady=5)
        
        tk.Button(generate_frame, text="生成单张热力图", command=self.generate_single,
                 bg="#4CAF50", fg="white", font=("Arial", 12, "bold")).pack(fill=tk.X, pady=2)
        
        tk.Button(generate_frame, text="生成所有配色方案", command=self.generate_all_colormaps,
                 bg="#795548", fg="white", font=("Arial", 12, "bold")).pack(fill=tk.X, pady=2)
        
        # 状态显示
        self.status_var = tk.StringVar(value="就绪 - 调整参数后生成热力图")
        status_label = tk.Label(main_frame, textvariable=self.status_var, 
                               font=("Arial", 9), fg="blue")
        status_label.pack(pady=5)
        
        # 启动界面
        self.adjustment_window.mainloop()
    
    def update_colormap_description(self, event=None):
        """更新配色方案描述"""
        colormap_key = self.colormap_var.get()
        if colormap_key in self.classic_colormaps:
            desc = f"{self.classic_colormaps[colormap_key]['name']}\n{self.classic_colormaps[colormap_key]['description']}"
            self.colormap_desc_var.set(desc)
    
    def preset_light(self):
        """轻度活动预设"""
        values = [0.2, 0.18, 0.15, 0.12, 0.16, 0.19, 0.17, 0.14, 0.11, 0.15]
        self.apply_preset_values(values, "轻度活动")
    
    def preset_medium(self):
        """中度活动预设"""
        values = [0.5, 0.48, 0.45, 0.42, 0.46, 0.52, 0.49, 0.44, 0.41, 0.47]
        self.apply_preset_values(values, "中度活动")
    
    def preset_high(self):
        """高强度活动预设"""
        values = [0.8, 0.78, 0.75, 0.72, 0.76, 0.82, 0.79, 0.74, 0.71, 0.77]
        self.apply_preset_values(values, "高强度活动")
    
    def preset_asymmetric(self):
        """不对称活动预设"""
        # 右侧较强
        right_values = [0.7, 0.65, 0.6, 0.5, 0.58]
        left_values = [0.4, 0.35, 0.3, 0.25, 0.32]
        
        right_names = [name for name in self.muscle_points.keys() if name.startswith('R_')]
        left_names = [name for name in self.muscle_points.keys() if name.startswith('L_')]
        
        for i, name in enumerate(right_names):
            if i < len(right_values):
                self.muscle_vars[name].set(right_values[i])
        
        for i, name in enumerate(left_names):
            if i < len(left_values):
                self.muscle_vars[name].set(left_values[i])
        
        self.status_var.set("已应用不对称活动预设")
    
    def apply_preset_values(self, values, preset_name):
        """应用预设值"""
        muscle_names = list(self.muscle_points.keys())
        for i, name in enumerate(muscle_names):
            if i < len(values):
                self.muscle_vars[name].set(values[i])
        
        self.status_var.set(f"已应用{preset_name}预设")
    
    def reset_values(self):
        """重置到默认值"""
        for name, data in self.muscle_points.items():
            self.muscle_vars[name].set(data['value'])
        self.status_var.set("已重置为默认值")
    
    def generate_preview(self):
        """生成预览"""
        self.status_var.set("生成预览中...")
        self.adjustment_window.update()
        
        try:
            fig = self.create_heatmap_figure(preview_mode=True)
            plt.show()
            self.status_var.set("预览生成完成")
        except Exception as e:
            self.status_var.set(f"预览生成失败: {str(e)}")
    
    def show_contour_fitting(self):
        """显示轮廓贴合度调试"""
        self.status_var.set("显示轮廓贴合度...")
        self.adjustment_window.update()
        
        try:
            # 创建贴合度调试图
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            fig.suptitle('轮廓贴合度调试', fontsize=16, fontweight='bold')
            
            # 原图 + 轮廓
            axes[0, 0].imshow(cv2.cvtColor(self.original_img, cv2.COLOR_BGR2RGB))
            contour_array = np.array(self.contour_points)
            axes[0, 0].plot(contour_array[:, 0], contour_array[:, 1], 'r-', linewidth=2, label='原始轮廓')
            axes[0, 0].scatter(contour_array[:, 0], contour_array[:, 1], c='red', s=50, zorder=5)
            axes[0, 0].set_title('原始轮廓')
            axes[0, 0].axis('off')
            axes[0, 0].legend()
            
            # 调整后的轮廓
            adjusted_mask = self.create_adjusted_mask()
            axes[0, 1].imshow(cv2.cvtColor(self.original_img, cv2.COLOR_BGR2RGB))
            axes[0, 1].imshow(adjusted_mask, alpha=0.5, cmap='Reds')
            axes[0, 1].set_title('调整后的掩码')
            axes[0, 1].axis('off')
            
            # 掩码边缘对比
            axes[1, 0].imshow(self.face_mask, cmap='gray')
            axes[1, 0].set_title('原始掩码')
            axes[1, 0].axis('off')
            
            axes[1, 1].imshow(adjusted_mask, cmap='gray')
            axes[1, 1].set_title('调整后掩码')
            axes[1, 1].axis('off')
            
            plt.tight_layout()
            plt.show()
            self.status_var.set("轮廓贴合度调试完成")
            
        except Exception as e:
            self.status_var.set(f"贴合度调试失败: {str(e)}")
    
    def debug_interpolation(self):
        """调试插值网格"""
        self.status_var.set("调试插值网格...")
        self.adjustment_window.update()
        
        try:
            # 获取当前肌肉点数值
            current_values = {}
            for name, var in self.muscle_vars.items():
                current_values[name] = var.get()
            
            # 创建插值网格调试图
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            fig.suptitle('插值网格调试', fontsize=16, fontweight='bold')
            
            # 测量点分布
            axes[0, 0].imshow(cv2.cvtColor(self.original_img, cv2.COLOR_BGR2RGB))
            for name, data in self.muscle_points.items():
                x, y = data['coord']
                value = current_values[name]
                color = 'red' if name.startswith('R_') else 'blue'
                axes[0, 0].scatter(x, y, c=color, s=150, alpha=0.8, edgecolors='white', linewidth=2)
                axes[0, 0].text(x+20, y-20, f'{name.split("_")[1]}\n{value:.2f}', 
                               fontsize=8, color='white', weight='bold',
                               bbox=dict(boxstyle='round', facecolor=color, alpha=0.7))
            axes[0, 0].set_title('测量点分布')
            axes[0, 0].axis('off')
            
            # 生成插值网格
            points_list = []
            values_list = []
            for name, data in self.muscle_points.items():
                x, y = data['coord']
                if 0 <= x < self.img_w and 0 <= y < self.img_h:
                    if self.face_mask[y, x] > 0.1:
                        points_list.append((x, y))
                        values_list.append(current_values[name])
            
            if len(points_list) >= 3:
                # 创建插值网格
                resolution = self.resolution_var.get()
                xi = np.linspace(0, self.img_w-1, resolution)
                yi = np.linspace(0, self.img_h-1, resolution)
                x_grid, y_grid = np.meshgrid(xi, yi)
                
                # RBF插值
                points_array = np.array(points_list)
                values_array = np.array(values_list)
                
                rbf = Rbf(points_array[:, 0], points_array[:, 1], values_array,
                         function='thin_plate', epsilon=0.1)
                grid_values = rbf(x_grid, y_grid)
                
                # 显示插值结果
                im = axes[0, 1].imshow(grid_values, extent=[0, self.img_w, self.img_h, 0], 
                                      cmap='jet', alpha=0.8)
                axes[0, 1].set_title('原始插值网格')
                plt.colorbar(im, ax=axes[0, 1])
                
                # 应用掩码后的结果
                adjusted_mask = self.create_adjusted_mask()
                mask_grid = cv2.resize(adjusted_mask, (grid_values.shape[1], grid_values.shape[0]))
                masked_values = grid_values.copy()
                masked_values[mask_grid < 0.1] = np.nan
                
                im2 = axes[1, 0].imshow(masked_values, extent=[0, self.img_w, self.img_h, 0], 
                                       cmap='jet', alpha=0.8)
                axes[1, 0].set_title('应用掩码后')
                plt.colorbar(im2, ax=axes[1, 0])
                
                # 最终结果预览
                axes[1, 1].imshow(cv2.cvtColor(self.original_img, cv2.COLOR_BGR2RGB))
                im3 = axes[1, 1].imshow(masked_values, extent=[0, self.img_w, self.img_h, 0], 
                                       cmap='jet', alpha=self.alpha_var.get())
                axes[1, 1].set_title('最终效果预览')
                axes[1, 1].axis('off')
                
            plt.tight_layout()
            plt.show()
            self.status_var.set("插值网格调试完成")
            
        except Exception as e:
            self.status_var.set(f"插值调试失败: {str(e)}")
    
    def create_adjusted_mask(self):
        """创建调整后的掩码"""
        mask = self.face_mask.copy()
        
        # 轮廓调整
        contour_adjust = self.contour_adjust_var.get()
        if contour_adjust != 0:
            kernel = np.ones((abs(contour_adjust), abs(contour_adjust)), np.uint8)
            if contour_adjust > 0:
                mask = cv2.dilate(mask, kernel, iterations=1)
            else:
                mask = cv2.erode(mask, kernel, iterations=1)
        
        # 边缘羽化
        feather = self.feather_var.get()
        if feather > 0:
            mask = cv2.GaussianBlur(mask, (int(feather*2)+1, int(feather*2)+1), feather)
        
        return mask
    
    def create_heatmap_figure(self, preview_mode=False):
        """创建热力图"""
        import matplotlib
        matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']  # 优先使用黑体或微软雅黑
        matplotlib.rcParams['axes.unicode_minus'] = False  # 正确显示负号
        # 获取当前参数
        colormap_key = self.colormap_var.get()
        colormap_config = self.classic_colormaps[colormap_key]
        
        # 创建自定义配色
        custom_cmap = LinearSegmentedColormap.from_list(
            f'custom_{colormap_key}', 
            colormap_config['colors'], 
            N=256
        )
        custom_cmap.set_bad('white', alpha=0)
        
        # 收集测量点数据
        points_list = []
        values_list = []
        
        for name, data in self.muscle_points.items():
            x, y = data['coord']
            value = self.muscle_vars[name].get()
            
            # 确保点在图像和掩码范围内
            if 0 <= x < self.img_w and 0 <= y < self.img_h:
                adjusted_mask = self.create_adjusted_mask()
                if adjusted_mask[y, x] > 0.1:
                    points_list.append((x, y))
                    values_list.append(value)
        
        if len(points_list) < 3:
            raise ValueError(f"有效测量点不足，只有{len(points_list)}个点")
        
        # 创建插值网格
        resolution = self.resolution_var.get()
        xi = np.linspace(0, self.img_w-1, resolution)
        yi = np.linspace(0, self.img_h-1, resolution)
        x_grid, y_grid = np.meshgrid(xi, yi)
        
        # RBF插值
        points_array = np.array(points_list)
        values_array = np.array(values_list)
        
        rbf = Rbf(points_array[:, 0], points_array[:, 1], values_array,
                 function='thin_plate', epsilon=0.1, smooth=0.01)
        grid_values = rbf(x_grid, y_grid)
        
        # 应用调整后的掩码
        adjusted_mask = self.create_adjusted_mask()
        mask_grid = cv2.resize(adjusted_mask, (grid_values.shape[1], grid_values.shape[0]))
        
        masked_values = grid_values.copy()
        masked_values[mask_grid < 0.1] = np.nan
        
        # 创建图形
        plt.figure(figsize=(16, 12) if not preview_mode else (12, 9))
        
        # 显示原图
        plt.imshow(cv2.cvtColor(self.original_img, cv2.COLOR_BGR2RGB))
        
        # 显示热力图
        im = plt.imshow(masked_values, 
                       extent=[0, self.img_w, self.img_h, 0],
                       alpha=self.alpha_var.get(), 
                       cmap=custom_cmap,
                       interpolation='bilinear')
        
        
        # === 等高线代码 ===
        try:
            # 增加等高线条数，减少间距
            contour_alpha = self.contour_alpha_var.get()
            levels = np.linspace(np.nanmin(masked_values), np.nanmax(masked_values), 25)
            xi = np.linspace(0, self.img_w-1, masked_values.shape[1])
            yi = np.linspace(0, self.img_h-1, masked_values.shape[0])
            X, Y = np.meshgrid(xi, yi)
            cs = plt.contour(X, Y, masked_values, levels=levels, colors='white', linewidths=1, alpha=0.7)
            plt.clabel(cs, inline=True, fontsize=8, fmt="%.2f")
            # 虚线、白色
            cs = plt.contour(
                X, Y, masked_values, 
                levels=levels, 
                colors='white', 
                linewidths=0.8, 
                alpha=0.9, 
                linestyles='dashed'
            )
            plt.clabel(cs, inline=True, fontsize=8, fmt="%.2f", colors='white')
        except Exception as e:
            print(f"等高线绘制失败: {e}")
             
        
        # 绘制测量点
        for name, data in self.muscle_points.items():
            x, y = data['coord']
            value = self.muscle_vars[name].get()
            
            # 计算点的颜色
            norm_value = (value - values_array.min()) / (values_array.max() - values_array.min())
            point_color = custom_cmap(norm_value)
            
            marker = 'o' if name.startswith('R_') else 's'
            plt.scatter(x, y, s=150, c=[point_color], marker=marker,
                       edgecolors='white', linewidths=2, zorder=10)
            
            # 添加标签
            muscle_name = name.split('_')[1]
            side = 'R' if name.startswith('R_') else 'L'
            offset_x = 140 if name.startswith('R_') else -160
            offset_y = -40

            
            
            plt.text(x + offset_x, y + offset_y, f'{side}-{muscle_name}', 
                    color='white', fontweight='bold', fontsize=9,
                    bbox=dict(facecolor='black', alpha=0.8, boxstyle='round,pad=0.3'),
                    ha='center', va='center', zorder=12)
        
        # 添加面部中心线
        plt.axvline(x=self.face_center, color='lime', linestyle='--', alpha=0.8, linewidth=2)
        
        # 添加颜色条
        cbar = plt.colorbar(im, shrink=0.8, aspect=25, pad=0.02)
        def spaced_text(text):
            return ' '.join(list(text))
        cbar.set_label('Activity Intensity', rotation=270, labelpad=20, fontsize=12, fontweight='bold')
        
        # 设置标题
        def spaced_text(text):
            return ' '.join(list(text))
        
        title = spaced_text('ENGLISH FACIAL MOVEMENT')
        plt.title(title, fontsize=16, fontweight='bold', pad=20)
        plt.axis('off')

        # 添加图例
        legend_elements = [
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='gray', markersize=10, 
                      label='right', markeredgecolor='white', markeredgewidth=2),
            plt.Line2D([0], [0], marker='s', color='w', markerfacecolor='gray', markersize=10, 
                      label='left', markeredgecolor='white', markeredgewidth=2),
            plt.Line2D([0], [0], color='lime', linestyle='--', linewidth=2, label='centerline')
        ]
        
        plt.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(0.98, 0.98))
        plt.tight_layout()

        if len(self.contour_points) > 3:
            contour_array = np.array(self.contour_points)
            # 闭合轮廓
            x = np.r_[contour_array[:, 0], contour_array[0, 0]]
            y = np.r_[contour_array[:, 1], contour_array[0, 1]]
            try:
                tck, u = splprep([x, y], s=3.0, per=1)
                unew = np.linspace(0, 1.0, 400)
                out = splev(unew, tck)
                
            except Exception as e:
                # 插值失败也不要画线
                plt.plot(x, y, linewidth=0, alpha=0.9, zorder=20)
        
        return plt.gcf()
    
    def generate_single(self):
        """生成单张热力图"""
        self.status_var.set("正在生成热力图...")
        self.adjustment_window.update()
        
        try:
            fig = self.create_heatmap_figure()
            
            # 保存图像
            colormap_key = self.colormap_var.get()
            output_path = os.path.join(self.output_dir, f'classic_heatmap_{colormap_key}.png')
            fig.savefig(output_path, bbox_inches='tight', dpi=300, facecolor='white')
            plt.show()
            
            self.status_var.set(f"热力图已保存: {output_path}")
            
        except Exception as e:
            self.status_var.set(f"生成失败: {str(e)}")
    
    def generate_all_colormaps(self):
        """生成所有配色方案"""
        self.status_var.set("正在生成所有配色方案...")
        self.adjustment_window.update()
        
        original_colormap = self.colormap_var.get()
        
        for i, colormap_key in enumerate(self.classic_colormaps.keys()):
            try:
                self.colormap_var.set(colormap_key)
                fig = self.create_heatmap_figure()
                
                output_path = os.path.join(self.output_dir, f'classic_heatmap_{colormap_key}.png')
                fig.savefig(output_path, bbox_inches='tight', dpi=300, facecolor='white')
                plt.close(fig)
                
                self.status_var.set(f"已生成 {i+1}/{len(self.classic_colormaps)}: {colormap_key}")
                self.adjustment_window.update()
                
            except Exception as e:
                self.status_var.set(f"生成{colormap_key}失败: {str(e)}")
        
        # 恢复原始选择
        self.colormap_var.set(original_colormap)
        self.status_var.set(f"所有配色方案已生成完成 (共{len(self.classic_colormaps)}张)")


def generate_classic_face_heatmap(img, contour_points, face_center=None, output_dir='output'):
    """生成经典配色的面部热力图"""
    h, w = img.shape[:2]
    
    if face_center is None:
        face_center = w // 2
    
    print(f"使用面部中心线X坐标: {face_center}")
    print("正在启动经典热力图生成器...")
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 启动调整工具
    generator = ClassicHeatmapGenerator(img, contour_points, face_center, output_dir)
    generator.open_adjustment_interface()


def run_editor(image_path, initial_detector='mediapipe', contour_path=None, center_x=None):
    """运行交互式编辑器"""
    editor = FaceContourEditor(image_path, initial_detector, contour_path, center_x)
    editor.root.mainloop()
    
    # 返回最终的轮廓点和中心点
    return editor.contour_points, editor.center_x


def parse_arguments():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='经典配色面部热力图生成器')
    parser.add_argument('--image', type=str, default=os.path.join('data', 'Englishface.jpg'),
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
    
    print(f"开始启动经典热力图生成器，使用{len(contour_points)}个轮廓点...")
    
    # 启动热力图生成器
    generate_classic_face_heatmap(
        img, 
        contour_points,
        face_center=center_x,
        output_dir=args.output
    )
    
    print("经典热力图生成器已完成！")


# 示例用法和测试函数
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


def demo_classic_heatmap():
    """演示经典热力图生成"""
    print("创建测试数据...")
    test_img, contour_points, center_x = create_test_data()
    
    # 保存测试图像
    os.makedirs('Facespeak_v2_data', exist_ok=True)
    cv2.imwrite('Facespeak_v2_data/test_face.jpg', test_img)
    
    print("启动经典热力图生成器演示...")
    generate_classic_face_heatmap(
        test_img, 
        contour_points, 
        face_center=center_x, 
        output_dir='Facespeak_v2_data/output'
    )


if __name__ == "__main__":
    import sys
    
    # 检查是否是演示模式
    if len(sys.argv) > 1 and sys.argv[1] == 'demo':
        demo_classic_heatmap()
    else:
        try:
            main()
        except Exception as e:
            import traceback
            print(f"发生错误: {e}")
            traceback.print_exc()