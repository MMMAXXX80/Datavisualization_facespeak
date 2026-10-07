import cv2
import numpy as np
import os

def detect_blue_circles(image_path, output_dir='data/output'):
    """
    检测图像中的蓝色圆圈并返回中心坐标
    优化了VSCode中的显示效果
    """
    # 读取图像
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"无法加载图像，请检查路径: {image_path}")
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)

    # 调整显示窗口大小（适合VSCode）
    def resize_window(img, window_name, scale=0.3):
        h, w = img.shape[:2]
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, int(w*scale), int(h*scale))

    # 调试模式：颜色范围调整窗口
    def nothing(x):
        pass

    cv2.namedWindow('Color Adjust', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Color Adjust', 600, 300)
    cv2.createTrackbar('H Min', 'Color Adjust', 90, 179, nothing)
    cv2.createTrackbar('H Max', 'Color Adjust', 140, 179, nothing)
    cv2.createTrackbar('S Min', 'Color Adjust', 50, 255, nothing)
    cv2.createTrackbar('S Max', 'Color Adjust', 255, 255, nothing)
    cv2.createTrackbar('V Min', 'Color Adjust', 50, 255, nothing)
    cv2.createTrackbar('V Max', 'Color Adjust', 255, 255, nothing)
    cv2.createTrackbar('Min Area', 'Color Adjust', 50, 500, nothing)
    cv2.createTrackbar('Circularity', 'Color Adjust', 70, 100, nothing)

    print("调试模式说明:")
    print("1. 调整滑动条直到蓝色圆圈被正确识别")
    print("2. 按 's' 键保存结果")
    print("3. 按 'q' 键退出")

    while True:
        # 获取滑动条值
        h_min = cv2.getTrackbarPos('H Min', 'Color Adjust')
        h_max = cv2.getTrackbarPos('H Max', 'Color Adjust')
        s_min = cv2.getTrackbarPos('S Min', 'Color Adjust')
        s_max = cv2.getTrackbarPos('S Max', 'Color Adjust')
        v_min = cv2.getTrackbarPos('V Min', 'Color Adjust')
        v_max = cv2.getTrackbarPos('V Max', 'Color Adjust')
        min_area = cv2.getTrackbarPos('Min Area', 'Color Adjust')
        circularity_thresh = cv2.getTrackbarPos('Circularity', 'Color Adjust') / 100

        # 颜色检测
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        lower_blue = np.array([h_min, s_min, v_min])
        upper_blue = np.array([h_max, s_max, v_max])
        mask = cv2.inRange(hsv, lower_blue, upper_blue)
        
        # 形态学操作
        kernel = np.ones((5,5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        
        # 查找轮廓
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # 绘制结果
        result_img = img.copy()
        blue_circle_centers = []
        
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < min_area:
                continue
                
            (x,y), radius = cv2.minEnclosingCircle(cnt)
            center = (int(x), int(y))
            radius = int(radius)
            
            perimeter = cv2.arcLength(cnt, True)
            circularity = 4 * np.pi * area / (perimeter**2) if perimeter > 0 else 0
            
            if circularity > circularity_thresh:
                blue_circle_centers.append(center)
                cv2.circle(result_img, center, radius, (0,255,0), 2)
                cv2.circle(result_img, center, 2, (0,0,255), 3)
                cv2.putText(result_img, f"{len(blue_circle_centers)}", 
                            (center[0]+10, center[1]), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 2)
        
        # 显示结果（缩放适合窗口）
        resize_window(mask, 'Mask', 0.5)
        resize_window(result_img, 'Result', 0.5)
        cv2.imshow('Mask', mask)
        cv2.imshow('Result', result_img)
        
        # 按键处理
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            # 保存结果
            output_img_path = os.path.join(output_dir, 'detected_blue_circles.jpg')
            cv2.imwrite(output_img_path, result_img)
            
            coord_file = os.path.join(output_dir, 'blue_circle_coordinates.txt')
            with open(coord_file, 'w') as f:
                for x, y in blue_circle_centers:
                    f.write(f"{x},{y}\n")
            
            print(f"\n保存结果: 找到 {len(blue_circle_centers)} 个蓝色圆圈")
            print(f"标记图像: {output_img_path}")
            print(f"坐标文件: {coord_file}")
            break
    
    cv2.destroyAllWindows()
    return blue_circle_centers

# 使用示例
image_path = os.path.join('data', 'Englishface.jpg')
centers = detect_blue_circles(image_path)

# 打印检测结果
print("\n检测到的蓝色圆心坐标(x,y):")
for i, (x, y) in enumerate(centers, 1):
    print(f"圆圈 {i}: ({x}, {y})")