import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
from PIL import Image

# 参数设置
CSV_PATH = "data/processed/combined_cleaned_data.csv"
OUTPUT_DIR = "outputs/frames"
GIF_PATH = "outputs/averaged_animation.gif"
CHANNEL_LABELS = ['channel_1', 'channel_2', 'channel_3', 'channel_4', 'channel_5', 'channel_6']
GROUP_SIZE = 100
GIF_DURATION = 300  # 每帧时间 (ms)

# 确保目录存在
def ensure_dirs():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(os.path.dirname(GIF_PATH), exist_ok=True)

# 加载数据
def load_data():
    df = pd.read_csv(CSV_PATH)
    print(f"✅ 读取数据文件成功：{CSV_PATH}")
    
    if not set(CHANNEL_LABELS).issubset(df.columns):
        found = [col for col in df.columns if "channel" in col.lower()]
        print(f"⚠️ 数据中实际通道列为：{found}")
        raise ValueError("❌ CHANNEL_LABELS 与 CSV 列名不匹配。请检查。")
    
    return df[CHANNEL_LABELS]

# 设置中文字体
def set_chinese_font():
    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
    plt.rcParams['axes.unicode_minus'] = False

# 绘制雷达图
def plot_radar_frame(averaged_values, group_idx):
    # 计算雷达图的角度
    angles = np.linspace(0, 2 * np.pi, len(CHANNEL_LABELS), endpoint=False).tolist()
    
    # 创建数据的闭合版本
    values = np.concatenate((averaged_values, [averaged_values[0]]))
    angles += angles[:1]
    
    # 创建图形
    fig, ax = plt.subplots(figsize=(8, 8), dpi=100, subplot_kw=dict(polar=True), facecolor='white')
    
    # 设置背景和轴线为白色
    ax.set_facecolor('white')
    
    # 固定的网格线半径
    grid_radii = [0.25, 0.5, 0.75, 1.0]
    
    # 手动绘制同心圆网格线
    for radius in grid_radii:
        circle = plt.Circle((0, 0), radius, 
                            fill=False, 
                            edgecolor='lightgray', 
                            linestyle='--', 
                            alpha=0.5)
        ax.add_artist(circle)
    
    # 绘制雷达图轮廓（橙色细线）
    # 标准化数据到 0-1 范围
    max_val = max(averaged_values)
    normalized_values = averaged_values / max_val
    normalized_values = np.concatenate((normalized_values, [normalized_values[0]]))
    
    ax.plot(angles, normalized_values, linewidth=2, color='brown')
    
    # 配置轴
    ax.set_ylim(0, 1.2)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    
    # 添加通道标签
    for i, label in enumerate(CHANNEL_LABELS):
        angle = angles[i]
        
        # 调整标签位置到更远的位置
        radius = 1.4  # 固定位置
        
        plt.text(
            angle, 
            radius, 
            f"{label}\n(Max: {averaged_values[i]:.2f})", 
            ha='center', 
            va='center', 
            fontsize=9,
            fontname='Microsoft YaHei'
        )
    
    # 添加标题，调整位置和间距
    plt.title("阿拉伯语面部发音动态", 
              fontname='Microsoft YaHei', 
              fontsize=16, 
              pad=40,  # 增加标题与图形的距离
              y=1)  # 将标题抬高
    
    # 调整整体布局，给更多空间
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    # 确保输出目录存在
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # 保存帧图
    path = os.path.join(OUTPUT_DIR, f"frame_{group_idx:03d}.png")
    plt.savefig(path, bbox_inches='tight', dpi=300, facecolor='white', edgecolor='none')
    plt.close()
    
    return path

# 生成每一帧的图像
def generate_frames(df):
    num_groups = len(df) // GROUP_SIZE
    print(f"🔄 共生成 {num_groups} 组图像（每 {GROUP_SIZE} 行取平均）")
    
    frame_paths = []
    for i in tqdm(range(num_groups), desc="📊 正在生成帧图像"):
        chunk = df.iloc[i * GROUP_SIZE: (i + 1) * GROUP_SIZE]
        avg_values = chunk.mean().values
        path = plot_radar_frame(avg_values, i)
        frame_paths.append(path)
    return frame_paths

# 创建GIF
def create_gif(frame_paths, gif_path):
    frames = [Image.open(p) for p in frame_paths]
    frames[0].save(
        gif_path,
        format='GIF',
        append_images=frames[1:],
        save_all=True,
        duration=GIF_DURATION,
        loop=0
    )
    print(f"🎞️ GIF 动画已保存：{gif_path}")


# 主程序
def main():
    set_chinese_font()  # 设置中文字体
    ensure_dirs()
    df = load_data()
    frame_paths = generate_frames(df)
    create_gif(frame_paths, GIF_PATH)

if __name__ == "__main__":
    main()