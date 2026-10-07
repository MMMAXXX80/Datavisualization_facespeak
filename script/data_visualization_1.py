import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import os
import numpy as np

# 设置路径
BASE_DIR = Path(__file__).parent.parent
DATA_FILE = BASE_DIR / "data" / "processed" / "combined_cleaned_data.csv"
OUTPUT_DIR = BASE_DIR / "output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 自定义配色方案
COLOR_PALETTES = {
    "科学": ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"],
    "柔和": ["#66c2a5", "#fc8d62", "#8da0cb", "#e78ac3", "#a6d854", "#ffd92f"],
    "深色": ["#393b79", "#637939", "#8c6d31", "#843c39", "#7b4173", "#5254a3"],
    "明亮": ["#e41a1c", '#0000FF', "#4daf4a", "#984ea3", "#ff7f00", "#ffff33"],
    "冷色调": ["#1b9e77", "#7570b3", "#d95f02", "#e7298a", "#66a61e", "#e6ab02"],
}

def load_processed_data():
    """加载处理后的数据"""
    print(f"\n正在加载数据文件: {DATA_FILE}")
    if not DATA_FILE.exists():
        raise FileNotFoundError(f"错误：数据文件不存在于 {DATA_FILE}")
    
    df = pd.read_csv(DATA_FILE)
    print("数据加载成功，维度:", df.shape)
    print("可用列名:", df.columns.tolist())
    return df

def plot_multi_channel(df, channels=None, figsize=(14, 8), palette_name="科学"):
    """
    绘制多通道折线图（带自定义配色）
    参数:
        df: 包含时序数据的DataFrame
        channels: 要绘制的列名列表，若为None则自动选择数值列
        figsize: 图像尺寸
        palette_name: 配色方案名称（"科学"、"柔和"、"深色"、"明亮"、"冷色调"）
    """
    # 自动选择数值型列（如果未指定通道）
    if channels is None:
        channels = df.select_dtypes(include=[np.number]).columns.tolist()
        print("\n自动选择数值型列进行绘制:", channels)
    
    # 设置样式和配色
    sns.set_style('white')
    
    # 获取配色方案
    if palette_name not in COLOR_PALETTES:
        print(f"警告: 配色方案 '{palette_name}' 不存在，使用默认方案")
        palette = COLOR_PALETTES["科学"]
    else:
        palette = COLOR_PALETTES[palette_name]
    
    # 创建画布
    plt.figure(figsize=(24, 5), dpi=120) 
    ax = plt.gca()
    
    # 绘制每个通道
    for i, channel in enumerate(channels):
        ax.plot(
            df.index,  # 使用索引作为x轴
            df[channel],
            color=palette[i % len(palette)],  # 使用自定义配色
            linewidth=2,
            alpha=0.9,
            marker='o' if len(df) < 30 else None,
            markersize=6,
            label=channel
        )
    
    # 美化图表
    ax.set_title(f"Facial Muscle Activation_Russian", pad=20, fontsize=14)
    ax.set_xlabel("time", labelpad=10)
    ax.set_ylabel("data", labelpad=10)
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # 自动旋转x轴标签
    plt.xticks(rotation=45 if len(df) > 12 else 0)
    
    # 保存输出
    output_path = OUTPUT_DIR / f"multi_channel_lines_{palette_name}.png"
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"\n图表已保存至: {output_path}")

def main():
    """主执行函数"""
    print("=== 开始多通道数据可视化 ===")
    print("可选配色方案:", list(COLOR_PALETTES.keys()))
    
    try:
        # 1. 加载数据
        df = load_processed_data()
        
        # 2. 绘制折线图（示例选择前5个数值列）
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        plot_cols = numeric_cols[:5] if len(numeric_cols) >= 5 else numeric_cols
        
        # 3. 使用不同配色方案绘制图表
        for palette_name in COLOR_PALETTES:
            plot_multi_channel(df, channels=plot_cols, palette_name=palette_name)
        
        print("\n=== 可视化完成 ===")
        return True
    except Exception as e:
        print(f"\n!!! 错误: {str(e)}")
        return False

if __name__ == "__main__":
    if not main():
        exit(1)