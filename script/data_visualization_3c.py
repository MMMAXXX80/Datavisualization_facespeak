import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import os

# ─── 设置中文字体 ───
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['axes.unicode_minus'] = False

BASE_DIR = Path(__file__).parent.parent
DATA_FILE = BASE_DIR / "data" / "processed" / "combined_cleaned_data.csv"
OUTPUT_DIR = BASE_DIR / "output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def load_processed_data():
    df = pd.read_csv(DATA_FILE)
    print("数据维度:", df.shape)
    return df

def generate_statistical_radar_chart(df, output_name="radar_stats.png"):
    # 保证有6个通道
    numeric = df.select_dtypes(include=[np.number]).columns.tolist()
    if len(numeric) < 6:
        for i in range(6 - len(numeric)):
            df[f"虚拟通道{i+1}"] = 0
        numeric = df.select_dtypes(include=[np.number]).columns.tolist()
    channels = numeric[:6]

    # 统计量
    stats_df = df[channels].agg(['mean','max','min','std','median']).T
    stats_order = ['mean','max','min','std','median']
    stats_names = {'mean':'均值','max':'最大值','min':'最小值','std':'标准差','median':'中位数'}

    # 角度和封闭
    N = len(channels)
    angles = np.linspace(0, 2*np.pi, N, endpoint=False).tolist()
    angles += angles[:1]

    # 图形
    fig, ax = plt.subplots(figsize=(12, 8), subplot_kw=dict(polar=True))
    fig.subplots_adjust(left=0.1, right=0.6, top=0.9, bottom=0.1)

    # 绘图（只折线，无填充）
    colors = ['tab:blue','tab:green','tab:red','tab:orange','tab:purple']
    for idx, stat in enumerate(stats_order):
        vals = stats_df[stat].tolist()
        vals += vals[:1]
        ax.plot(angles, vals, color=colors[idx], linewidth=2, label=stats_names[stat])

    ax.set_yticklabels([])
    ax.set_xticks([])

    # 调整文字距离圆圈远一些
    r_max = stats_df.values.max() * 1
    r_label = r_max * 1.2
    for i, ch in enumerate(channels):
        ax.text(angles[i], r_label, ch, ha='center', va='center', fontsize=10)

    ax.set_title("西班牙语面部发音动态关键数据", pad=20, fontsize=16)

    # 右侧通道说明
    channel_notes = [
        "channel_1：唇下肌肉点位",
        "channel_2：唇上肌肉点位",
        "channel_3：咬合肌肉点位",
        "channel_4：颞下颌关节点位",
        "channel_5：锁骨对照组点位",
        "channel_6：额头对照组点位"
    ]
    text_x = 0.65
    text_y_start = 0.9
    line_spacing = 0.06
    for i, note in enumerate(channel_notes):
        fig.text(text_x, text_y_start - i*line_spacing, note,
                 fontsize=10, ha='left', va='top')

    ax.legend(loc='lower left', bbox_to_anchor=(0.02, 0.02), fontsize=10)
    plt.savefig(OUTPUT_DIR / output_name, dpi=300, bbox_inches='tight')
    plt.close()
    print("已保存：", output_name)

def main():
    df = load_processed_data()
    generate_statistical_radar_chart(df)

if __name__ == "__main__":
    main()
