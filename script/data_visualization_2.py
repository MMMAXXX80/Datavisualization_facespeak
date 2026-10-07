import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

# 读取数据
DATA_PATH = os.path.join("data", "processed", "combined_cleaned_data.csv")
df = pd.read_csv(DATA_PATH)

# 选择通道数据
channels = ['channel_1', 'channel_2', 'channel_3', 'channel_4', 'channel_5', 'channel_6']
data = df[channels]

# 自定义渐变色：
custom_cmap = LinearSegmentedColormap.from_list(
    "blue_green_orange",
    ['#FFFFCC', '#FF6B6B', '#6A5ACD', '#2E004F'],  # 蛋黄 → 橘红 → 红 → 蓝 → 蓝紫 
    N=256
)

# 绘图
plt.figure(figsize=(28, 6))
sns.heatmap(
    data.T,
    cmap=custom_cmap,
    cbar=True,
    vmin=0,       # 最小值映射为蓝色
    vmax=500      # 最大值映射为橘色
)

# 设置标签
plt.title("Facial Muscle Activation_Chinese", fontsize=16)
plt.xlabel("Time Frame")
plt.ylabel("Channels")
plt.yticks(ticks=range(len(channels)), labels=channels, rotation=0)

# 保存图像
output_path = os.path.join("output", "facial_movement_heatmap.png")
plt.tight_layout()
plt.savefig(output_path)
plt.close()

print(f"Heatmap saved to {output_path}")

