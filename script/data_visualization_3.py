import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 中文支持
plt.rcParams['font.family'] = 'Microsoft YaHei'

# 通道标签
CHANNEL_LABELS = ['channel_1', 'channel_2', 'channel_3', 'channel_4', 'channel_5', 'channel_6']

def load_combined_data(folder='data/processed/combined_cleaned_data'):
    """合并combined_cleaned_data文件夹下所有csv为一个DataFrame"""
    if not os.path.exists(folder):
        print(f"❌ 路径不存在: {folder}")
        print("📁 当前目录下的文件夹结构:")
        if os.path.exists('data'):
            print("   data/")
            if os.path.exists('data/processed'):
                print("     processed/")
                processed_folders = os.listdir('data/processed')
                for subfolder in processed_folders:
                    subfolder_path = os.path.join('data/processed', subfolder)
                    if os.path.isdir(subfolder_path):
                        print(f"       {subfolder}/")
            else:
                print("     (processed文件夹不存在)")
        else:
            print("   (data文件夹不存在)")

        possible_paths = [
            'data/processed/combined',
            'data/combined_cleaned_data',
            'combined_cleaned_data',
            'data/processed'
        ]

        print("\n🔍 尝试寻找可能的数据路径:")
        for path in possible_paths:
            if os.path.exists(path):
                csv_files = [f for f in os.listdir(path) if f.endswith('.csv')]
                if csv_files:
                    print(f"   ✅ 找到路径: {path} (包含 {len(csv_files)} 个CSV文件)")
                    folder = path
                    break
                else:
                    print(f"   📁 路径存在但没有CSV文件: {path}")
            else:
                print(f"   ❌ 路径不存在: {path}")
        else:
            raise FileNotFoundError(f"无法找到包含CSV文件的数据路径")

    all_files = [os.path.join(folder, f) for f in os.listdir(folder) if f.endswith('.csv')]

    if not all_files:
        raise FileNotFoundError(f"在路径 {folder} 中没有找到CSV文件")

    print(f"📂 使用数据路径: {folder}")
    print(f"📄 找到CSV文件: {[os.path.basename(f) for f in all_files]}")

    dfs = []
    for file in all_files:
        try:
            df_temp = pd.read_csv(file)
            dfs.append(df_temp)
            print(f"   ✅ 读取文件: {os.path.basename(file)} ({len(df_temp)} 行)")
        except Exception as e:
            print(f"   ❌ 读取文件失败: {os.path.basename(file)} - {e}")

    if not dfs:
        raise ValueError("没有成功读取任何CSV文件")

    df = pd.concat(dfs, ignore_index=True)
    print(f"✅ 合并 {len(all_files)} 个文件，总行数: {len(df)}")
    print(f"📊 数据列: {list(df.columns)}")

    return df

def plot_radar(stats_dict, labels, stats_to_plot, title, output_path):
    import numpy as np
    import matplotlib.pyplot as plt

    num_vars = len(labels)
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(9, 9), subplot_kw=dict(polar=True))

    # 颜色顺序
    colors = ['blue', 'green', 'red', 'orange', 'pink']
    legend_labels = ['均值', '最大值', '最小值', '标准差', '中位数']

    # 如果只绘制 Mean，设置为黑色
    if stats_to_plot == ['Mean']:
        values = stats_dict['Mean']
        v = values.tolist() if isinstance(values, pd.Series) else list(values)
        v += v[:1]
        ax.plot(angles, v, linewidth=2, color='black', label='均值')
    else:
        for i, stat in enumerate(stats_to_plot):
            values = stats_dict[stat]
            v = values.tolist() if isinstance(values, pd.Series) else list(values)
            v += v[:1]
            ax.plot(angles, v, linewidth=2, color=colors[i % len(colors)], label=legend_labels[i % len(legend_labels)])

    # 设置 channel 标签离雷达图远一些，使用 labelpad 参数
    ax.set_thetagrids(np.degrees(angles[:-1]), labels, fontsize=13)

    # 用手动方法让标签看起来离雷达图远：设置极径范围和中心位置
    ax.set_ylim(0, 1)
    ax.tick_params(pad=20)  # 控制所有刻度标签的距离

    # 去除径向刻度
    ax.set_ylim(0, 1)
    ax.set_yticklabels([])

    # 美化图形
    ax.spines['polar'].set_visible(False)
    ax.grid(True, linestyle='dotted', linewidth=0.8)

    # 设置标题并手动移动位置
    ax.set_title(title, va='bottom', fontsize=16, fontweight='bold')
    ax.title.set_position([0.5, 1.1])  # 控制标题位置

    # 图例放右上角
    ax.legend(loc='upper right', bbox_to_anchor=(1.2, 1.0))

    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight', dpi=150)
    plt.close()
    print(f"✅ 雷达图已保存: {output_path}")


def calculate_statistics(df):
    stats = {
        'Mean': df.mean(),
        'Max': df.max(),
        'Min': df.min(),
        'Std': df.std(),
        'Median': df.median(),
        'Skewness': df.skew(),
        'Kurtosis': df.kurtosis()
    }
    return stats

def normalize_data_by_column(df):
    return (df - df.min()) / (df.max() - df.min())

def normalize_stats_for_radar(stats):
    all_values = pd.DataFrame(stats).T
    normalized_stats = {}
    for stat_name in stats.keys():
        values = all_values.loc[stat_name]
        min_val = values.min()
        max_val = values.max()
        if max_val > min_val:
            normalized_stats[stat_name] = (values - min_val) / (max_val - min_val)
        else:
            normalized_stats[stat_name] = values * 0
    return normalized_stats

def print_statistics_table(stats, channel_labels):
    print("\n" + "="*100)
    print("📊 通道统计数据表")
    print("="*100)

    stats_df = pd.DataFrame(stats).T
    stats_df.to_csv('channel_statistics.csv')
    print(stats_df.round(4))
    print("="*100)

    print("\n📖 统计指标说明：")
    print("   Mean (均值): 数据的平均值")
    print("   Max (最大值): 数据的最大值") 
    print("   Min (最小值): 数据的最小值")
    print("   Std (标准差): 数据的离散程度")
    print("   Median (中位数): 数据的中位数")
    print("   Skewness (偏度): 数据分布的偏斜程度 (>0右偏, <0左偏)")
    print("   Kurtosis (峰度): 数据分布的尖锐程度 (>0尖峭, <0平缓)")
    print("="*100)

def main():
    try:
        df = load_combined_data()

        print(f"\n📋 数据概览:")
        print(f"   数据形状: {df.shape}")
        print(f"   列名: {list(df.columns)}")

        missing_channels = [col for col in CHANNEL_LABELS if col not in df.columns]
        if missing_channels:
            print(f"⚠️  警告: 以下通道列不存在: {missing_channels}")
            available_channels = [col for col in CHANNEL_LABELS if col in df.columns]
            if not available_channels:
                if len(df.columns) >= 6:
                    available_channels = list(df.columns[-6:])
                    print(f"🔄 使用最后6列作为通道数据: {available_channels}")
                else:
                    available_channels = list(df.columns)
                    print(f"🔄 使用所有列作为通道数据: {available_channels}")
            else:
                print(f"✅ 使用找到的通道列: {available_channels}")
        else:
            available_channels = CHANNEL_LABELS
            print(f"✅ 所有指定通道列都存在: {available_channels}")

        channel_df = df[available_channels]
        stats = calculate_statistics(channel_df)
        print_statistics_table(stats, available_channels)
        norm_stats = normalize_stats_for_radar(stats)

        # ✅ 均值雷达图为黑色线条
        mean_values = norm_stats['Mean']
        plot_radar({'Mean': mean_values}, available_channels, ['Mean'], 
                   "俄语面部发音动态图", "mean_radar.png")

        # 其他雷达图
        basic_stats = ['Mean', 'Max', 'Min', 'Median']
        stats_to_plot = ['Mean', 'Max', 'Min', 'Std', 'Median']
        plot_radar(norm_stats, available_channels, stats_to_plot,
                   "俄语面部发音动态图", "statistics_radar.png")
        plot_radar(norm_stats, available_channels, basic_stats, 
                   "俄语面部发音动态关键数据", "basic_statistics_radar.png")

        print("\n🎯 分析完成！生成的图表：")
        print("   1. mean_radar.png - 均值雷达图")
        print("   2. statistics_radar.png - 全部基础统计图")
        print("   3. basic_statistics_radar.png - 精简统计图")

    except Exception as e:
        print(f"❌ 程序运行出错: {e}")

if __name__ == "__main__":
    main()
