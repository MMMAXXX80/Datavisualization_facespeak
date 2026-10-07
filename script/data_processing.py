import pandas as pd
import os
import glob
from datetime import datetime
import numpy as np
from pathlib import Path

# 设置路径 - 使用Path对象更安全
BASE_DIR = Path(__file__).parent.parent  # 项目根目录
DATA_RAW_DIR = BASE_DIR / 'data' / 'raw'            # 原始数据目录
DATA_PROCESSED_DIR = BASE_DIR / 'data' / 'processed' # 处理后的数据目录
OUTPUT_FILENAME = 'combined_cleaned_data.csv'        # 合并后的文件名
LOG_FILE = 'data_processing.log'                     # 日志文件名

def setup_directories():
    """确保所有需要的目录都存在"""
    DATA_RAW_DIR.mkdir(parents=True, exist_ok=True)
    DATA_PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

def setup_logging():
    """初始化日志记录"""
    with open(DATA_PROCESSED_DIR / LOG_FILE, 'w') as f:
        f.write(f"数据处理日志 {datetime.now()}\n")
        f.write("="*50 + "\n")

def log_message(message):
    """记录处理日志"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(DATA_PROCESSED_DIR / LOG_FILE, 'a') as f:
        f.write(f"[{timestamp}] {message}\n")
    print(message)  # 同时在控制台输出

def clean_data(df, file_name):
    """
    数据清理函数
    参数:
        df: 要清理的DataFrame
        file_name: 原始文件名(用于日志记录)
    返回:
        清理后的DataFrame
    """
    original_rows = len(df)
    
    # 1. 处理缺失值
    # 删除所有值为空的列
    df = df.dropna(axis=1, how='all')
    
    # 对数值列用中位数填充，分类列用众数填充
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            fill_value = df[col].median()
        else:
            fill_value = df[col].mode()[0] if not df[col].mode().empty else 'MISSING'
        df[col] = df[col].fillna(fill_value)
    
    # 2. 删除完全重复的行
    df = df.drop_duplicates()
    
    # 3. 数据类型转换
    # 将所有可能是数值的列尝试转换为数值
    for col in df.select_dtypes(include=['object']).columns:
        df[col] = pd.to_numeric(df[col], errors='ignore')
    
    # 4. 异常值处理（对数值列使用3σ原则）
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        mean = df[col].mean()
        std = df[col].std()
        if std > 0:  # 避免除零错误
            df[col] = np.where(
                (df[col] < mean - 3*std) | (df[col] > mean + 3*std),
                mean,
                df[col]
            )
    
    cleaned_rows = len(df)
    log_message(f"文件 {file_name}: 原始行数 {original_rows}, 清理后行数 {cleaned_rows}, 删除行数 {original_rows - cleaned_rows}")
    
    return df

def standardize_columns(df_list):
    """
    标准化所有DataFrame的列名
    参数:
        df_list: 包含多个DataFrame的列表
    返回:
        统一列名后的DataFrame列表
    """
    # 获取所有DataFrame的所有列名
    all_columns = set()
    for df in df_list:
        all_columns.update(df.columns)
    
    # 创建标准列名映射（转为小写并替换空格/特殊字符）
    column_mapping = {
        col: col.strip().lower()
             .replace(' ', '_')
             .replace('-', '_')
             .replace('/', '_')
             .replace('\\', '_')
             .replace('.', '_')
        for col in all_columns
    }
    
    # 应用列名标准化
    standardized_dfs = []
    for df in df_list:
        df = df.rename(columns=column_mapping)
        # 确保所有DataFrame有相同的列（缺失的列填充NaN）
        for std_col in column_mapping.values():
            if std_col not in df.columns:
                df[std_col] = np.nan
        standardized_dfs.append(df)
    
    return standardized_dfs

def find_data_files():
    """查找所有数据文件，支持多种格式"""
    extensions = ['csv', 'xlsx', 'xls', 'json']
    files = []
    for ext in extensions:
        files.extend(glob.glob(str(DATA_RAW_DIR / f'*.{ext}')))
    return files

def read_data_file(file_path):
    """根据文件扩展名读取数据文件"""
    ext = os.path.splitext(file_path)[1].lower()
    try:
        if ext == '.csv':
            return pd.read_csv(file_path, sep=None, engine='python')
        elif ext in ('.xlsx', '.xls'):
            return pd.read_excel(file_path)
        elif ext == '.json':
            return pd.read_json(file_path)
        else:
            log_message(f"不支持的文件格式: {file_path}")
            return None
    except Exception as e:
        log_message(f"读取文件 {file_path} 时出错: {str(e)}")
        return None

def process_data():
    """主处理函数"""
    setup_directories()
    setup_logging()
    log_message("开始数据处理流程")
    
    try:
        # 获取所有数据文件
        all_files = find_data_files()
        if not all_files:
            raise FileNotFoundError(f"在 {DATA_RAW_DIR} 目录中未找到任何数据文件")
        
        log_message(f"找到 {len(all_files)} 个数据文件")
        
        # 读取并清理每个文件
        dfs = []
        for file in all_files:
            try:
                file_name = os.path.basename(file)
                log_message(f"正在处理文件: {file_name}")
                
                # 读取数据文件
                df = read_data_file(file)
                if df is None or df.empty:
                    log_message(f"文件 {file_name} 为空或读取失败，跳过")
                    continue
                
                # 清理数据
                cleaned_df = clean_data(df, file_name)
                dfs.append(cleaned_df)
                
            except Exception as e:
                log_message(f"处理文件 {file_name} 时出错: {str(e)}")
                continue
        
        if not dfs:
            raise ValueError("没有成功加载任何数据文件")
        
        # 标准化列名
        log_message("正在标准化列名...")
        dfs = standardize_columns(dfs)
        
        # 合并所有DataFrame
        log_message("合并数据文件...")
        combined_df = pd.concat(dfs, ignore_index=True)
        
        # 最终清理
        log_message("执行最终数据清理...")
        combined_df = clean_data(combined_df, "合并后的数据")
        
        # 保存合并后的数据
        output_path = DATA_PROCESSED_DIR / OUTPUT_FILENAME
        combined_df.to_csv(output_path, index=False)
        log_message(f"数据已成功保存到 {output_path}")
        
        # 保存处理后的统计信息
        stats = {
            '原始文件数': len(all_files),
            '成功加载文件数': len(dfs),
            '总行数': len(combined_df),
            '总列数': len(combined_df.columns),
            '数值列': len(combined_df.select_dtypes(include=[np.number]).columns),
            '分类列': len(combined_df.select_dtypes(include=['object']).columns),
            '处理时间': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        
        stats_path = DATA_PROCESSED_DIR / 'processing_stats.txt'
        with open(stats_path, 'w') as f:
            for key, value in stats.items():
                f.write(f"{key}: {value}\n")
        
        log_message("数据处理完成！")
        return True
        
    except Exception as e:
        log_message(f"处理过程中发生严重错误: {str(e)}")
        return False

if __name__ == "__main__":
    success = process_data()
    if not success:
        exit(1)