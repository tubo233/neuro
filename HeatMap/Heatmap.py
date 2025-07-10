import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scienceplots
import os
from Correlation import CorrelationAnalyzer

def calculate_correlation_matrix(df, feature, channels):
    """计算相关性矩阵"""
    n = len(channels)
    corr_matrix = np.zeros((n, n))
    
    for i in range(n):
        for j in range(n):
            if i != j:
                # 获取两个通道的数据
                data1 = df[df['ch_name'] == channels[i]][feature].values
                data2 = df[df['ch_name'] == channels[j]][feature].values
                
                # 确保数据长度匹配
                min_len = min(len(data1), len(data2))
                if min_len > 0:
                    # 创建配对数据的DataFrame
                    paired_data = pd.DataFrame({
                        'channel1': data1[:min_len],
                        'channel2': data2[:min_len]
                    })
                    
                    # 同时对配对数据进行异常值处理
                    Q1_1 = paired_data['channel1'].quantile(0.25)
                    Q3_1 = paired_data['channel1'].quantile(0.75)
                    IQR_1 = Q3_1 - Q1_1
                    Q1_2 = paired_data['channel2'].quantile(0.25)
                    Q3_2 = paired_data['channel2'].quantile(0.75)
                    IQR_2 = Q3_2 - Q1_2
                    
                    # 同时删除任一通道中的异常值对应的行
                    mask = ~((paired_data['channel1'] < (Q1_1 - 30 * IQR_1)) | 
                           (paired_data['channel1'] > (Q3_1 + 30 * IQR_1)) |
                           (paired_data['channel2'] < (Q1_2 - 30 * IQR_2)) |
                           (paired_data['channel2'] > (Q3_2 + 30 * IQR_2)))
                    
                    clean_data = paired_data[mask]
                    
                    if len(clean_data) > 0:  # 确保清洗后还有数据
                        analyzer = CorrelationAnalyzer(
                            clean_data['channel1'].values,
                            clean_data['channel2'].values
                        )
                        result = analyzer.calculate_correlation()
                        corr_matrix[i, j] = result['correlation']
                    else:
                        corr_matrix[i, j] = np.nan
                else:
                    corr_matrix[i, j] = np.nan
            else:
                corr_matrix[i, j] = 1.0  # 对角线设为1
    
    return corr_matrix

def plot_heatmaps(data, feature, save_path):
    """绘制热力图"""
    plt.style.use(['science', 'nature'])
    
    # 获取唯一的通道名称
    channels = sorted(data['ch_name'].unique())
    
    # 创建图形
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle(f'Channel Connectivity Heatmap - {feature}', fontsize=16)
    
    # 定义条件组合
    conditions = [
        ('POD', 'open-eyes', axes[0,0]),
        ('POD', 'close-eyes', axes[0,1]),
        ('noPOD', 'open-eyes', axes[1,0]),
        ('noPOD', 'close-eyes', axes[1,1])
    ]
    
    results_list = []
    
    for group, state, ax in conditions:
        # 筛选数据
        mask = (data['delirium'] == (1 if group == 'POD' else 0)) & (data['state'] == state)
        subset = data[mask]
        
        # 计算相关性矩阵
        corr_matrix = calculate_correlation_matrix(subset, feature, channels)
        
        # 绘制热力图
        sns.heatmap(corr_matrix, ax=ax, xticklabels=channels, yticklabels=channels,
                    cmap='RdBu_r', vmin=-1, vmax=1, center=0)
        ax.set_title(f'{group} - {state}')
        
        # 收集结果用于CSV
        for i in range(len(channels)):
            for j in range(len(channels)):
                results_list.append({
                    'group': group,
                    'channel_target': f'{channels[i]}-{channels[j]}',
                    'feature': feature,
                    'state': state,
                    'correlation': corr_matrix[i,j]
                })
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, f'heatmap_{feature}.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    return results_list

def main():
    # 创建保存目录
    os.makedirs('./save_jpg/HeatMap', exist_ok=True)
    os.makedirs('./save_csv/HeatMap', exist_ok=True)
    
    # 读取数据
    data = pd.read_csv('./save_csv/data/process_filename_frequency_distribution_median.tsv', sep='\t')
    
    # 将delirium转换为数值型
    data['delirium'] = pd.to_numeric(data['delirium'])
    
    # 定义特征列表
    features = ['delta_proportion', 'theta_proportion', 'alpha_proportion', 
                'beta_proportion', 'gamma_proportion', 'delta_power', 
                'theta_power', 'alpha_power', 'beta_power', 'gamma_power',
                'state_entropy', 'spectral_entropy', 'sef_95', 'total_power']
    

    
    # 存储所有结果
    all_results = []
    
    # 为每个特征创建热力图
    for feature in features:
        print(f"Processing feature: {feature}")
        results = plot_heatmaps(data, feature, './save_jpg/HeatMap')
        all_results.extend(results)
    
    # 保存结果到CSV
    results_df = pd.DataFrame(all_results)
    results_df.to_csv('./save_csv/HeatMap/features_statistics_HeatMap_results.csv', 
                      index=False)

if __name__ == "__main__":
    main()