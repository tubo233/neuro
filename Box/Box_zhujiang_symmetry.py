import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from Statis import StatisticalComparison
import scienceplots

def read_data(file_path):
    """读取数据文件并按状态分组"""
    df = pd.read_csv(file_path, sep='\t')
    # 分别获取开眼和闭眼的数据
    open_eyes = df[df['state'] == 'open-eyes']
    close_eyes = df[df['state'] == 'close-eyes']
    return open_eyes, close_eyes

def split_groups(data):
    """将数据分为POD和noPOD两组"""
    pod = data[data['delirium'] == 1]
    nopod = data[data['delirium'] == 0]
    return pod, nopod

def remove_outliers(data, feature):
    """使用IQR方法去除异常值"""
    if len(data) < 4:
        return data
        
    q1 = np.percentile(data, 25)
    q3 = np.percentile(data, 75)
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    
    cleaned_data = [x for x in data if lower_bound <= x <= upper_bound]
    if len(cleaned_data) != len(data):
        print(f"在特征 {feature} 中检测到 {len(data) - len(cleaned_data)} 个异常值")
    return cleaned_data

def create_box_plots(pod_data, nopod_data, state, output_path, csv_data):
    """创建箱型图并保存"""
    plt.style.use(['science', 'nature'])
    
    # 设置特征名称
    feature_names = ['total_ratio','total_dB','delta_ratio','delta_dB',
                    'theta_ratio','theta_dB','alpha_ratio','alpha_dB',
                    'beta_ratio','beta_dB','gamma_ratio','gamma_dB']
    
    # 获取所有通道对
    all_channels = sorted(pod_data['target_channels'].unique())
    
    # 设置图形大小和布局
    fig = plt.figure(figsize=(20, 2.5*len(all_channels)))
    gs = fig.add_gridspec(len(all_channels), 1, hspace=0)
    fig.suptitle(f'noPOD vs POD ({state}) Statistical Comparison', 
                fontsize=16, y=1.02)
    
    # 创建子图
    axes = []
    for i in range(len(all_channels)):
        if i == 0:
            ax = fig.add_subplot(gs[i])
        else:
            ax = fig.add_subplot(gs[i], sharex=axes[0])
        axes.append(ax)
    
    # 设置每个箱的位置
    num_features = len(feature_names)
    positions = np.arange(num_features) * 3
    
    # 计算y轴范围
    y_max_all = float('-inf')
    y_min_all = float('inf')
    
    # 第一遍循环获取数据范围
    for channel in all_channels:
        for feature in feature_names:
            pod_values = pod_data[pod_data['target_channels'] == channel][feature].dropna().tolist()
            nopod_values = nopod_data[nopod_data['target_channels'] == channel][feature].dropna().tolist()
            
            if pod_values and nopod_values:
                pod_values = remove_outliers(pod_values, f"{channel}-{feature}-POD")
                nopod_values = remove_outliers(nopod_values, f"{channel}-{feature}-noPOD")
                
                if pod_values and nopod_values:
                    y_max_all = max(y_max_all, max(max(pod_values), max(nopod_values)))
                    y_min_all = min(y_min_all, min(min(pod_values), min(nopod_values)))
    
    y_range = y_max_all - y_min_all
    p_value_height = y_max_all - y_range * 0.25
    
    # 第二遍循环绘制图形
    for idx, (channel, ax) in enumerate(zip(all_channels, axes)):
        for j, feature in enumerate(feature_names):
            try:
                pod_values = pod_data[pod_data['target_channels'] == channel][feature].dropna().tolist()
                nopod_values = nopod_data[nopod_data['target_channels'] == channel][feature].dropna().tolist()
                
                if pod_values and nopod_values:
                    pod_values = remove_outliers(pod_values, f"{channel}-{feature}-POD")
                    nopod_values = remove_outliers(nopod_values, f"{channel}-{feature}-noPOD")
                    
                    if pod_values and nopod_values:
                        box_data = [nopod_values, pod_values]
                        bp = ax.boxplot(box_data,
                                      positions=[positions[j]-0.3, positions[j]+0.3],
                                      widths=0.4,
                                      patch_artist=True)
                        
                        # 设置颜色
                        colors = ['#2ecc71', '#e74c3c']
                        for patch, color in zip(bp['boxes'], colors):
                            patch.set_facecolor(color)
                            patch.set_alpha(0.7)
                        
                        # 计算显著性差异
                        comparison = StatisticalComparison(nopod_values, pod_values)
                        p_value = comparison.perform_statistical_test()
                        
                        # 保存结果
                        csv_data.append({
                            'state': state,
                            'channel_target': channel,
                            'feature': feature,
                            'p_value': p_value
                        })
                        
                        # 添加p值标注
                        ax.plot([positions[j]-0.3, positions[j]+0.3],
                               [p_value_height, p_value_height], 'k-', linewidth=1)
                        text_color = 'red' if p_value < 0.05 else 'black'
                        ax.text(positions[j], p_value_height, f'p={p_value:.3f}',
                               ha='center', va='bottom', color=text_color, fontsize=10,
                               bbox=dict(facecolor='white', edgecolor='black', alpha=0.7))
            
            except Exception as e:
                print(f"处理特征 {feature} 时出错: {str(e)}")
                continue
        
        # 设置子图样式
        ax.set_ylabel(channel, fontsize=12, labelpad=10)
        ax.tick_params(axis='y', labelsize=10)
        ax.set_ylim(y_min_all - y_range*0.01, p_value_height + y_range*0.2)
        
        if idx != len(all_channels)-1:
            ax.set_xticklabels([])
        else:
            ax.set_xticks(positions)
            ax.set_xticklabels(feature_names, rotation=45, ha='right', fontsize=12)
        
        # 添加分隔线和网格
        for pos in positions[:-1]:
            ax.axvline(x=pos + 1.5, color='gray', linestyle='-', 
                      alpha=0.5, linewidth=1.5, zorder=0)
        ax.grid(True, axis='y', alpha=0.3, zorder=0)
        ax.set_xlim(positions[0]-1.5, positions[-1]+1.5)
    
    # 添加图例
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor='#2ecc71', alpha=0.7, label='noPOD'),
                      Patch(facecolor='#e74c3c', alpha=0.7, label='POD')]
    fig.legend(handles=legend_elements,
              loc='upper right',
              bbox_to_anchor=(0.98, 1.02),
              ncol=1,
              fontsize=12)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_path, f'noPOD_vs_POD_{state}_symmetry.jpg'),
                bbox_inches='tight', dpi=300)
    plt.close()

def main():
    # 设置路径
    data_path = "save_csv/data/process_filename_symmetry_median.tsv"
    output_path = "./save_jpg/Box"
    csv_path = "./save_csv/Box"
    
    # 创建输出目录
    os.makedirs(output_path, exist_ok=True)
    os.makedirs(csv_path, exist_ok=True)
    
    # 读取数据
    open_eyes_data, close_eyes_data = read_data(data_path)
    
    # 分别处理开眼和闭眼数据
    open_pod, open_nopod = split_groups(open_eyes_data)
    close_pod, close_nopod = split_groups(close_eyes_data)
    
    # 存储CSV数据
    csv_data = []
    
    # 创建图表
    create_box_plots(open_pod, open_nopod, "open-eyes", output_path, csv_data)
    create_box_plots(close_pod, close_nopod, "close-eyes", output_path, csv_data)
    
    # 保存统计结果
    csv_df = pd.DataFrame(csv_data)
    csv_df.to_csv(os.path.join(csv_path, "delirium_symmetry_statistics_results.csv"), index=False)

if __name__ == "__main__":
    main()