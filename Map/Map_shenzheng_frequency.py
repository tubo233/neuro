import os
import pandas as pd
import numpy as np
import mne
from Statis import StatisticalComparison
import matplotlib.pyplot as plt
import scienceplots
import seaborn as sns
from pathlib import Path
import re

class BrainMapAnalysis:
    def __init__(self):
        self.features_of_interest = [
            'delta_proportion', 'theta_proportion','alpha_proportion','beta_proportion','gamma_proportion',
            'delta_power',      'theta_power',     'alpha_power',     'beta_power',     'gamma_power',
            'state_entropy',    'spectral_entropy','sef_95',          'total_power'
        ]
        
        self.groups = {
            'HC': [],
            'PPD_T0': [],
            'PPD_T3': [],
            'PPD_T4': []
        }
        # 初始化通道集合
        self.channels = set()

    def get_channels_from_file(self, filepath):
        """从文件中获取通道信息"""
        df = pd.read_csv(filepath, sep='\t')
        return sorted(df['ch_name'].unique())

    def parse_filename(self, filename):
        """解析文件名获取组别信息"""
        pattern = r'processed_((?:HC|PPD)\d+(?:T[034])?)-.*\.tsv'
        match = re.match(pattern, filename)
        if not match:
            return None
        
        subject_id = match.group(1)
        if subject_id.startswith('HC'):
            return 'HC', subject_id
        elif 'T0' in subject_id:
            return 'PPD_T0', subject_id
        elif 'T3' in subject_id:
            return 'PPD_T3', subject_id
        elif 'T4' in subject_id:
            return 'PPD_T4', subject_id
        return None

    def read_and_process_data(self, base_path):
        """读取和处理数据"""
        group_data = {group: {} for group in self.groups.keys()}
        
        # 首先收集所有文件中的通道信息，并计算交集
        all_channels_list = []
        for filename in os.listdir(base_path):
            if not filename.endswith('_frequency_distribution.tsv'):
                continue
                
            filepath = os.path.join(base_path, filename)
            file_channels = set(self.get_channels_from_file(filepath))
            all_channels_list.append(file_channels)
            
            # 先计算所有通道的并集，用于后续计算每个文件缺失的通道
            if len(all_channels_list) == 1:
                all_channels_union = file_channels
            else:
                all_channels_union = all_channels_union.union(file_channels)
                
            print(f"处理文件 {filename} 的通道信息，当前文件通道数: {len(file_channels)}")
            # 显示缺失的通道
            if len(all_channels_list) > 1:  # 从第二个文件开始显示缺失通道
                missing_channels = all_channels_union - file_channels
                if missing_channels:
                    print(f"文件 {filename} 缺失的通道: {sorted(list(missing_channels))}")
        
        # 计算所有文件的通道交集
        if all_channels_list:
            all_channels = set.intersection(*all_channels_list)
            print(f"\n计算交集前各文件通道数: {[len(channels) for channels in all_channels_list]}")
            print(f"所有文件中出现的通道总数（并集）: {len(all_channels_union)}")
            print(f"所有文件共有的通道数（交集）: {len(all_channels)}")
            
            # 显示哪些通道不是所有文件都有的
            uncommon_channels = all_channels_union - all_channels
            if uncommon_channels:
                print(f"不是所有文件都包含的通道: {sorted(list(uncommon_channels))}\n")
        else:
            all_channels = set()

        # 将通道交集转换为排序列表
        self.channels = sorted(list(all_channels))
        print(f"所有文件共有的通道数: {len(self.channels)}")
        print(f"共有通道列表: {self.channels}")
        
        # 处理每个符合条件的文件数据
        for filename in os.listdir(base_path):
            if not filename.endswith('_frequency_distribution.tsv'):
                continue
                
            group_info = self.parse_filename(filename)
            if not group_info:
                continue
                
            group, subject_id = group_info
            filepath = os.path.join(base_path, filename)
            
            # 读取数据文件
            df = pd.read_csv(filepath, sep='\t')
            print(f"处理文件: {filename}")
            
            # 获取此文件的可用通道
            available_channels = sorted(df['ch_name'].unique())
            
            # 为每个通道处理数据
            for channel in available_channels:
                channel_data = df[df['ch_name'] == channel]
                if len(channel_data) == 0:
                    continue
                
                # 计算每个特征的中位数
                for feature in self.features_of_interest:
                    key = (channel, feature)
                    if key not in group_data[group]:
                        group_data[group][key] = {}
                    group_data[group][key][subject_id] = channel_data[feature].median()
        
        return group_data

    def calculate_statistics(self, group_data):
        """计算统计数据"""
        results = []
        comparisons = [
            ('HC', 'PPD_T0'),
            ('HC', 'PPD_T3'),
            ('HC', 'PPD_T4')
        ]
        
        for group1, group2 in comparisons:
            for channel in self.channels:
                for feature in self.features_of_interest:
                    key = (channel, feature)
                    
                    if key not in group_data[group1] or key not in group_data[group2]:
                        continue
                    
                    group1_values = np.array(list(group_data[group1][key].values()))
                    group2_values = np.array(list(group_data[group2][key].values()))
                    
                    if len(group1_values) == 0 or len(group2_values) == 0:
                        continue
                    
                    # 清理异常值
                    group1_values = self.remove_outliers(group1_values, f"{channel}-{feature}-{group1}")
                    group2_values = self.remove_outliers(group2_values, f"{channel}-{feature}-{group2}")
                    
                    # 标准化数据
                    if len(group1_values) > 1 and len(group2_values) > 1:
                        group1_norm = group1_values
                        group2_norm = group2_values

                        # 计算显著性差异
                        comparison = StatisticalComparison(group1_values, group2_values)
                        p_value = comparison.perform_statistical_test()
                        
                        # 计算组间差异
                        difference = np.mean(group2_norm) - np.mean(group1_norm)
                        
                        results.append({
                            'group': f'{group1}vs{group2}',
                            'channel': channel,
                            'feature': feature,
                            'p_value': p_value,
                            'difference': difference
                        })
        
        return pd.DataFrame(results)

    def plot_topomap(self, data, values, title, filename, measure_type):
        """绘制地形图"""
        plt.style.use(['science', 'nature'])
        
        try:
            # 创建montage和info
            montage = mne.channels.make_standard_montage('standard_1020')
            info = mne.create_info(ch_names=self.channels, sfreq=300, ch_types='eeg')
            info.set_montage(montage)

            # 创建图表布局
            fig, axes = plt.subplots(3, 5, figsize=(15, 9))
            plt.subplots_adjust(
                left=0.05, right=0.85, bottom=0.05,
                top=0.92, wspace=0.2, hspace=0.2
            )
            fig.suptitle(f'{title}', fontsize=16)
            
            axes_flat = axes.flatten()
            
            for idx, feature in enumerate(self.features_of_interest):
                if idx < 14:
                    ax = axes_flat[idx]
                    # 获取特定特征的数据，并确保与通道顺序匹配
                    feature_data = data[data['feature'] == feature]
                    feature_values = []
                    for ch in self.channels:  # 使用self.channels的顺序
                        ch_value = feature_data[feature_data['channel'] == ch]['difference' if measure_type == 'difference' else 'p_value'].values
                        if len(ch_value) > 0:
                            feature_values.append(ch_value[0])
                        else:
                            print(f"Warning: Missing value for channel {ch} in feature {feature}")
                            feature_values.append(0)  # 或者使用其他默认值
                    
                    feature_values = np.array(feature_values)
                    im = mne.viz.plot_topomap(
                        feature_values, info, axes=ax, show=False,
                        names=self.channels, cmap='RdBu_r',
                        outlines='head', sensors=True,
                        contours=6, size=2,
                        sphere = 0.11  # 控制图缩放
                    )
                    ax.set_title(feature, pad=2,fontdict={'size':15})
            
            # 移除多余的子图
            for idx in range(14, 15):
                fig.delaxes(axes_flat[idx])
            
            # 添加颜色条
            cax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
            cbar = plt.colorbar(im[0], cax=cax)
            
            if measure_type == 'p_value':
                cbar.set_label('P-value', fontsize=12)
            else:
                cbar.set_label('Group-Difference', fontsize=12)
            
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            plt.close()
            
        except Exception as e:
            print(f"绘图错误: {str(e)}")
            print(f"当前通道: {self.channels}")

    def remove_outliers(self, data, feature_name):
        """使用1.5倍IQR方法去除异常值"""
        if len(data) < 4:  # 数据太少不处理
            return data
            
        q1 = np.percentile(data, 25)
        q3 = np.percentile(data, 75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        
        cleaned_data = data[(data >= lower_bound) & (data <= upper_bound)]
        if len(cleaned_data) != len(data):
            print(f"在特征 {feature_name} 中检测到 {len(data) - len(cleaned_data)} 个异常值")
        return cleaned_data

    def run_analysis(self):
        """运行完整分析流程"""
        # 创建必要的目录
        Path("./save_jpg/Map/Frequency/").mkdir(parents=True, exist_ok=True)
        Path("./save_csv/Map/Frequency/").mkdir(parents=True, exist_ok=True)
        
        # 读取数据
        base_path = "data/statistics-cerebral_raw_filtered_ref_average/features"
        group_data = self.read_and_process_data(base_path)
        
        # 计算统计结果
        results_df = self.calculate_statistics(group_data)
        
        # 保存统计结果
        results_df.to_csv("./save_csv/Map/Frequency/features_statistics_results_frequency_ref_average.csv", index=False)
        
        # 为每个组比较绘制地形图
        for group_comparison in results_df['group'].unique():
            comparison_data = results_df[results_df['group'] == group_comparison]
            
            # 绘制p值地形图
            self.plot_topomap(
                comparison_data,
                comparison_data['p_value'].values,
                f'P-value Topographic Map - {group_comparison}',
                f"./save_jpg/Map/Frequency/{group_comparison}-raw_features_Map_pvalue_frequency_ref_average.jpg",
                'p_value'
            )
            
            # 绘制差异值地形图
            self.plot_topomap(
                comparison_data,
                comparison_data['difference'].values,
                f'Group Difference Topographic Map - {group_comparison}',
                f"./save_jpg/Map/Frequency/{group_comparison}-raw_features_Map_differences_frequency_ref_average.jpg",
                'difference'
            )

if __name__ == "__main__":
    analysis = BrainMapAnalysis()
    analysis.run_analysis()