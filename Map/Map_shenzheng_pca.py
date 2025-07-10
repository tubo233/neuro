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
            'delta_alpha_mi','delta_alpha_mvl',
            'delta_beta_mi','delta_beta_mvl','delta_gamma_mi',
            'delta_gamma_mvl','theta_alpha_mi','theta_alpha_mvl',
            'theta_beta_mi','theta_beta_mvl','theta_gamma_mi',
            'theta_gamma_mvl','alpha_beta_mi','alpha_beta_mvl',
            'alpha_gamma_mi','alpha_gamma_mvl'
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
        
        # 首先收集符合条件的文件中的通道信息
        for filename in os.listdir(base_path):
            if not filename.endswith('_pac.tsv'):
                continue
                
            filepath = os.path.join(base_path, filename)
            file_channels = self.get_channels_from_file(filepath)
            self.channels.update(file_channels)
            print(f"读取通道信息: {filename}")
        
        # 将通道集合转换为排序列表
        self.channels = sorted(list(self.channels))
        print(f"发现的通道: {len(self.channels)}")
        print(f"通道列表: {self.channels}")
        
        # 处理每个符合条件的文件数据
        for filename in os.listdir(base_path):
            if not filename.endswith('_pac.tsv'):
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
                    
                    # 标准化数据
                    if len(group1_values) > 1 and len(group2_values) > 1:
                        # group1_norm = (group1_values - np.mean(group1_values)) / np.std(group1_values)
                        # group2_norm = (group2_values - np.mean(group2_values)) / np.std(group2_values)
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
            fig, axes = plt.subplots(4, 4, figsize=(20, 20)) #4×4=16张图
            plt.subplots_adjust(
                left=0.05, right=0.85, bottom=0.05,
                top=0.92, wspace=0.2, hspace=0.2
            )
            fig.suptitle(f'{title}', fontsize=16)
            
            axes_flat = axes.flatten()
            
            for idx, feature in enumerate(self.features_of_interest):
                if idx < 16: #16个特征16张图
                    ax = axes_flat[idx]
                    feature_values = values[data['feature'] == feature]
                    
                    im = mne.viz.plot_topomap(
                        feature_values, info, axes=ax, show=False,
                        names=self.channels, cmap='RdBu_r',
                        outlines='head', sensors=True,
                        contours=6, size=2,
                        sphere = 0.11 #控制图缩放
                    )
                    ax.set_title(feature, pad=2, fontdict={'size':18})
            
            # # 移除多余的子图
            # for idx in range(14, 15):
            #     fig.delaxes(axes_flat[idx])
            
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

    def run_analysis(self):
        """运行完整分析流程"""
        # 创建必要的目录
        Path("./save_jpg/Map/PAC").mkdir(parents=True, exist_ok=True)
        Path("./save_csv/Map/PAC").mkdir(parents=True, exist_ok=True)
        
        # 读取数据
        base_path = "./data/statistics-cerebral_raw_filtered_ref_average_autoreject/features/processed_pac_del/"
        group_data = self.read_and_process_data(base_path)
        
        # 计算统计结果
        results_df = self.calculate_statistics(group_data)
        
        # 保存统计结果
        results_df.to_csv("./save_csv/Map/PAC/features_statistics_results_pac.csv", index=False)
        
        # 为每个组比较绘制地形图
        for group_comparison in results_df['group'].unique():
            comparison_data = results_df[results_df['group'] == group_comparison]
            
            # 绘制p值地形图
            self.plot_topomap(
                comparison_data,
                comparison_data['p_value'].values,
                f'P-value Topographic Map - {group_comparison}',
                f"./save_jpg/Map/PAC/{group_comparison}-raw_features_Map_pvalue_pac.jpg",
                'p_value'
            )
            
            # 绘制差异值地形图
            self.plot_topomap(
                comparison_data,
                comparison_data['difference'].values,
                f'Group Difference Topographic Map - {group_comparison}',
                f"./save_jpg/Map/PAC/{group_comparison}-raw_features_Map_differences_pac.jpg",
                'difference'
            )

if __name__ == "__main__":
    analysis = BrainMapAnalysis()
    analysis.run_analysis()