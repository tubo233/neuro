import pandas as pd
import numpy as np
import mne
import matplotlib.pyplot as plt
import scienceplots
from pathlib import Path
from scipy import stats
from Statis import StatisticalComparison
from Correlation import CorrelationAnalyzer

class BrainTopographyPlotter:
    def __init__(self):
        self.features = [
            'delta_alpha_mi','delta_alpha_mvl',
            'delta_beta_mi','delta_beta_mvl','delta_gamma_mi',
            'delta_gamma_mvl','theta_alpha_mi','theta_alpha_mvl',
            'theta_beta_mi','theta_beta_mvl','theta_gamma_mi',
            'theta_gamma_mvl','alpha_beta_mi','alpha_beta_mvl',
            'alpha_gamma_mi','alpha_gamma_mvl'
        ]
        
    def load_and_process_data(self, filepath):
        """加载并预处理数据"""
        df = pd.read_csv(filepath, sep='\t')
        # 获取所有通道的交集
        self.channels = sorted(df['ch_name'].unique())
        return df
        
    def calculate_statistics_and_correlation(self, df, state):
        """计算统计差异和相关性"""
        # 筛选特定状态的数据
        state_data = df[df['state'] == state]
        
        results = []
        for channel in self.channels:
            for feature in self.features:
                # 获取当前通道的数据
                channel_data = state_data[state_data['ch_name'] == channel].copy()
                
                # 使用IQR方法检测异常值
                Q1 = channel_data[feature].quantile(0.25)
                Q3 = channel_data[feature].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 30 * IQR
                upper_bound = Q3 + 30 * IQR
                is_normal = (channel_data[feature] >= lower_bound) & (channel_data[feature] <= upper_bound)
                
                # 移除异常值
                channel_data = channel_data[is_normal]
                
                # 获取POD组和noPOD组的数据
                pod_data = channel_data[channel_data['delirium'] == 1][feature]
                nopod_data = channel_data[channel_data['delirium'] == 0][feature]
                
                # 获取年龄数据并处理缺失值
                feature_data = channel_data[feature].values
                age_data = channel_data['age'].values
                
                # 处理年龄数据中的缺失值
                valid_indices = ~pd.isna(age_data)  # 获取非缺失值的索引
                filtered_age_data = age_data[valid_indices]  # 过滤年龄数据
                filtered_feature_data = feature_data[valid_indices]  # 相应过滤特征数据
                
                if len(pod_data) > 0 and len(nopod_data) > 0:
                    # 计算显著性差异
                    comparison = StatisticalComparison(pod_data, nopod_data)
                    p_value = comparison.perform_statistical_test()
                    # 计算效应量（组间差异）
                    difference = np.mean(pod_data) - np.mean(nopod_data)
                    
                    # 计算与年龄的相关性（使用过滤后的数据）
                    correlation = np.nan  # 默认设置为NaN
                    if len(filtered_age_data) > 0:  # 确保还有数据可以计算相关性
                        analyzer = CorrelationAnalyzer(filtered_feature_data, filtered_age_data)
                        correlation_result = analyzer.calculate_correlation()
                        correlation = correlation_result['correlation']
                    
                    results.append({
                        'group': f'POD_vs_noPOD_{state}',
                        'channel': channel,
                        'feature': feature,
                        'p_value': p_value,
                        'difference': difference,
                        'correlation': correlation
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
            fig, axes = plt.subplots(4, 4, figsize=(20, 20))
            plt.subplots_adjust(left=0.05, right=0.85, bottom=0.05,
                              top=0.92, wspace=0.2, hspace=0.2)
            fig.suptitle(title, fontsize=16)
            
            axes_flat = axes.flatten()
            
            for idx, feature in enumerate(self.features):
                if idx < 16:  # 只处理特征
                    ax = axes_flat[idx]
                    feature_data = data[data['feature'] == feature]
                    feature_values = []
                    
                    for ch in self.channels:
                        ch_value = feature_data[feature_data['channel'] == ch][measure_type].values
                        feature_values.append(ch_value[0] if len(ch_value) > 0 else 0)
                    
                    feature_values = np.array(feature_values)
                    
                    # 为相关性分析设置固定的值范围
                    kwargs = {
                        'names': self.channels,
                        'cmap': 'RdBu_r',
                        'outlines': 'head',
                        'sensors': True,
                        'contours': 6,
                        'size': 2,
                        'sphere': 0.11
                    }
                    if measure_type == 'correlation':
                        kwargs['vlim'] = (-1,1)
                        
                    im = mne.viz.plot_topomap(
                        feature_values, info, axes=ax, show=False,
                        **kwargs
                    )
                    ax.set_title(feature, pad=2, fontdict={'size': 15})
            
            # # 移除多余的子图
            # for idx in range(14, 15):
            #     fig.delaxes(axes_flat[idx])
            
            # 添加颜色条
            cax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
            cbar = plt.colorbar(im[0], cax=cax)
            if measure_type == 'p_value':
                cbar.set_label('P-value', fontsize=12)
            elif measure_type == 'difference':
                cbar.set_label('Group-Difference', fontsize=12)
            else:
                cbar.set_label('Correlation with Age', fontsize=12)
            
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            plt.close()
            
        except Exception as e:
            print(f"绘图错误: {str(e)}")

    def run_analysis(self):
        """运行完整分析流程"""
        # 创建保存目录
        Path("./save_jpg/Map/PAC/").mkdir(parents=True, exist_ok=True)
        Path("./save_csv/Map/PAC/").mkdir(parents=True, exist_ok=True)
        
        # 读取数据
        df = self.load_and_process_data("save_csv/data/process_filename_pac_median.tsv")
        
        all_results = []
        # 对每个状态进行分析
        for state in ['open-eyes', 'close-eyes']:
            # 计算统计结果和相关性
            results = self.calculate_statistics_and_correlation(df, state)
            all_results.append(results)
            
            # 绘制p值地形图
            self.plot_topomap(
                results,
                results['p_value'].values,
                f'P-value Topographic Map - POD vs noPOD ({state})',
                f"./save_jpg/Map/PAC/POD_vs_noPOD_{state}_pvalue_map_pac.jpg",
                'p_value'
            )
            
            # 绘制差异值地形图
            self.plot_topomap(
                results,
                results['difference'].values,
                f'Group Difference Topographic Map - POD vs noPOD ({state})',
                f"./save_jpg/Map/PAC/POD_vs_noPOD_{state}_difference_map_pac.jpg",
                'difference'
            )
            
            # 绘制相关性地形图
            self.plot_topomap(
                results,
                results['correlation'].values,
                f'Age Correlation Topographic Map ({state})',
                f"./save_jpg/Map/PAC/{state}_age_correlation_map_pac.jpg",
                'correlation'
            )
        
        # 合并所有结果并保存到CSV
        all_results_df = pd.concat(all_results, ignore_index=True)
        all_results_df.to_csv("./save_csv/Map/PAC/features_statistics_results.csv", index=False)

if __name__ == "__main__":
    plotter = BrainTopographyPlotter()
    plotter.run_analysis()