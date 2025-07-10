import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scienceplots
from Statis import StatisticalComparison
import glob

plt.style.use(['science', 'no-latex', 'nature'])

class PSDAnalysis:
    def __init__(self, channel):
        self.channel = channel
        self.data_path = "data/statistics-cerebral_raw_filtered_ref_Fpz/time-frequency"
        self.save_psd_path = "./save_jpg/PSD/"
        self.save_ps_path = "./save_jpg/PS/"
        self.save_ratio_path = "./save_jpg/Ratio/"
        
        # 确保保存路径存在
        for path in [self.save_psd_path, self.save_ps_path, self.save_ratio_path]:
            os.makedirs(path, exist_ok=True)

        # 定义频段范围
        self.freq_bands = {
            'δ': (0.5, 4),
            'θ': (4, 8),
            'α': (8, 13),
            'β': (13, 30),
            'γ': (30, 47)
        }

    def read_and_aggregate_data(self, file_pattern):
        """读取并聚合数据"""
        files = glob.glob(os.path.join(self.data_path, file_pattern))
        if not files:
            return None, None

        all_data = []
        freqs = None
        
        for file in files:
            try:
                df = pd.read_csv(file, sep='\t')
                freq_cols = [col for col in df.columns if 'Hz' in col and float(col.split()[0]) <= 47]
                data = df[freq_cols].median().values
                
                # 检查这条数据是否包含无效值
                if not np.any(pd.isna(data)) and not np.any(np.isinf(data)):
                    all_data.append(data)
                    if freqs is None:
                        freqs = np.array([float(col.split()[0]) for col in freq_cols])
                else:
                    print(f"警告: 文件 {file} 无有效值，已跳过此数据")
                    
            except Exception as e:
                print(f"Error reading file {file}: {str(e)}")
                continue

        if all_data:
            return np.array(all_data), freqs
        return None, None

    def process_groups(self):
        """处理所有组的数据"""
        # 定义组别
        groups = {
            'HC': f"processed_HC*_{self.channel}_psd.tsv",
            'PPD_T0': f"processed_PPD*T0*_{self.channel}_psd.tsv",
            'PPD_T3': f"processed_PPD*T3*_{self.channel}_psd.tsv",
            'PPD_T4': f"processed_PPD*T4*_{self.channel}_psd.tsv"
        }

        group_data = {}
        freqs = None
        
        for group_name, pattern in groups.items():
            data, current_freqs = self.read_and_aggregate_data(pattern)
            if data is not None:
                # 确保数据类型为float
                data = data.astype(float)
                
                # 检查数据是否全为0
                if np.all(data == 0):
                    print(f"警告: 通道 {self.channel} 在组 {group_name} 中没有有效值")
                    return None, None
                
                group_data[group_name] = data
                freqs = current_freqs
            else:
                print(f"警告: 通道 {self.channel} 在组 {group_name} 中没有找到有效数据")
                return None, None

        # 检查是否所有组的数据都已获取
        if len(group_data) != len(groups):
            print(f"警告: 通道 {self.channel} 缺少某些组的数据")
            return None, None

        return group_data, freqs



    def plot_spectra(self, group_data, freqs, plot_type='psd'):
        """绘制频谱图"""
        if group_data is None or len(group_data) == 0 or freqs is None or len(freqs) == 0:
            print("No data to plot")
            return

        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        comparisons = [('HC', 'PPD_T0'), ('HC', 'PPD_T3'), ('HC', 'PPD_T4')]
        titles = ['HC vs PPD_T0', 'HC vs PPD_T3', 'HC vs PPD_T4']
        
        mask = freqs <= 47
        freqs_masked = freqs[mask]

        # 计算y轴范围
        y_min = float('inf')
        y_max = float('-inf')
        for group_name, data in group_data.items():
            if plot_type == 'db':
                data = 10 * np.log10(data)
            elif plot_type == 'ratio':
                data = data / np.sum(data, axis=1, keepdims=True) * 100
            
            data = data[:, mask]
            y_min = min(y_min, np.percentile(data, 0.05))
            y_max = max(y_max, np.percentile(data, 99.5))

        y_range = y_max - y_min
        y_max += y_range * 0.1  # 减小顶部留白空间

        colors = ['lightblue', 'lightgreen', 'lightyellow', 'lightpink', 'lightgray']

        for ax, (group1, group2), title in zip(axes, comparisons, titles):
            ax.set_ylim(y_min, y_max)

            # 设置频段背景
            for (band, (start, end)), color in zip(self.freq_bands.items(), colors):
                ax.axvspan(start, end, color=color, alpha=0.3)
                y_text = y_min + (y_max - y_min) * 0.9  # 将频段标签移到上方
                ax.text((start + end) / 2, y_text, band,
                    horizontalalignment='center', fontsize=12)

            # 存储每个组的均值数据用于显著性标记
            group_means = {}
            
            # 处理和绘制数据
            for group_name, label, color in [(group1, 'HC', 'blue'),
                                        (group2, group2.replace('PPD_', ''), 'red')]:  # 简化PPD标签
                data = group_data[group_name]
                # 先应用频率掩码
                data = data[:, mask]
                
                # 应用数据变换
                if plot_type == 'db':
                    data = 10 * np.log10(data)
                elif plot_type == 'ratio':
                    data = data / np.sum(data, axis=1, keepdims=True) * 100
                    
                mean = np.mean(data, axis=0)
                q1 = np.percentile(data, 5, axis=0)
                q3 = np.percentile(data, 95, axis=0)
                group_means[group_name] = mean

                line = ax.plot(freqs_masked, mean, label=label, color=color)
                ax.fill_between(freqs_masked, q1, q3, color=color, alpha=0.2)

            # 显著性检验和标记
            has_significant = False
            y_offset = (y_max - y_min) * 0.001  # 固定偏移量
            
            for f_idx, freq in enumerate(freqs_masked):
                # 获取原始数据并应用相同的数据变换
                g1_data = group_data[group1][:, mask][:, f_idx]
                g2_data = group_data[group2][:, mask][:, f_idx]
                
                # 对数据进行相同的变换
                if plot_type == 'db':
                    g1_values = 10 * np.log10(g1_data)
                    g2_values = 10 * np.log10(g2_data)
                elif plot_type == 'ratio':
                    g1_sum = np.sum(group_data[group1][:, mask], axis=1)
                    g2_sum = np.sum(group_data[group2][:, mask], axis=1)
                    g1_values = g1_data / g1_sum * 100
                    g2_values = g2_data / g2_sum * 100
                else:  # psd
                    g1_values = g1_data
                    g2_values = g2_data
                
                comparison = StatisticalComparison(g1_values, g2_values)
                p_value = comparison.perform_statistical_test()

                if p_value < 0.05:
                    has_significant = True
                    max_val = max(group_means[group1][f_idx], group_means[group2][f_idx])
                    # 在数据线正上方标记显著性
                    ax.plot(freq, max_val + y_offset, 'k*', markersize=4)

            # 添加显著性图例
            if has_significant:
                # 在图例中添加显著性说明
                ax.plot([], [], 'k*', markersize=4, label='p < 0.05')

            ax.set_title(title)
            ax.set_xlabel('Frequency (Hz)')
            if plot_type == 'psd':
                ax.set_ylabel('Power Spectral Density (µV²/Hz)')
            elif plot_type == 'db':
                ax.set_ylabel('Power (dB)')
            else:
                ax.set_ylabel('Normalized Power (%)')

            ax.legend(loc='upper right')
            ax.grid(True)

        plt.tight_layout()

        # 保存图片
        filename = f"{self.channel}"
        if plot_type == 'psd':
            save_path = f"{self.save_psd_path}{filename}_PSD_ref_Fpz.jpg"
        elif plot_type == 'db':
            save_path = f"{self.save_ps_path}{filename}_PS_ref_Fpz.jpg"
        else:
            save_path = f"{self.save_ratio_path}{filename}_Ratio_ref_Fpz.jpg"

        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

def main():
    #设置要处理的通道
    # channels = ['F3','AF3','FC5','FC1','F9','T9','P9','CP5','P3','Cz','C3','AF4',
    #             	'Fz','Pz','T10','FC2','C4','F10','FC6','F4','PO3','CP2','CP1',
    #                     'O1','O2','P10','CP6','P4','Oz','PO4']
    channels =['AF3','AF4','Fp1','Fp2','Fz','Cz','F7','F8']
    
    for channel in channels:
        # 创建分析实例
        analysis = PSDAnalysis(channel)
        
        # 处理数据
        group_data, freqs = analysis.process_groups()
        
        # 绘制不同类型的图
        analysis.plot_spectra(group_data, freqs, plot_type='psd')
        analysis.plot_spectra(group_data, freqs, plot_type='db')
        analysis.plot_spectra(group_data, freqs, plot_type='ratio')

if __name__ == "__main__":
    main()