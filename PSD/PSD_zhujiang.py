import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scienceplots
from Statis import StatisticalComparison

plt.style.use(['science', 'no-latex', 'nature'])

class PSDAnalysis:
    def __init__(self):
        self.data_path = "save_csv/data/process_time-frequency_median.tsv"
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
    def remove_outliers_iqr(self, data):
        """使用IQR 法则去除异常值"""
        # 对每个频率点分别计算IQR
        q1 = np.percentile(data, 25, axis=0)
        q3 = np.percentile(data, 75, axis=0)
        iqr = q3 - q1
        
        # 计算上下界
        lower_bound = q1 - 30 * iqr
        upper_bound = q3 + 30 * iqr
        
        # 创建掩码
        mask = np.all((data >= lower_bound) & (data <= upper_bound), axis=1)
        
        # 返回清理后的数据
        return data[mask]

    def read_data(self, channel, eye_state):
        """读取数据"""
        try:
            # 读取TSV文件
            df = pd.read_csv(self.data_path, sep='\t')
            
            # 过滤特定通道和眼睛状态的数据
            df = df[(df['ch_name'] == channel) & (df['state'] == eye_state)]
            
            # 获取频率列（0-47Hz）
            freq_cols = [col for col in df.columns if 'Hz' in col and float(col.split()[0]) <= 47]
            freqs = np.array([float(col.split()[0]) for col in freq_cols])
            
            # 分组获取数据
            pod_data = df[df['delirium'] == 1][freq_cols].values
            nopod_data = df[df['delirium'] == 0][freq_cols].values
            
            # 应用IQR异常值去除
            pod_data = self.remove_outliers_iqr(pod_data)
            nopod_data = self.remove_outliers_iqr(nopod_data)
            
            # 检查数据有效性
            if len(pod_data) == 0 or len(nopod_data) == 0:
                print(f"警告: 通道 {channel} 在 {eye_state} 状态下缺少数据")
                return None, None
                
            # 打印异常值去除后的样本数量
            print(f"通道 {channel} - {eye_state}:")
            print(f"POD组: {len(pod_data)}个样本")
            print(f"noPOD组: {len(nopod_data)}个样本")
            
            group_data = {
                'POD': pod_data,
                'noPOD': nopod_data
            }
            
            return group_data, freqs
            
        except Exception as e:
            print(f"Error reading data: {str(e)}")
            return None, None

    def plot_spectra(self, channel, plot_type='psd'):
        """绘制频谱图"""
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        eye_states = ['open-eyes', 'close-eyes']
        
        for ax, eye_state in zip(axes, eye_states):
            group_data, freqs = self.read_data(channel, eye_state)
            
            if group_data is None or freqs is None:
                continue
                
            # 首先对所有数据进行单位转换
            for group_name in group_data:
                if plot_type == 'db':
                    group_data[group_name] = 10 * np.log10(group_data[group_name])
                elif plot_type == 'ratio':
                    group_data[group_name] = (group_data[group_name] / 
                        np.sum(group_data[group_name], axis=1, keepdims=True) * 100)
                
            # 计算y轴范围
            y_min = float('inf')
            y_max = float('-inf')
            
            for data in group_data.values():
                y_min = min(y_min, np.percentile(data, 0.05))
                y_max = max(y_max, np.percentile(data, 99.5))

            y_range = y_max - y_min
            y_max += y_range * 0.1

            colors = ['lightblue', 'lightgreen', 'lightyellow', 'lightpink', 'lightgray']
            
            # 设置频段背景
            for (band, (start, end)), color in zip(self.freq_bands.items(), colors):
                ax.axvspan(start, end, color=color, alpha=0.3)
                y_text = y_min + (y_max - y_min) * 0.9
                ax.text((start + end) / 2, y_text, band,
                    horizontalalignment='center', fontsize=12)

            # 存储每个组的均值数据
            group_means = {}
            
            # 绘制数据
            for group_name, color in [('noPOD', 'blue'), ('POD', 'red')]:
                data = group_data[group_name]
                mean = np.mean(data, axis=0)
                q1 = np.percentile(data, 5, axis=0)
                q3 = np.percentile(data, 95, axis=0)
                group_means[group_name] = mean

                ax.plot(freqs, mean, label=group_name, color=color)
                ax.fill_between(freqs, q1, q3, color=color, alpha=0.2)

            # 显著性检验和标记
            y_offset = (y_max - y_min) * 0.001
            has_significant = False
            
            # 使用转换后的数据进行统计检验
            for f_idx, freq in enumerate(freqs):
                comparison = StatisticalComparison(
                    group_data['noPOD'][:, f_idx],
                    group_data['POD'][:, f_idx]
                )
                p_value = comparison.perform_statistical_test()

                if p_value < 0.05:
                    has_significant = True
                    max_val = max(group_means['noPOD'][f_idx], group_means['POD'][f_idx])
                    ax.plot(freq, max_val + y_offset, 'k*', markersize=4)

            if has_significant:
                ax.plot([], [], 'k*', markersize=4, label='p < 0.05')

            ax.set_title(f'{eye_state}')
            ax.set_xlabel('Frequency (Hz)')
            if plot_type == 'psd':
                ax.set_ylabel('Power Spectral Density (µV²/Hz)')
            elif plot_type == 'db':
                ax.set_ylabel('Power (dB)')
            else:
                ax.set_ylabel('Normalized Power (%)')

            ax.legend(loc='upper right')
            ax.grid(True)
            ax.set_ylim(y_min, y_max)

        plt.suptitle(f'Channel: {channel}')
        plt.tight_layout()

        # 保存图片
        filename = f"{channel}"
        if plot_type == 'psd':
            save_path = f"{self.save_psd_path}{filename}_PSD.jpg"
        elif plot_type == 'db':
            save_path = f"{self.save_ps_path}{filename}_PS.jpg"
        else:
            save_path = f"{self.save_ratio_path}{filename}_Ratio.jpg"

        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

def main():
    # 设置要处理的通道
    channels = ['Fp1','Fp2','F3','F4','C3','C4','P3','P4','O1','O2','F7','F8','T3','T4','Fz','Pz']
    
    # channels = ['Fp1']
    
    # 创建分析实例
    analysis = PSDAnalysis()
    
    # 为每个通道生成三种类型的图
    for channel in channels:
        analysis.plot_spectra(channel, plot_type='psd')
        analysis.plot_spectra(channel, plot_type='db')
        analysis.plot_spectra(channel, plot_type='ratio')

if __name__ == "__main__":
    main()