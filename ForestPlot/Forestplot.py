import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import scienceplots
from Statis import StatisticalComparison
import os

class ForestPlotter:
    def __init__(self):
        self.features = ['state_entropy', 'spectral_entropy', 'sef_95', 'total_power',
                        'delta_proportion', 'delta_power', 'theta_proportion','theta_power',
                        'alpha_proportion', 'alpha_power', 'beta_proportion', 'beta_power',
                        'gamma_proportion', 'gamma_power'
                        ]
        
    def remove_outliers(self, data):
        Q1 = data.quantile(0.25)
        Q3 = data.quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 30 * IQR
        upper_bound = Q3 + 30 * IQR
        return data[(data >= lower_bound) & (data <= upper_bound)]

    def calculate_odds_ratio(self, feature_data_pod, feature_data_nopod):
        # 计算中位数作为分界点
        median = np.median(pd.concat([feature_data_pod, feature_data_nopod]))
        
        # 计算高于和低于中位数的数量
        a = sum(feature_data_pod > median)    # POD组高于中位数
        b = sum(feature_data_pod <= median)   # POD组低于中位数
        c = sum(feature_data_nopod > median)  # noPOD组高于中位数
        d = sum(feature_data_nopod <= median) # noPOD组低于中位数
        
        # 添加平滑因子0.5防止除零
        odds_ratio = ((a + 0.5) * (d + 0.5)) / ((b + 0.5) * (c + 0.5))
        
        # 计算95%置信区间
        log_or = np.log(odds_ratio)
        se = np.sqrt(1/(a+0.5) + 1/(b+0.5) + 1/(c+0.5) + 1/(d+0.5))
        ci_lower = np.exp(log_or - 1.96*se)
        ci_upper = np.exp(log_or + 1.96*se)
        
        return odds_ratio, ci_lower, ci_upper

    def plot_forest(self, channel='Fp1'):
        # 读取数据
        data = pd.read_csv('./save_csv/data/process_filename_frequency_distribution_median.tsv', sep='\t')
        
        # 只选择指定通道的数据
        data = data[data['ch_name'] == channel]
        
        # 创建结果存储DataFrame
        results_df = pd.DataFrame(columns=['channel', 'feature', 'state', 'OR'])
        
        # 定义特征分组
        feature_groups = {
            'others': ['state_entropy', 'spectral_entropy', 'sef_95', 'total_power'],
            'delta': ['delta_proportion', 'delta_power'],
            'theta': ['theta_proportion', 'theta_power'],
            'alpha': ['alpha_proportion', 'alpha_power'],
            'beta': ['beta_proportion', 'beta_power'],
            'gamma': ['gamma_proportion', 'gamma_power']
        }
        
        # 定义每个组的颜色
        group_colors = {
            'others': '#E6B3B3',  # 浅红色
            'delta': '#B3D1E6',   # 浅蓝色
            'theta': '#B3E6CC',   # 浅绿色
            'alpha': '#E6CCB3',   # 浅橙色
            'beta': '#CCB3E6',    # 浅紫色
            'gamma': '#E6E6B3'    # 浅黄色
        }
        
        # 定义每个组的深色版本（用于文字）
        group_colors_dark = {
            'others': '#993333',  # 深红色
            'delta': '#336699',   # 深蓝色
            'theta': '#339966',   # 深绿色
            'alpha': '#996633',   # 深橙色
            'beta': '#663399',    # 深紫色
            'gamma': '#999933'    # 深黄色
        }
        
        # 创建图形
        plt.style.use(['science', 'nature','no-latex'])
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 15))
        fig.subplots_adjust(wspace=0.3)
        fig.suptitle(f'Forest Plot - {channel}', fontsize=36)
        
        for state, ax in zip(['open-eyes', 'close-eyes'], [ax1, ax2]):
            state_data = data[data['state'] == state]
            
            odds_ratios = []
            ci_lowers = []
            ci_uppers = []
            p_values = []
            feature_colors = []  # 存储每个特征的颜色
            
            for feature in self.features:
                # 确定特征所属组
                group = next(g for g, features in feature_groups.items() 
                           if feature in features)
                feature_colors.append(group_colors_dark[group])
                
                pod_data = state_data[state_data['delirium'] == 1][feature]
                nopod_data = state_data[state_data['delirium'] == 0][feature]
                
                pod_data = self.remove_outliers(pod_data)
                nopod_data = self.remove_outliers(nopod_data)
                
                or_value, ci_lower, ci_upper = self.calculate_odds_ratio(pod_data, nopod_data)
                
                comparison = StatisticalComparison(pod_data.values, nopod_data.values)
                p_value = comparison.perform_statistical_test()
                
                odds_ratios.append(or_value)
                ci_lowers.append(ci_lower)
                ci_uppers.append(ci_upper)
                p_values.append(p_value)
                
                new_row = {
                    'channel': channel,
                    'feature': feature,
                    'state': state,
                    'OR': or_value
                }
                results_df = pd.concat([results_df, pd.DataFrame([new_row])], ignore_index=True)
            
            # 绘制森林图
            current_group = None
            group_start_idx = 0
            
            for idx, (or_value, ci_lower, ci_upper, p_value) in enumerate(
                zip(odds_ratios, ci_lowers, ci_uppers, p_values)):
                
                feature_name = self.features[idx]
                # 确定特征所属组
                group = next(g for g, features in feature_groups.items() 
                           if feature_name in features)
                
                # 如果是新的组，添加背景色带
                if group != current_group:
                    if current_group:
                        ax.axhspan(group_start_idx-0.5, idx-0.5, 
                                 color=group_colors[current_group], alpha=0.3)
                    current_group = group
                    group_start_idx = idx
                
                # 设置颜色（显著性用红色，其他用蓝色）
                line_color = 'red' if p_value < 0.05 else '#4169E1'
                
                # 绘制误差线和点
                ax.plot([ci_lower, ci_upper], [idx, idx], color=line_color, linewidth=2)
                ax.scatter(or_value, idx, color=line_color, s=200)
            
            # 为最后一组添加背景色带
            ax.axhspan(group_start_idx-0.5, len(self.features)-0.5, 
                      color=group_colors[current_group], alpha=0.3)
            
            # 设置y轴标签，使用对应的深色
            ax.set_yticks(range(len(self.features)))
            ax.set_yticklabels(self.features, fontsize=20)
            
            # 设置特征标签颜色
            for label, color in zip(ax.get_yticklabels(), feature_colors):
                label.set_color(color)
            
            # 设置x轴
            ax.set_xscale('log')
            ax.set_xticks([0.1, 0.5, 1, 2, 10])
            ax.set_xticklabels(['0.1', '0.5', '1', '2', '10'], fontsize=24)
            ax.axvline(x=1, color='black', linestyle='--', alpha=0.5)
            ax.set_xlabel('Odds Ratio (95% CI)', fontsize=28)
            
            # 添加p值标注到右侧
            ax2 = ax.twinx()
            ax2.set_ylim(ax.get_ylim())
            ax2.set_yticks(range(len(self.features)))
            p_value_labels = [f'p={p:.3f}' for p in p_values]
            ax2.set_yticklabels(p_value_labels, fontsize=16)
            
            # 设置p值标签的颜色
            for i, p in enumerate(p_values):
                color = 'red' if p < 0.05 else '#4169E1'
                ax2.get_yticklabels()[i].set_color(color)
            
            ax.set_title(f'{state}', fontsize=32)
            
        plt.tight_layout()
        
        # 创建保存目录
        os.makedirs('./save_jpg/ForestPlot/', exist_ok=True)
        os.makedirs('./save_csv/ForestPlot/', exist_ok=True)
        
        # 保存图形和数据
        plt.savefig(f'./save_jpg/ForestPlot/{channel}_Forestplot.jpg', dpi=300, bbox_inches='tight')
        results_df.to_csv('./save_csv/ForestPlot/features_statistics_ForestPlot_results.csv', index=False)
        plt.close()

if __name__ == "__main__":
    # 设置要处理的通道
    channels = ['Fp1','Fp2','F3','F4','C3','C4','P3','P4','O1','O2','F7','F8','T3','T4','Fz','Pz']
    forest_plotter = ForestPlotter()
    for channel_ in channels:
        forest_plotter.plot_forest(channel=channel_)