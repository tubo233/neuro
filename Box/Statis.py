#显著性差异方法类
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

class StatisticalComparison:
    def __init__(self, group1, group2, alpha=0.05):
        """
        初始化类
        :param group1: 第一组数据(数组)
        :param group2: 第二组数据(数组)
        :param alpha: 显著性水平，默认0.05
        """
        self.group1 = np.array(group1)
        self.group2 = np.array(group2)
        self.alpha = alpha
        self.p_values = None
        self.test_method = None
    
    def check_normality(self):
        """检查两组数据是否符合正态分布"""
        _, p1 = stats.shapiro(self.group1)
        _, p2 = stats.shapiro(self.group2)
        return p1 > self.alpha and p2 > self.alpha
    
    def perform_statistical_test(self):
        """执行统计检验"""
        is_normal = self.check_normality()
        
        if is_normal:
            _, p_value = stats.ttest_ind(self.group1, self.group2)
            self.test_method = "T-test"
        else:
            _, p_value = stats.mannwhitneyu(self.group1, self.group2, alternative='two-sided')
            self.test_method = "Mann-Whitney U test"
        
        return p_value
    
    def plot_comparison(self, x_axis=None, title="Statistical Comparison"):
        """
        绘制比较图并标注显著性差异
        :param x_axis: x轴数据，如果为None则使用索引
        :param title: 图表标题
        """
        if x_axis is None:
            x_axis = np.arange(len(self.group1))
            
        plt.figure(figsize=(12, 6))
        plt.plot(x_axis, self.group1, 'b-', label='Group 1')
        plt.plot(x_axis, self.group2, 'r-', label='Group 2')
        
        # 执行统计检验
        p_value = self.perform_statistical_test()
        
        # 在图中标注显著性差异
        if p_value < self.alpha:
            plt.text(0.02, 0.98, f'p={p_value:.4f} ({self.test_method})\nSignificant difference',
                    transform=plt.gca().transAxes, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.5))
        else:
            plt.text(0.02, 0.98, f'p={p_value:.4f} ({self.test_method})\nNo significant difference',
                    transform=plt.gca().transAxes, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.5))
        
        plt.title(title)
        plt.legend()
        plt.grid(True)
        #plt.show()
        print("----------savepdf------")
        plt.savefig("./save_pdf/Statis/testpdf.png")