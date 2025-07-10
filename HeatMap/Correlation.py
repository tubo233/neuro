import numpy as np
from scipy import stats

class CorrelationAnalyzer:
    """数据相关性分析类"""
    
    def __init__(self, data1, data2):
        """
        初始化函数
        :param data1: 第一组数据（numpy数组）
        :param data2: 第二组数据（numpy数组）
        """
        self.data1 = np.array(data1)
        self.data2 = np.array(data2)
        
    def check_normality(self, data):
        """
        使用Shapiro-Wilk检验判断数据是否符合正态分布
        :param data: 输入数据
        :return: 布尔值，True表示符合正态分布
        """
        _, p_value = stats.shapiro(data)
        return p_value > 0.05  # 显著性水平设为0.05
    
    def calculate_correlation(self):
        """
        计算两组数据的相关性
        :return: 相关系数和相关系数类型
        """
        # 检查两组数据是否都符合正态分布
        is_normal1 = self.check_normality(self.data1)
        is_normal2 = self.check_normality(self.data2)
        
        # 如果两组数据都符合正态分布，使用皮尔逊相关系数
        if is_normal1 and is_normal2:
            correlation, p_value = stats.pearsonr(self.data1, self.data2)
            method = "Pearson"
        # 否则使用斯皮尔曼秩相关系数
        else:
            correlation, p_value = stats.spearmanr(self.data1, self.data2)
            method = "Spearman"
            
        return {
            "correlation": correlation,
            "p_value": p_value,
            "method": method,
            "is_significant": p_value < 0.05
        }

# 使用示例
if __name__ == "__main__":
    # 生成示例数据
    # 正态分布数据示例
    np.random.seed(42)
    normal_data1 = np.random.normal(0, 1, 100)
    normal_data2 = 2 * normal_data1 + np.random.normal(0, 0.5, 100)
    
    # 非正态分布数据示例
    non_normal_data1 = np.random.exponential(1, 100)
    non_normal_data2 = 2 * non_normal_data1 + np.random.exponential(0.5, 100)
    
    # 示例1：正态分布数据
    print("示例1：正态分布数据分析")
    analyzer1 = CorrelationAnalyzer(normal_data1, normal_data2)
    result1 = analyzer1.calculate_correlation()
    print(f"相关系数类型: {result1['method']}")
    print(f"相关系数: {result1['correlation']:.4f}")
    print(f"P值: {result1['p_value']:.4f}")
    print(f"是否显著: {result1['is_significant']}\n")
    
    # 示例2：非正态分布数据
    print("示例2：非正态分布数据分析")
    analyzer2 = CorrelationAnalyzer(non_normal_data1, non_normal_data2)
    result2 = analyzer2.calculate_correlation()
    print(f"相关系数类型: {result2['method']}")
    print(f"相关系数: {result2['correlation']:.4f}")
    print(f"P值: {result2['p_value']:.4f}")
    print(f"是否显著: {result2['is_significant']}")