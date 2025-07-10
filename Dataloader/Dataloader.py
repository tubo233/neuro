import os
import pandas as pd
import numpy as np
from glob import glob

class DataLoader:
    def __init__(self, data_path, data_type):
        """
        初始化数据加载器
        Args:
            data_path: 数据根目录路径
            data_type: 数据类型 ('frequency_distribution', 'pac', 'symmetry', 'time-frequency')
        """
        self.data_path = data_path
        self.data_type = data_type
        self.annotations_path = "data/annotations/annotations.csv"
        self.patients_path = "data/annotations/patientsdata_list.xlsx"
        
        # 特征列定义
        self.feature_columns = {
            'frequency_distribution': ['delta_proportion', 'theta_proportion', 'alpha_proportion', 
                                    'beta_proportion', 'gamma_proportion', 'delta_power', 
                                    'theta_power', 'alpha_power', 'beta_power', 'gamma_power',
                                    'state_entropy', 'spectral_entropy', 'sef_95', 'total_power'],
            'pac': ['delta_alpha_mi', 'delta_alpha_mvl', 'delta_beta_mi', 'delta_beta_mvl',
                   'delta_gamma_mi', 'delta_gamma_mvl', 'theta_alpha_mi', 'theta_alpha_mvl',
                   'theta_beta_mi', 'theta_beta_mvl', 'theta_gamma_mi', 'theta_gamma_mvl',
                   'alpha_beta_mi', 'alpha_beta_mvl', 'alpha_gamma_mi', 'alpha_gamma_mvl'],
            'symmetry':['total_ratio','total_dB','delta_ratio',
                        'delta_dB','theta_ratio','theta_dB','alpha_ratio','alpha_dB',
                        'beta_ratio','beta_dB','gamma_ratio','gamma_dB']
        }

    def _read_annotations(self):
        """读取标注文件"""
        annotations = pd.read_csv(self.annotations_path)
        return annotations[annotations['annotation_name'].isin(['open-eyes', 'close-eyes'])]

    def _read_patients_data(self):
        """读取患者信息"""
        return pd.read_excel(self.patients_path, sheet_name='Sheet1')

    def _get_time_periods(self, annotations):
        """获取时间段
        相同状态连续出现两次时，第一次的时间为开始时间，第二次的时间为结束时间
        例如：对于两个连续的open-eyes，第一个的relative_time为start，第二个的为end
        """
        periods = []
        # 按文件名分组，确保同一文件的标注在一起处理
        for filename in annotations['filename'].unique():
            file_annot = annotations[annotations['filename'] == filename].sort_values('relative_time')
            
            i = 0
            while i < len(file_annot) - 1:
                cur_state = file_annot.iloc[i]['annotation_name']
                next_state = file_annot.iloc[i+1]['annotation_name']
                
                # 检查连续两个标注是否为同一状态（open-eyes或close-eyes）
                if cur_state == next_state:
                    periods.append({
                        'state': cur_state,
                        'start': file_annot.iloc[i]['relative_time'],
                        'end': file_annot.iloc[i+1]['relative_time'],
                        'filename': filename
                    })
                    i += 2  # 跳过已处理的一对标注
                else:
                    i += 1  # 移动到下一个标注
        
        return periods

    def _remove_outliers(self, data, columns):
        """使用IQR方法移除异常值"""
        for col in columns:
            Q1 = data[col].quantile(0.25)
            Q3 = data[col].quantile(0.75)
            IQR = Q3 - Q1
            data = data[~((data[col] < (Q1 - 1.5 * IQR)) | (data[col] > (Q3 + 1.5 * IQR)))]
        return data

    def _process_data(self):
        """处理数据并返回结果DataFrame"""
        annotations = self._read_annotations()
        patients_data = self._read_patients_data()
        periods = self._get_time_periods(annotations)
        results = []

        # 按filename分组处理periods
        unique_filenames = {period['filename'] for period in periods}
        
        for filename in unique_filenames:
            # 获取该文件的所有时间段
            file_periods = [p for p in periods if p['filename'] == filename]
            channel_results = []

            if self.data_type in ['frequency_distribution', 'pac', 'symmetry']:
                file_path = os.path.join(self.data_path, 'features', 
                                       f"{filename}_{self.data_type}.tsv")
                if not os.path.exists(file_path):
                    continue

                # 读取数据
                data = pd.read_csv(file_path, sep='\t')
                
                # 设置列名和特征
                if self.data_type == 'frequency_distribution':
                    features = self.feature_columns['frequency_distribution']
                    ch_col = 'ch_name'
                elif self.data_type == 'pac':
                    features = self.feature_columns['pac']
                    ch_col = 'ch_name'
                elif self.data_type == 'symmetry':
                    features = self.feature_columns['symmetry']
                    ch_col = 'target_channels'
                
                # 按通道分组处理
                for ch_name in data[ch_col].unique():
                    ch_data = data[data[ch_col] == ch_name]
                    
                    # 处理每个通道的每个时间段
                    for period in file_periods:
                        # 筛选时间段数据
                        mask = (ch_data['epoch_id'] >= period['start']) & (ch_data['epoch_id'] <= period['end'])
                        period_data = ch_data[mask]
                        
                        if len(period_data) == 0:
                            continue
                            
                        # 不再计算中位数，直接使用所有数据点
                        channel_rows = pd.DataFrame({
                            ch_col: [ch_name] * len(period_data),
                            **{col: period_data[col].values for col in features},
                            'state': [period['state']] * len(period_data),
                            'filename': [filename] * len(period_data),
                            'epoch_id': period_data['epoch_id'].values
                        })
                        channel_results.append(channel_rows)

            else:  # time-frequency
                # 获取该文件名对应的所有通道文件
                file_pattern = os.path.join(self.data_path, 'time-frequency', 
                                          f"{filename}_*_psd.tsv")
                files = glob(file_pattern)
                if not files:
                    continue
                    
                for file_path in files:
                    if not os.path.exists(file_path):
                        continue
                        
                    # 从文件名中提取通道名
                    ch_name = file_path.split('_')[-2]  # 获取通道名
                    
                    # 读取数据
                    data = pd.read_csv(file_path, sep='\t')
                    features = [col for col in data.columns if 'Hz' in col]
                    
                    # 处理每个时间段
                    for period in file_periods:
                        # 筛选时间段数据
                        mask = (data['epoch id'] >= period['start']) & (data['epoch id'] <= period['end'])
                        period_data = data[mask]
                        
                        if len(period_data) == 0:
                            continue
                        
                        # 不再计算中位数，直接使用所有数据点
                        channel_rows = pd.DataFrame({
                            'ch_name': [ch_name] * len(period_data),
                            **{col: period_data[col].values for col in features},
                            'state': [period['state']] * len(period_data),
                            'filename': [filename] * len(period_data),
                            'epoch_id': period_data['epoch id'].values
                        })
                        channel_results.append(channel_rows)
            
            if not channel_results:
                continue

            # 合并该文件的所有通道和时间段的数据
            file_data = pd.concat(channel_results, ignore_index=True)
            
            # 提取患者信息并添加
            patient_name = filename.split('_')[1].split('-')[0]
            patient_info = patients_data[patients_data['name'] == patient_name].iloc[0]
            
            file_data['name'] = patient_name
            file_data['age'] = patient_info['age']
            file_data['LORCeP'] = patient_info['LORCeP']
            file_data['delirium'] = patient_info['delirium']
            
            results.append(file_data)

        if not results:
            return pd.DataFrame()

        return pd.concat(results, ignore_index=True)

    def load_data(self):
        """加载并处理数据"""
        return self._process_data()

if __name__ == "__main__":
    # 示例使用
    data_path = 'data/statistics-cerebral_raw_filtered_ref_average/'
    # data_path = '/workspace/data/nas/DataAndAlgorithm/analyze_candidate/zhujiang-weining/statistics/'
    data_type = 'frequency_distribution'#frequency_distribution symmetry pac time-frequency
    
    # 创建保存目录
    os.makedirs('bo_save_csv/data', exist_ok=True)
    
    # 初始化数据加载器并处理数据
    loader = DataLoader(data_path, data_type)
    processed_data = loader.load_data()
    
    # 保存处理后的数据（移除_median后缀）
    output_path = f'bo_save_csv/data/process_{data_type}_nomedian.tsv'
    processed_data.to_csv(output_path, sep='\t', index=False)
    print(f"Data has been processed and saved to {output_path}")