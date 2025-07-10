import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
import seaborn as sns
import scienceplots
import os
import json
import onnxmltools
from onnxmltools.convert import convert_lightgbm

# 创建所需的目录
os.makedirs('./save_jpg/ROC/', exist_ok=True)
os.makedirs('./save_csv/ROC/', exist_ok=True)
os.makedirs('./save_model/LGBM/', exist_ok=True)

# 设置绘图样式
plt.style.use(['science', 'nature'])

class LGBMClassifier:
    def __init__(self):
        self.data_types = {
            'frequency_distribution': {
                'file': 'save_csv/data/process_filename_frequency_distribution_median.tsv',
                'features': ['delta_proportion', 'theta_proportion', 'alpha_proportion',
                           'beta_proportion', 'gamma_proportion', 'delta_power',
                           'theta_power', 'alpha_power', 'beta_power', 'gamma_power',
                           'state_entropy', 'spectral_entropy', 'sef_95', 'total_power'],
                'ch_col': 'ch_name'
            },
            'pac': {
                'file': 'save_csv/data/process_filename_pac_median.tsv',
                'features': ['delta_alpha_mi', 'delta_alpha_mvl', 'delta_beta_mi', 'delta_beta_mvl',
                           'delta_gamma_mi', 'delta_gamma_mvl', 'theta_alpha_mi', 'theta_alpha_mvl',
                           'theta_beta_mi', 'theta_beta_mvl', 'theta_gamma_mi', 'theta_gamma_mvl',
                           'alpha_beta_mi', 'alpha_beta_mvl', 'alpha_gamma_mi', 'alpha_gamma_mvl'],
                'ch_col': 'ch_name'
            },
            'symmetry': {
                'file': 'save_csv/data/process_filename_symmetry_median.tsv',
                'features': ['total_ratio', 'total_dB', 'delta_ratio',
                           'delta_dB', 'theta_ratio', 'theta_dB', 'alpha_ratio', 'alpha_dB',
                           'beta_ratio', 'beta_dB', 'gamma_ratio', 'gamma_dB'],
                'ch_col': 'target_channels'
            }
        }
        self.states = ['open-eyes', 'close-eyes']
        self.results = []
        
    def prepare_data(self, data, feature, state, ch_col):
        # 筛选特定状态的数据
        state_data = data[data['state'] == state]
        
        # 准备特征数据：每个filename为一个样本，每个通道为一个特征
        X = state_data.pivot(index='filename', columns=ch_col, values=feature)
        
        # 确保每个样本都有完整的特征（所有通道的值）
        if X.isnull().any().any():
            print(f"Warning: Missing values found for {feature} in {state} state")
            X = X.dropna()  # 删除包含缺失值的样本
            
        # 获取每个文件对应的标签（delirium值）并确保文件名匹配
        y = pd.Series(index=X.index, dtype=float)
        for filename in X.index:
            delirium_value = state_data[state_data['filename'] == filename]['delirium'].iloc[0]
            y[filename] = delirium_value
            
        return X, y
    
    def train_and_evaluate(self, X, y):
        # 首先将数据集分为训练集和测试集（7:3）
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
        
        # 在训练集上进行10折交叉验证
        skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
        best_cv_auc = 0
        best_model = None
        
        # 在训练集上进行交叉验证以选择最佳模型
        for fold, (train_idx, val_idx) in enumerate(skf.split(X_train, y_train)):
            X_fold_train, X_fold_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
            y_fold_train, y_fold_val = y_train.iloc[train_idx], y_train.iloc[val_idx]
            
            # 训练模型，设置合适的参数
            model = lgb.LGBMClassifier(
                n_estimators=100,
                learning_rate=0.05,
                max_depth=4,
                num_leaves=15,
                min_child_samples=15,
                random_state=42
            )
            model.fit(X_fold_train, y_fold_train)
            
            # 在验证集上评估
            y_fold_pred = model.predict_proba(X_fold_val)[:, 1]
            fpr, tpr, _ = roc_curve(y_fold_val, y_fold_pred)
            fold_auc = auc(fpr, tpr)
            
            # 保存最佳模型
            if fold_auc > best_cv_auc:
                best_cv_auc = fold_auc
                best_model = model
        
        # 使用最佳模型在测试集上进行最终评估
        y_test_pred = best_model.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_test_pred)
        test_auc = auc(fpr, tpr)
        
        return fpr, tpr, test_auc, best_model
    
    def save_model_onnx(self, model, state, feature):
        model_file = f'./save_model/LGBM/POD-noPOD_Classifier_{state}_{feature}.txt'
        onnx_file = f'./save_model/LGBM/POD-noPOD_Classifier_{state}_{feature}.onnx'
        
        # 首先保存为LightGBM文本格式
        model.booster_.save_model(model_file)
        
        # 读取保存的模型
        with open(model_file, 'r') as f:
            model_str = f.read()
            
        # 转换为ONNX格式
        onx = convert_lightgbm(model.booster_, initial_types=[('input', onnxmltools.convert.common.data_types.FloatTensorType([None, model.n_features_in_]))])
        
        # 保存ONNX模型
        with open(onnx_file, "wb") as f:
            f.write(onx.SerializeToString())
        
        # 删除临时文件
        if os.path.exists(model_file):
            os.remove(model_file)
    
    def plot_roc_curves(self, state, data_type, features):
        n_features = len(features)
        n_cols = 5
        n_rows = (n_features + n_cols - 1) // n_cols
        
        plt.figure(figsize=(25, 5 * n_rows))
        for i, feature in enumerate(features, 1):
            plt.subplot(n_rows, n_cols, i)
            for result in self.results:
                if (result['status'] == state and 
                    result['feature'] == feature and 
                    result['data_type'] == data_type):
                    plt.plot(result['fpr'], result['tpr'],
                            label=f'POD vs noPOD (AUC = {result["AUC"]:.3f})')
                    break
            
            plt.plot([0, 1], [0, 1], 'k--')
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title(f'{feature}')
            plt.legend(loc="lower right")
        
        plt.suptitle(f'ROC Curves for {state} ({data_type})', y=1.02, fontsize=16)
        plt.tight_layout()
        plt.savefig(f'./save_jpg/ROC/{state}_Classifier_ROC_AUC_{data_type}.jpg', 
                    bbox_inches='tight', dpi=300)
        plt.close()
    
    def save_results_csv(self):
        df_results = pd.DataFrame(self.results)
        df_results = df_results[['group', 'data_type', 'feature', 'status', 'AUC']]
        df_results.to_csv('./save_csv/ROC/POD-noPOD_Classifier_AUC_results.csv', 
                         index=False)
    
    def run(self):
        # 对每种数据类型进行处理
        for data_type, config in self.data_types.items():
            print(f"\nProcessing {data_type} data...")
            # 读取数据
            data = pd.read_csv(config['file'], sep='\t')
            
            # 对每个状态和特征进行训练和评估
            for state in self.states:
                for feature in config['features']:
                    print(f"Processing {state} - {feature}")
                    
                    # 准备数据
                    X, y = self.prepare_data(data, feature, state, config['ch_col'])
                    
                    # 训练和评估
                    fpr, tpr, auc_value, best_model = self.train_and_evaluate(X, y)
                    
                    # 保存结果
                    self.results.append({
                        'group': '0+1',
                        'data_type': data_type,
                        'feature': feature,
                        'status': state,
                        'AUC': auc_value,
                        'fpr': fpr,
                        'tpr': tpr
                    })
                    
                    # 保存模型
                    self.save_model_onnx(best_model, state, f"{data_type}_{feature}")
            
            # 为每个状态绘制ROC曲线
            for state in self.states:
                self.plot_roc_curves(state, data_type, config['features'])
        
        # 保存结果到CSV
        self.save_results_csv()

if __name__ == "__main__":
    classifier = LGBMClassifier()
    classifier.run()