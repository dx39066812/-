import numpy as np
import pandas as pd
import sys
from adaboost import AdaBoost, split_k_fold
import time
import os

def calculate_accuracy(y_true, y_pred):
    """
    计算预测准确率
    
    Args:
        y_true: 真实标签
        y_pred: 预测标签
        
    Returns:
        float: 准确率（正确预测的样本比例）
    """
    return np.mean(y_true == y_pred)

def analyze_data_distribution(y):
    """
    分析数据集的类别分布
    
    Args:
        y: 标签数据
    """
    unique_labels, counts = np.unique(y, return_counts=True)
    print("\nData distribution:")
    for label, count in zip(unique_labels, counts):
        print(f"Class {label}: {count} samples ({count/len(y)*100:.2f}%)")

def main():
    """
    主函数：实现AdaBoost算法的训练、评估和结果输出流程
    
    命令行参数:
        argv[1]: 特征数据文件路径
        argv[2]: 标签数据文件路径
        argv[3]: 基分类器类型（0表示逻辑回归，1表示决策树桩）
    """
    if len(sys.argv) != 4:
        print("Usage: python main.py /path/to/data/data.csv /path/to/data/target.csv <base_classifier_type>")
        print("base_classifier_type: 0 for logistic regression, 1 for decision stump")
        sys.exit(1)
    
    data_path = sys.argv[1]
    target_path = sys.argv[2]
    base_classifier_type = int(sys.argv[3])
    
    if base_classifier_type not in [0, 1]:
        print("Error: base_classifier_type must be 0 or 1")
        print("0: logistic regression")
        print("1: decision stump")
        sys.exit(1)
    
    classifier_type = 'logistic' if base_classifier_type == 0 else 'stump'
    print(f"\nUsing {classifier_type} as base classifier...")
    
    try:
        print("\nLoading data...")
        X = pd.read_csv(data_path).values
        y = pd.read_csv(target_path).values.ravel()
        print(f"Data loaded: {X.shape[0]} samples, {X.shape[1]} features")
        
        analyze_data_distribution(y)
        
        if set(y) == {-1, 1}:
            y = (y + 1) / 2
        
        if not os.path.exists('experiments'):
            os.makedirs('experiments')
            print("Created experiments directory")
        
        n_estimators_list = [1, 5, 10, 100]
        
        for n_estimators in n_estimators_list:
            print(f"\nTraining with {n_estimators} base classifiers...")
            accuracies = []
            start_time = time.time()
            
            folds = split_k_fold(X, y, n_splits=10, shuffle=True, random_state=42)
            
            for fold_idx, (train_idx, test_idx) in enumerate(folds, 1):
                print(f"  Fold {fold_idx}/10...", end='', flush=True)
                fold_start_time = time.time()
                
                X_train, X_test = X[train_idx], X[test_idx]
                y_train, y_test = y[train_idx], y[test_idx]
                
                model = AdaBoost(base_classifier_type=classifier_type, 
                               n_estimators=n_estimators)
                model.fit(X_train, y_train)
                
                y_pred = model.predict(X_test)
                accuracy = calculate_accuracy(y_test, y_pred)
                accuracies.append(accuracy)
                
                ordered_indices = np.arange(len(X))[test_idx] + 1
                
                output_file = f'experiments/base{n_estimators}_fold{fold_idx}.csv'
                predictions_df = pd.DataFrame({
                    'index': ordered_indices,
                    'prediction': y_pred
                })
                predictions_df = predictions_df.sort_values('index')
                predictions_df.to_csv(output_file, index=False)
                
                fold_time = time.time() - fold_start_time
                print(f" completed in {fold_time:.2f}s, accuracy: {accuracy:.4f}")
            
            total_time = time.time() - start_time
            print(f"Completed {n_estimators} base classifiers in {total_time:.2f}s")
            print(f"Average accuracy: {np.mean(accuracies):.4f}")
            
    except Exception as e:
        print(f"Error: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()