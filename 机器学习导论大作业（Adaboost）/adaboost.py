import numpy as np
import pandas as pd
import os

def split_k_fold(X, y, n_splits=10, shuffle=True, random_state=None):
    """
    将数据集分成k份，用于k折交叉验证
    
    Args:
        X: 特征数据，形状为 (n_samples, n_features)
        y: 标签数据，形状为 (n_samples,)
        n_splits: 折数，默认10
        shuffle: 是否打乱数据，默认True
        random_state: 随机数种子，用于复现结果
        
    Returns:
        list of tuples: 每个元素是(train_idx, test_idx)
    """
    n_samples = len(X)
    indices = np.arange(n_samples)
    
    if shuffle:
        if random_state is not None:
            np.random.seed(random_state)
        np.random.shuffle(indices)
    
    fold_size = n_samples // n_splits
    folds = []
    
    for i in range(n_splits):
        start_idx = i * fold_size
        end_idx = start_idx + fold_size if i < n_splits - 1 else n_samples
        
        test_idx = indices[start_idx:end_idx]
        train_idx = np.concatenate([indices[:start_idx], indices[end_idx:]])
        
        folds.append((train_idx, test_idx))
    
    return folds

class LogisticRegression:
    """
    逻辑回归分类器实现
    
    属性:
        reg_lambda: L2正则化系数
        learning_rate: 学习率
        max_iter: 最大迭代次数
        w: 特征权重
        b: 偏置项
        mean_: 特征均值（用于标准化）
        scale_: 特征标准差（用于标准化）
    """
    def __init__(self, reg_lambda=0.001, learning_rate=0.05, max_iter=200):
        self.reg_lambda = reg_lambda
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.w = None
        self.b = 0.0
        self.mean_ = None
        self.scale_ = None

    def _sigmoid(self, z):
        """数值稳定的sigmoid函数实现"""
        z = np.clip(z, -1000, 1000)
        return 1 / (1 + np.exp(-z))

    def standardize(self, X):
        """特征标准化处理"""
        if self.mean_ is None or self.scale_ is None:
            self.mean_ = np.mean(X, axis=0)
            self.scale_ = np.std(X, axis=0)
            self.scale_[self.scale_ == 0] = 1.0
            
        return (X - self.mean_) / self.scale_

    def fit(self, X, y, sample_weight=None):
        """
        训练逻辑回归模型
        
        Args:
            X: 特征矩阵 (n_samples, n_features)
            y: 目标标签 (n_samples,)
            sample_weight: 样本权重 (n_samples,)
        """
        X = self.standardize(X)
        n_samples, n_features = X.shape
        
        self.w = np.zeros(n_features)
        self.b = 0.0
        
        if sample_weight is None:
            sample_weight = np.ones(n_samples)
        sample_weight = sample_weight / np.sum(sample_weight)

        for _ in range(self.max_iter):
            linear = np.dot(X, self.w) + self.b
            prob = self._sigmoid(linear)
            error = prob - y
            
            grad_w = (X.T @ (error * sample_weight)) + self.reg_lambda * self.w
            grad_b = np.sum(error * sample_weight)
            
            self.w -= self.learning_rate * grad_w
            self.b -= self.learning_rate * grad_b

    def predict(self, X, return_proba=False):
        """
        预测样本的类别
        
        Args:
            X: 特征矩阵
            return_proba: 是否返回概率值
            
        Returns:
            预测的类别或概率值
        """
        X = self.standardize(X)
        linear = np.dot(X, self.w) + self.b
        proba = self._sigmoid(linear)
        return proba if return_proba else (proba >= 0.5).astype(int)

class DecisionStump:
    """
    决策树桩分类器实现
    
    属性:
        feature_idx: 选择的特征索引
        threshold: 分割阈值
        polarity: 决策极性(1或-1)
    """
    def __init__(self):
        self.feature_idx = None
        self.threshold = None
        self.polarity = None
        
    def fit(self, X, y, sample_weight=None):
        """
        训练决策树桩
        
        Args:
            X: 训练特征 (n_samples, n_features)
            y: 训练标签 (n_samples,)
            sample_weight: 样本权重 (n_samples,)
        """
        n_samples, n_features = X.shape
        if sample_weight is None:
            sample_weight = np.ones(n_samples) / n_samples
            
        min_error = float('inf')
        best_feature_idx = 0
        best_threshold = 0
        best_polarity = 1
        
        def compute_weighted_error(predictions):
            return np.sum(sample_weight * (predictions != y))
        
        for feature_idx in range(n_features):
            feature_values = X[:, feature_idx]
            percentiles = np.linspace(0, 100, 50)
            thresholds = np.percentile(feature_values, percentiles)
            
            if len(thresholds) < 10:
                thresholds = np.unique(feature_values)
            
            for threshold in thresholds:
                predictions_positive = np.ones(n_samples)
                predictions_positive[feature_values < threshold] = 0
                error_positive = compute_weighted_error(predictions_positive)
                
                predictions_negative = np.ones(n_samples)
                predictions_negative[feature_values >= threshold] = 0
                error_negative = compute_weighted_error(predictions_negative)
                
                if error_positive < min_error:
                    min_error = error_positive
                    best_feature_idx = feature_idx
                    best_threshold = threshold
                    best_polarity = 1
                
                if error_negative < min_error:
                    min_error = error_negative
                    best_feature_idx = feature_idx
                    best_threshold = threshold
                    best_polarity = -1
                
                if min_error < 0.1:
                    break
            
            if min_error < 0.1:
                break
        
        self.feature_idx = best_feature_idx
        self.threshold = best_threshold
        self.polarity = best_polarity
    
    def predict(self, X):
        """
        使用训练好的决策树桩进行预测
        
        Args:
            X: 测试特征 (n_samples, n_features)
            
        Returns:
            预测标签 (n_samples,)
        """
        predictions = np.ones(X.shape[0])
        feature_values = X[:, self.feature_idx]
        
        if self.polarity == 1:
            predictions[feature_values < self.threshold] = 0
        else:
            predictions[feature_values >= self.threshold] = 0
            
        return predictions

class AdaBoost:
    """
    AdaBoost集成学习算法实现
    
    属性:
        base_classifier_type: 基分类器类型 ('logistic'或'stump')
        n_estimators: 基分类器数量
        classifiers: 基分类器列表
        alphas: 每个基分类器的权重系数
    """
    def __init__(self, base_classifier_type='logistic', n_estimators=100):
        self.base_classifier_type = base_classifier_type
        self.n_estimators = n_estimators
        self.classifiers = []
        self.alphas = []
        
    def fit(self, X, y):
        """
        训练AdaBoost模型
        
        Args:
            X: 训练特征 (n_samples, n_features)
            y: 训练标签 (n_samples,)
        """
        n_samples = X.shape[0]
        w = np.ones(n_samples) / n_samples
        
        for t in range(self.n_estimators):
            if self.base_classifier_type == 'logistic':
                classifier = LogisticRegression()
            else:
                classifier = DecisionStump()
            
            classifier.fit(X, y, sample_weight=w)
            y_pred = classifier.predict(X)
            
            error = np.sum(w * (y_pred != y))
            
            if error > 0.5:
                y_pred = 1 - y_pred
                error = 1 - error

            alpha = 0.5 * np.log((1 - error) / (error + 1e-10))
            
            w *= np.exp(-alpha * (2*y-1) * (2*y_pred-1))
            w /= np.sum(w)
            
            self.classifiers.append(classifier)
            self.alphas.append(alpha)
    
    def predict(self, X):
        """
        使用训练好的AdaBoost模型进行预测
        
        Args:
            X: 测试特征 (n_samples, n_features)
            
        Returns:
            预测标签 (n_samples,)
        """
        predictions = np.zeros(X.shape[0])
        
        for alpha, classifier in zip(self.alphas, self.classifiers):
            predictions += alpha * (2*classifier.predict(X)-1)
            
        return np.where(predictions > 0, 1, 0)