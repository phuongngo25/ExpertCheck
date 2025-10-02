# enhanced_machine_learning_models.py 

# enhanced_machine_learning_models.py 

import numpy as np
import joblib
from pathlib import Path
import logging
from typing import Dict, Any
import warnings

import lightgbm as lgb
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.metrics import classification_report, accuracy_score, balanced_accuracy_score

warnings.filterwarnings('ignore')

class EnhancedMLModels:
    def __init__(self, config):
        self.config = config.model
        self.logger = logging.getLogger(__name__)
        self.models_to_run = self.config.ml_models
        self.trained_models = {}

    def _get_param_grids(self) -> Dict[str, Dict[str, Any]]:
        return {
            'random_forest': {
                'n_estimators': [100, 200, 300],
                'max_depth': [10, 20, 30],
                'min_samples_leaf': [1, 2, 4],
                'min_samples_split': [2, 5, 10],
            },
            'lightgbm': {
                'n_estimators': [200, 400, 600],
                'learning_rate': [0.01, 0.05, 0.1],
                'num_leaves': [20, 31, 50],
                'colsample_bytree': [0.7, 0.8],
                'subsample': [0.7, 0.8],
            }
        }

    def _initialize_models(self) -> Dict[str, Any]:
        self.logger.info("Khởi tạo các mô hình ML...")
        base_params = {'random_state': self.config.random_state}
        return {
            'random_forest': RandomForestClassifier(**base_params, n_jobs=-1, class_weight='balanced'),
            'hist_gradient_boosting': HistGradientBoostingClassifier(random_state=self.config.random_state),
            'lightgbm': lgb.LGBMClassifier(**base_params, n_jobs=-1, objective='multiclass', verbose=-1),
        }

    def train_and_evaluate_all(self, X_train, y_train, X_test, y_test, use_tuning=True):
        all_models = self._initialize_models()
        results = {}
        for model_name in self.models_to_run:
            if model_name in all_models:
                self.logger.info(f"--- Bắt đầu xử lý mô hình: {model_name.upper()} ---")
                try:
                    results[model_name] = self._train_and_evaluate_single(
                        model_name, all_models[model_name],
                        X_train, y_train, X_test, y_test, use_tuning
                    )
                except Exception as e:
                    self.logger.error(f"Lỗi khi xử lý {model_name}: {e}", exc_info=True)
        return results

    def _train_and_evaluate_single(self, model_name, model, X_train, y_train, X_test, y_test, use_tuning):
        param_grids = self._get_param_grids()
        
        if use_tuning and model_name in param_grids:
            self.logger.info(f"   Bắt đầu tinh chỉnh siêu tham số cho {model_name} với RandomizedSearchCV...")
            tuner = RandomizedSearchCV(
                estimator=model,
                param_distributions=param_grids[model_name],
                # <<< THAY ĐỔI: Giảm n_iter để tinh chỉnh nhanh hơn >>>
                n_iter=15,
                cv=3,
                verbose=1,
                random_state=self.config.random_state,
                n_jobs=-1,
                scoring='balanced_accuracy'
            )
            tuner.fit(X_train, y_train)
            self.logger.info(f"   Tinh chỉnh hoàn tất. Tham số tốt nhất: {tuner.best_params_}")
            best_model = tuner.best_estimator_
        else:
            self.logger.info(f"   Bắt đầu fit mô hình {model_name} với tham số mặc định...")
            model.fit(X_train, y_train)
            best_model = model
            
        self.trained_models[model_name] = best_model
        
        self.logger.info("   Đang dự đoán trên tập test...")
        test_pred = best_model.predict(X_test)
        
        self.logger.info("   Đang tính toán các chỉ số đánh giá...")
        return {
            'test_accuracy': accuracy_score(y_test, test_pred),
            'test_balanced_accuracy': balanced_accuracy_score(y_test, test_pred),
            'classification_report': classification_report(y_test, test_pred, output_dict=True, zero_division=0)
        }

    def save_models(self, directory: str):
        self.logger.info("Đang lưu các mô hình Machine Learning...")
        ml_models_dir = Path(directory) / 'ML'
        ml_models_dir.mkdir(parents=True, exist_ok=True)
        for name, model in self.trained_models.items():
            try:
                joblib.dump(model, ml_models_dir / f"{name}_model.pkl")
                self.logger.info(f"   -> Đã lưu mô hình '{name}' tại thư mục ML")
            except Exception as e:
                self.logger.error(f"Lỗi khi lưu mô hình '{name}': {e}")