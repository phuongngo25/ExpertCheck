#comprehensive_training_pipeline.py 

import pandas as pd
import numpy as np
import logging
import os

from src.config.enhanced_config import get_config
from src.models.enhanced_machine_learning_models import EnhancedMLModels
from src.models.enhanced_neural_model import EnhancedNeuralModel
from src.models.enhanced_dt_model import EnhancedDTModel
from src.models.enhanced_dqn_model import EnhancedDQNModel
from src.models.enhanced_ppo_model import EnhancedPPOModel

class ComprehensiveTrainingPipeline:
    
    def __init__(self, config):
        """
        Khởi tạo pipeline với một đối tượng config được truyền từ bên ngoài.
        """
        self.config = config
        self.logger = logging.getLogger(__name__)

        # Khởi tạo tất cả các module mô hình có thể có dựa trên config
        if self.config.model.enable_ml_models:
            self.ml_models = EnhancedMLModels(self.config)
        if self.config.model.enable_neural_models:
            self.neural_models = EnhancedNeuralModel(self.config)
        if hasattr(self.config.model, 'enable_dqn') and self.config.model.enable_dqn:
            self.dqn_model = EnhancedDQNModel(self.config)
        if hasattr(self.config.model, 'enable_dt') and self.config.model.enable_dt:
            self.dt_model = EnhancedDTModel(self.config)
        if hasattr(self.config.model, 'enable_ppo') and self.config.model.enable_ppo:
            self.ppo_model = EnhancedPPOModel(self.config)

    def run_training(self, X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray, y_test: np.ndarray):
        """
        Thực thi pipeline huấn luyện từ dữ liệu đã được chia sẵn.
        """
        self.logger.info(f"Đã nhận dữ liệu huấn luyện: {len(X_train)} mẫu train, {len(X_test)} mẫu test.")

        all_results = {}
        # Huấn luyện lần lượt các nhóm mô hình được kích hoạt trong config
        if self.config.model.enable_ml_models:
            self.logger.info("--- Bắt đầu huấn luyện các mô hình Machine Learning ---")
            all_results.update(self.ml_models.train_and_evaluate_all(X_train, y_train, X_test, y_test))

        if self.config.model.enable_neural_models:
            self.logger.info("--- Bắt đầu huấn luyện các mô hình Neural Network ---")
            all_results.update(self.neural_models.train_and_evaluate_all(X_train, y_train, X_test, y_test))

        if hasattr(self.config.model, 'enable_dqn') and self.config.model.enable_dqn:
            self.logger.info("--- Bắt đầu huấn luyện mô hình DQN ---")
            all_results.update(self.dqn_model.train_dqn(X_train, y_train, X_test, y_test))

        if hasattr(self.config.model, 'enable_dt') and self.config.model.enable_dt:
            self.logger.info("--- Bắt đầu huấn luyện mô hình Decision Transformer ---")
            all_results.update(self.dt_model.train_dt(X_train, y_train, X_test, y_test))
            
        if hasattr(self.config.model, 'enable_ppo') and self.config.model.enable_ppo:
            self.logger.info("--- Bắt đầu huấn luyện mô hình PPO ---")
            all_results.update(self.ppo_model.train_ppo(X_train, y_train, X_test, y_test))

        best_model_info = self._find_best_model(all_results)
        self._print_summary_report(all_results, best_model_info)

        if self.config.output.save_models:
            self._save_all_models()

        return {"training_results": all_results, "best_model_info": best_model_info}

    def _save_all_models(self):
        self.logger.info("--- Bắt đầu lưu các mô hình đã huấn luyện ---")
        output_models_path = os.path.join(self.config.paths.output_dir, self.config.paths.models_dir)
        os.makedirs(output_models_path, exist_ok=True)
        self.logger.info(f"Các mô hình sẽ được lưu tại: {output_models_path}")

        if self.config.model.enable_ml_models and hasattr(self, 'ml_models'):
            self.ml_models.save_models(output_models_path)
        if self.config.model.enable_neural_models and hasattr(self, 'neural_models'):
            self.neural_models.save_models(output_models_path)
        if hasattr(self.config.model, 'enable_dqn') and self.config.model.enable_dqn and hasattr(self, 'dqn_model'):
            self.dqn_model.save_model(output_models_path)
        if hasattr(self.config.model, 'enable_dt') and self.config.model.enable_dt and hasattr(self, 'dt_model'):
            self.dt_model.save_model(output_models_path)
        if hasattr(self.config.model, 'enable_ppo') and self.config.model.enable_ppo and hasattr(self, 'ppo_model'):
            self.ppo_model.save_model(output_models_path)
            
        self.logger.info("--- Lưu mô hình hoàn tất ---")

    def _find_best_model(self, all_results: dict):
        best_score = -1.0
        best_model_name = "None"
        metric_to_optimize = 'test_balanced_accuracy'

        for name, result in all_results.items():
            score = 0.0
            
            # Xử lý linh hoạt các định dạng kết quả khác nhau
            inner_result = result
            # Nếu result là một dict chứa một key duy nhất và value cũng là một dict (trường hợp mô hình RL)
            if isinstance(result, dict) and len(result) == 1 and isinstance(next(iter(result.values())), dict):
                inner_result = next(iter(result.values()))

            if isinstance(inner_result, dict) and metric_to_optimize in inner_result:
                score = inner_result.get(metric_to_optimize, 0.0)
            
            if score > best_score:
                best_score = score
                best_model_name = name
                
        self.logger.info(f"Mô hình tốt nhất là '{best_model_name}' với Balanced Accuracy = {best_score:.4f}")
        return {"name": best_model_name, "balanced_accuracy": best_score}

    def _print_summary_report(self, all_results, best_model_info):
        print("\n" + "="*80)
        print("BÁO CÁO TÓM TẮT KẾT QUẢ HUẤN LUYỆN")
        print("="*80)
        
        report_data = []
        for model_name, result in all_results.items():
            
            inner_result = result
            if isinstance(result, dict) and len(result) == 1 and isinstance(next(iter(result.values())), dict):
                inner_result = next(iter(result.values()))

            if isinstance(inner_result, dict) and 'test_balanced_accuracy' in inner_result:
                report_data.append({
                    "Model": model_name,
                    "Accuracy": inner_result.get('test_accuracy', 0.0),
                    "Balanced Accuracy": inner_result.get('test_balanced_accuracy', 0.0),
                    "F1-Score (Macro)": inner_result.get('classification_report', {}).get('macro avg', {}).get('f1-score', 0.0)
                })

        if not report_data:
            print("Không có kết quả nào để báo cáo.")
            return

        report_df = pd.DataFrame(report_data)
        report_df = report_df.sort_values(by="Balanced Accuracy", ascending=False)
        
        print(report_df.to_string(index=False, float_format="%.4f"))
        print("-" * 80)
        print(f" Mô hình đề xuất: {best_model_info['name']} (Balanced Accuracy: {best_model_info['balanced_accuracy']:.4f})")

        print("="*80)
