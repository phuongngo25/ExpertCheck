# run_wyoflex_training.py


import logging
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from tensorflow import keras
import joblib

import sys
project_root = os.path.abspath(os.path.dirname(__file__))
if project_root not in sys.path:
    sys.path.append(project_root)

from src.config.enhanced_config import get_config
from src.products.comprehensive_training_pipeline import ComprehensiveTrainingPipeline

# =============================================================================
TARGET_FOREARM = 'right'
# =============================================================================

def prepare_feature_data(config, logger):
    """Chuẩn bị dữ liệu đặc trưng (48 features) cho các model ML, MLP, 2D-CNN."""
    feature_file = f'features_{TARGET_FOREARM}_with_participant.csv'
    feature_path = os.path.join(config.paths.processed_data_dir, feature_file)
    
    if not os.path.exists(feature_path):
        logger.error(f"File đặc trưng '{feature_path}' không tồn tại. Vui lòng chạy 'run_wyoflex_feature_extraction.py' trước.")
        return None, None, None

    logger.info(f"Đang tải dữ liệu đặc trưng từ: {feature_path}")
    df = pd.read_csv(feature_path)
    df.dropna(inplace=True)
    
    feature_cols = [col for col in df.columns if col not in ['participant', 'label']]
    X = df[feature_cols].values
    y = df['label'].values
    groups = df['participant'].values
    return X, y, groups

def prepare_raw_signal_data(config, logger):
    """Chuẩn bị dữ liệu tín hiệu thô theo cửa sổ cho model advanced_cnn."""
    signal_file = f'wyoflex_emg_processed_{TARGET_FOREARM}.csv'
    signal_path = os.path.join(config.paths.processed_data_dir, signal_file)
    
    if not os.path.exists(signal_path):
        logger.error(f"File tín hiệu '{signal_path}' không tồn tại. Vui lòng chạy 'process_wyodata.py' trước.")
        return None, None, None

    logger.info(f"Đang tải và tạo cửa sổ dữ liệu từ: {signal_path}")
    df = pd.read_csv(signal_path)
    df.dropna(inplace=True)
    
    WINDOW_SIZE = config.wyo_processing.segment_length
    NUM_CHANNELS = config.wyo_processing.max_channels
    emg_cols = [f'emg_{i+1}' for i in range(NUM_CHANNELS)]

    X, y, groups = [], [], []
    for p_id, p_group in df.groupby('participant'):
        for label, label_group in p_group.groupby('label'):
            emg_data = label_group[emg_cols].values
            for i in range(0, len(emg_data) - WINDOW_SIZE + 1, WINDOW_SIZE // 4):
                X.append(emg_data[i : i + WINDOW_SIZE])
                y.append(label)
                groups.append(p_id)

    return np.array(X), np.array(y), np.array(groups)

def main():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logger = logging.getLogger("UnifiedTrainer")
    config = get_config()
    
    if TARGET_FOREARM == 'right':
        config.paths.models_dir = config.paths.models_dir_wyo_right
    else:
        config.paths.models_dir = config.paths.models_dir_wyo_left

    # --- 1. Tự động quyết định và chuẩn bị dữ liệu ---
    is_ml_enabled = config.model.enable_ml_models
    is_nn_enabled = config.model.enable_neural_models
    is_advanced_cnn = 'advanced_cnn' in config.model.neural_architectures
    
    if is_nn_enabled and is_advanced_cnn:
        logger.info("Phát hiện model 'advanced_cnn'. Chuẩn bị dữ liệu tín hiệu thô...")
        X, y, groups = prepare_raw_signal_data(config, logger)
        if is_ml_enabled:
            logger.warning("Cảnh báo: Mô hình ML sẽ không được chạy vì dữ liệu đang ở định dạng tín hiệu thô.")
            config.model.enable_ml_models = False
    elif is_ml_enabled or is_nn_enabled:
        logger.info("Chuẩn bị dữ liệu đặc trưng cho model ML/NN tiêu chuẩn...")
        X, y, groups = prepare_feature_data(config, logger)
    else:
        logger.info("Không có mô hình nào được bật trong config. Dừng chương trình.")
        return

    if X is None: return

    # --- Logic Gom Nhóm Cử Chỉ ---
    if config.grouping.enable_gesture_grouping:
        logger.info("TÍNH NĂNG GOM NHÓM ĐANG BẬT.")
        y_df = pd.DataFrame(y, columns=['label'])
        y_df['grouped_label'] = y_df['label'].map(config.grouping.gesture_mapping)
        
        valid_indices = y_df['grouped_label'].notna()
        X = X[valid_indices]
        y = y_df['grouped_label'][valid_indices].values
        groups = groups[valid_indices]
        
        logger.info(f"Đã gom nhóm. Số lượng mẫu còn lại: {len(y)}")

    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    
    # --- 2. Chạy GroupKFold ---
    gkf = GroupKFold(n_splits=5)
    fold_scores, best_model, best_val_accuracy, best_scaler = [], None, 0.0, None
    best_model_name_overall = "None"

    for fold, (train_idx, test_idx) in enumerate(gkf.split(X, y_encoded, groups)):
        logger.info(f"\n{'='*50}\nĐang xử lý Fold {fold + 1}/5\n{'='*50}")
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y_encoded[train_idx], y_encoded[test_idx]

        scaler = StandardScaler()
        original_train_shape, original_test_shape = X_train.shape, X_test.shape
        
        X_train_reshaped = X_train.reshape(-1, original_train_shape[-1])
        X_test_reshaped = X_test.reshape(-1, original_test_shape[-1])
        
        X_train_scaled = scaler.fit_transform(X_train_reshaped).reshape(original_train_shape)
        X_test_scaled = scaler.transform(X_test_reshaped).reshape(original_test_shape)

        pipeline = ComprehensiveTrainingPipeline(config)
        results_dict = pipeline.run_training(X_train_scaled, y_train, X_test_scaled, y_test)
        
        best_fold_model_name = results_dict['best_model_info']['name']
        best_fold_accuracy = results_dict['best_model_info']['balanced_accuracy']
        fold_scores.append(best_fold_accuracy)
        logger.info(f"Fold {fold+1} Best Balanced Accuracy: {best_fold_accuracy:.4f} (from model: {best_fold_model_name})")

        if best_fold_accuracy > best_val_accuracy:
            best_val_accuracy = best_fold_accuracy
            best_model_name_overall = best_fold_model_name
            if config.model.enable_ml_models and hasattr(pipeline, 'ml_models') and best_fold_model_name in pipeline.ml_models.trained_models:
                best_model = pipeline.ml_models.trained_models[best_fold_model_name]
            elif config.model.enable_neural_models and hasattr(pipeline, 'neural_models') and best_fold_model_name in pipeline.neural_models.trained_models:
                best_model = pipeline.neural_models.trained_models[best_fold_model_name]
            best_scaler = scaler
            
    logger.info(f"\n{'='*80}\nKẾT QUẢ CUỐI CÙNG\n{'='*80}")
    logger.info(f"Mean Balanced Accuracy: {np.mean(fold_scores):.4f} (+/- {np.std(fold_scores):.4f})")

    # --- 3. Lưu model và scaler tốt nhất ---
    if best_model is not None:
        logger.info(f"Mô hình tốt nhất là '{best_model_name_overall}' với accuracy {best_val_accuracy:.4f}")
        output_dir = os.path.join(config.paths.output_dir, config.paths.models_dir)
        os.makedirs(output_dir, exist_ok=True)
        
        suffix = "_grouped" if config.grouping.enable_gesture_grouping else ""
        
        model_filename = f'{best_model_name_overall}_best{suffix}.keras' if isinstance(best_model, keras.Model) else f'{best_model_name_overall}_best{suffix}.pkl'
        scaler_filename = f'{best_model_name_overall}_best_scaler{suffix}.pkl'
        
        model_path = os.path.join(output_dir, model_filename)
        scaler_path = os.path.join(output_dir, scaler_filename)
        
        if isinstance(best_model, keras.Model):
            best_model.save(model_path)
        else:
            joblib.dump(best_model, model_path)
        
        joblib.dump(best_scaler, scaler_path)
        logger.info(f"Đã lưu model tốt nhất tại: {model_path}")
        logger.info(f"Đã lưu scaler tương ứng tại: {scaler_path}")

if __name__ == '__main__':
    main()