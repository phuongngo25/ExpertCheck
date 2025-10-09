
# run_wyoflex_training.py 

import logging
import os
import json
import pandas as pd
import numpy as np
from scipy import signal 
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
import joblib
import sys
import glob
from tqdm import tqdm

# Đảm bảo có thể import từ các thư mục con
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.append(project_root)

from src.config.enhanced_config import get_config
from src.models.enhanced_neural_model import EnhancedNeuralModel
from src.models.enhanced_machine_learning_models import EnhancedMLModels
from src.features.advanced_feature_extraction import AdvancedFeatureExtractor
from src.products.comprehensive_training_pipeline import ComprehensiveTrainingPipeline

# --- CÁC HÀM CHUẨN BỊ DỮ LIỆU CHUYÊN BIỆT ---

def _get_common_paths(config):
    """Lấy các đường dẫn file động dựa trên config."""
    target_forearm = config.project.target_forearm
    offset_type = "O1" if config.data_prep.use_offset_data else "O2"
    raw_filename = config.paths.wyoflex_emg_processed_template.format(forearm=target_forearm, offset_type=offset_type)
    raw_path = os.path.join(config.paths.processed_data_dir, raw_filename)
    feature_filename = config.paths.features_template.format(forearm=target_forearm, offset_type=offset_type)
    feature_path = os.path.join(config.paths.processed_data_dir, feature_filename)
    return raw_path, feature_path

def prepare_dann_data(config, logger):
    """Chuẩn bị dữ liệu cho mô hình DANN (tín hiệu thô + nhãn kép)."""
    raw_path, _ = _get_common_paths(config)
    if not os.path.exists(raw_path):
        logger.error(f"File tín hiệu '{raw_path}' không tồn tại.")
        return None, None, None, None
    
    logger.info(f"Đang tải và tạo chuỗi tín hiệu thô cho DANN từ: {raw_path}")
    df = pd.read_csv(raw_path).dropna()
    
    cfg = config.dann
    emg_cols = [f'emg_{i+1}' for i in range(4)]
    X_sequences, y_gestures, y_users = [], [], []

    for (p_id, m_id, c_id), group in tqdm(df.groupby(['participant', 'movement', 'cycle']), desc="Preparing DANN Data"):
        trial_signal = group.iloc[4900:11100][emg_cols].values
        for i in range(0, len(trial_signal) - cfg.sequence_length + 1, cfg.sequence_step):
            X_sequences.append(trial_signal[i : i + cfg.sequence_length])
            y_gestures.append(m_id)
            y_users.append(p_id)

    if not X_sequences: return None, None, None, None
        
    # Dữ liệu nhãn cho DANN là một dictionary
    y_combined = {'gesture': np.array(y_gestures), 'user': np.array(y_users)}
    
    return np.array(X_sequences), y_combined, np.array(y_users), 'dann_sequence'

def prepare_flat_feature_data(config, logger):
    """Chuẩn bị dữ liệu dạng vector đặc trưng phẳng (cho ML)."""
    _, feature_path = _get_common_paths(config)
    if not os.path.exists(feature_path):
        logger.error(f"File đặc trưng '{feature_path}' không tồn tại. Vui lòng chạy 'run_wyoflex_feature_extraction.py' trước.")
        return None, None, None, None
    logger.info(f"Đang tải dữ liệu đặc trưng phẳng từ: {feature_path}")
    df = pd.read_csv(feature_path).dropna()
    feature_cols = [col for col in df.columns if col not in ['participant', 'label']]
    return df[feature_cols].values, df['label'].values, df['participant'].values, 'flat_features'

def prepare_raw_signal_sequences(config, logger):
    """Chuẩn bị dữ liệu dạng chuỗi tín hiệu thô (cho CNN-LSTM)."""
    raw_path, _ = _get_common_paths(config)
    if not os.path.exists(raw_path): return None, None, None, None
    
    logger.info(f"Đang tải và tạo 'Trượt Cửa Sổ Chuỗi' từ: {raw_path}")
    df = pd.read_csv(raw_path).dropna()
    
    cfg = config.cnn_lstm
    emg_cols = [f'emg_{i+1}' for i in range(4)]
    X_sequences, y_sequences, groups_sequences = [], [], []

    for (p_id, m_id, c_id), group in tqdm(df.groupby(['participant', 'movement', 'cycle']), desc="Preparing Raw Sequences"):
        trial_signal = group.iloc[4900:11100][emg_cols].values
        for i in range(0, len(trial_signal) - cfg.sequence_length + 1, cfg.sequence_step):
            X_sequences.append(trial_signal[i : i + cfg.sequence_length])
            y_sequences.append(m_id)
            groups_sequences.append(p_id)

    if not X_sequences: return None, None, None, None
    return np.array(X_sequences), np.array(y_sequences), np.array(groups_sequences), 'raw_signal_sequence'

def prepare_feature_sequences(config, logger):
    """Chuẩn bị dữ liệu dạng chuỗi vector đặc trưng (cho LSTM-on-Features)."""
    X_features, y_labels, groups, _ = prepare_flat_feature_data(config, logger)
    if X_features is None: return None, None, None, None
        
    logger.info(f"Đã tải {len(X_features)} vector. Đang tạo chuỗi...")
    SEQ_LENGTH = config.lstm_on_features.sequence_length
    X_seq, y_seq, groups_seq = [], [], []

    temp_df = pd.DataFrame(X_features)
    temp_df['group'] = groups
    temp_df['label'] = y_labels

    for p_id, p_group in temp_df.groupby('group'):
        p_features = p_group.drop(columns=['group', 'label']).values
        p_labels = p_group['label'].values
        for i in range(len(p_features) - SEQ_LENGTH + 1):
            X_seq.append(p_features[i : i + SEQ_LENGTH])
            y_seq.append(p_labels[i + SEQ_LENGTH - 1])
            groups_seq.append(p_id)

    return np.array(X_seq, dtype=np.float32), np.array(y_seq), np.array(groups_seq), 'feature_sequence'

def prepare_spectrogram_data(config, logger):
    """Chuẩn bị dữ liệu dạng ảnh Spectrogram (cho 2D-CNN)."""
    raw_path, _ = _get_common_paths(config)
    if not os.path.exists(raw_path): return None, None, None, None

    logger.info(f"Đang tải và tạo ảnh Spectrogram từ: {raw_path}")
    df = pd.read_csv(raw_path).dropna()
    
    cfg_spec = config.spectrogram
    emg_cols = [f'emg_{i+1}' for i in range(4)]
    X_spectrograms, y_labels, all_groups = [], [], []

    for (p_id, m_id, c_id), group in tqdm(df.groupby(['participant', 'movement', 'cycle']), desc="Creating Spectrograms"):
        trial_signal = group.iloc[4900:11100][emg_cols].values
        if trial_signal.shape[0] < cfg_spec.nperseg: continue
        
        channels_spectrograms = []
        for i in range(trial_signal.shape[1]):
            f, t, Zxx = signal.stft(trial_signal[:, i], fs=1000, nperseg=cfg_spec.nperseg, noverlap=cfg_spec.noverlap)
            spectrogram = np.abs(Zxx)
            if cfg_spec.log_spectrogram:
                spectrogram = np.log(spectrogram + 1e-10)
            channels_spectrograms.append(spectrogram)
        
        multi_channel_spectrogram = np.stack(channels_spectrograms, axis=-1)
        X_spectrograms.append(multi_channel_spectrogram)
        y_labels.append(m_id)
        all_groups.append(p_id)

    return np.array(X_spectrograms, dtype=np.float32), np.array(y_labels), np.array(all_groups), 'spectrogram'

def run_ovr_training_pipeline(config, X, y, groups):
    """
    Thực thi quy trình huấn luyện One-vs-Rest (OvR).
    Hàm này sẽ huấn luyện một mô hình nhị phân cho mỗi lớp cử chỉ.
    """
    logger = logging.getLogger("WyoFlexTrainer.OVR")
    unique_classes = sorted(np.unique(y))
    num_classes = len(unique_classes)
    logger.info(f">>> BẮT ĐẦU QUY TRÌNH HUẤN LUYỆN ONE-VS-REST cho {num_classes} lớp <<<")

    output_dir = os.path.join(config.paths.output_dir, config.paths.models_dir_wyo_right) \
        if config.project.target_forearm == 'right' \
        else os.path.join(config.paths.output_dir, config.paths.models_dir_wyo_left)
    os.makedirs(output_dir, exist_ok=True)
    
    # Chỉ cần lưu 1 scaler chung cho tất cả các model
    main_scaler = StandardScaler()
    main_scaler.fit(X.reshape(-1, X.shape[-1]))
    scaler_path = os.path.join(output_dir, "ovr_main_scaler.pkl")
    joblib.dump(main_scaler, scaler_path)
    logger.info(f"Đã lưu Scaler chung cho OvR tại: {scaler_path}")

    # Lặp qua từng lớp để huấn luyện một model riêng
    for class_label in unique_classes:
        logger.info(f"\n{'='*30} Đang chuẩn bị cho LỚP {class_label} {'='*30}")

        # 1. Chuyển đổi nhãn thành nhị phân: (lớp hiện tại -> 1, các lớp khác -> 0)
        y_binary = np.where(y == class_label, 1, 0)
        logger.info(f"Đã chuyển đổi nhãn. Lớp {class_label} là Positive (1), còn lại là Negative (0).")

        # 2. Sử dụng GroupKFold để đánh giá khách quan cho từng bài toán nhị phân
        gkf = GroupKFold(n_splits=5)
        best_model_for_class = None
        best_accuracy_for_class = -1.0
        best_model_name_for_class = "None"

        for fold, (train_idx, test_idx) in enumerate(gkf.split(X, y_binary, groups)):
            logger.info(f"--- [Lớp {class_label}] Đang xử lý Fold {fold + 1}/{gkf.get_n_splits()} ---")
            X_train_fold, X_test_fold = X[train_idx], X[test_idx]
            y_train_fold, y_test_fold = y_binary[train_idx], y_binary[test_idx]
            
            # Sử dụng scaler đã fit trên toàn bộ dữ liệu
            X_train_scaled = main_scaler.transform(X_train_fold.reshape(-1, X_train_fold.shape[-1])).reshape(X_train_fold.shape)
            X_test_scaled = main_scaler.transform(X_test_fold.reshape(-1, X_test_fold.shape[-1])).reshape(X_test_fold.shape)

            # 3. Gọi pipeline huấn luyện như bình thường
            pipeline = ComprehensiveTrainingPipeline(config)
            results_dict = pipeline.run_training(X_train_scaled, y_train_fold, X_test_scaled, y_test_fold, data_format='flat_features')
            
            if not results_dict or not results_dict.get('best_model_info'): continue

            best_model_info_fold = results_dict['best_model_info']
            fold_accuracy = best_model_info_fold['balanced_accuracy']

            if fold_accuracy > best_accuracy_for_class:
                best_accuracy_for_class = fold_accuracy
                best_model_name_for_class = best_model_info_fold['name']
                if config.model.enable_ml_models and hasattr(pipeline, 'ml_models') and best_model_name_for_class in pipeline.ml_models.trained_models:
                    best_model_for_class = pipeline.ml_models.trained_models[best_model_name_for_class]

        # 4. Lưu lại model tốt nhất cho lớp này
        if best_model_for_class is not None:
            logger.info(f"🏆 Model tốt nhất cho lớp {class_label} là '{best_model_name_for_class}' với Balanced Accuracy = {best_accuracy_for_class:.4f}")
            
            suffix = "_grouped" if config.grouping.enable_gesture_grouping else ""
            # Tên file chứa thông tin về lớp nó phân loại
            model_filename = f"{best_model_name_for_class}_ovr_class_{class_label}{suffix}.pkl"
            model_path = os.path.join(output_dir, model_filename)
            joblib.dump(best_model_for_class, model_path)
            logger.info(f"✅ Đã lưu model tại: {model_path}")
        else:
            logger.warning(f"Không tìm được model tốt nhất cho lớp {class_label}.")


def main():
    """Hàm chính, điều phối toàn bộ quy trình huấn luyện một cách linh hoạt."""
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logger = logging.getLogger("WyoFlexTrainer")
    config = get_config()
    
    X, y, groups, data_format = None, None, None, None
    experiment_mode = config.project.experiment_mode
    logger.info(f"Chế độ thử nghiệm được chọn: '{experiment_mode}'. Chuẩn bị dữ liệu...")

    # --- LOGIC ĐIỀU HƯỚNG LINH HOẠT ---
    if experiment_mode in ['ml_models', 'train']:
        X, y, groups, data_format = prepare_flat_feature_data(config, logger)
    elif experiment_mode == 'cnn_lstm':
        X, y, groups, data_format = prepare_raw_signal_sequences(config, logger)
    elif experiment_mode == 'lstm_on_features':
        X, y, groups, data_format = prepare_feature_sequences(config, logger)
    # Thêm các chế độ khác nếu cần
    else:
        logger.error(f"Chế độ thử nghiệm '{experiment_mode}' không được hỗ trợ.")
        return

    if X is None or len(X) == 0:
        logger.error("Chuẩn bị dữ liệu thất bại. Dừng chương trình.")
        return

    logger.info(f"Chuẩn bị dữ liệu thành công. Shape của X: {X.shape}.")

    # --- PHẦN CÒN LẠI CỦA PIPELINE ---
    if config.model.enable_ovr_strategy:
        logger.info("Phát hiện cờ OVR. Chuyển sang pipeline One-vs-Rest.")
        # Dữ liệu cho OVR phải là 2D
        if data_format != 'flat_features':
            logger.error("Chế độ OvR chỉ hỗ trợ dữ liệu 'flat_features'.")
            return
        run_ovr_training_pipeline(config, X, y, groups)
    else:
        # Chạy pipeline đa lớp tiêu chuẩn...
        # (Toàn bộ code xử lý đa lớp của bạn nằm ở đây)
        logger.info(">>> BẮT ĐẦU QUY TRÌNH HUẤN LUYỆN ĐA LỚP TIÊU CHUẨN <<<")
        
        # --- BẮT ĐẦU PHẦN CODE HUẤN LUYỆN ĐA LỚP GỐC ---

        # 1. XỬ LÝ GOM NHÓM VÀ ENCODE NHÃN
        if config.grouping.enable_gesture_grouping:
            logger.info("Áp dụng tính năng gom nhóm cử chỉ...")
            original_y_series = pd.Series(y)
            y = original_y_series.map(config.grouping.gesture_mapping).values
            
            # Lọc bỏ các mẫu không có trong mapping
            valid_indices = ~np.isnan(y)
            X = X[valid_indices]
            y = y[valid_indices].astype(int)
            groups = groups[valid_indices]
            logger.info(f"Dữ liệu sau khi gom nhóm còn lại {len(y)} mẫu.")

        le = LabelEncoder()
        y_encoded = le.fit_transform(y)

        # 2. KHỞI TẠO VÒNG LẶP ĐÁNH GIÁ CHÉO (CROSS-VALIDATION)
        gkf = GroupKFold(n_splits=5)
        fold_scores = []
        best_model_overall, best_scaler_overall = None, None
        best_accuracy_overall, best_model_name_overall = -1.0, "None"

        for fold, (train_idx, test_idx) in enumerate(gkf.split(X, y_encoded, groups)):
            logger.info(f"\n--- Đang xử lý Fold {fold + 1}/{gkf.get_n_splits()} ---")
            X_train_fold, X_test_fold = X[train_idx], X[test_idx]
            y_train_fold, y_test_fold = y_encoded[train_idx], y_encoded[test_idx]
            
            scaler = StandardScaler()
            # Reshape, scale, và reshape lại về hình dạng ban đầu
            X_train_scaled = scaler.fit_transform(X_train_fold.reshape(-1, X_train_fold.shape[-1])).reshape(X_train_fold.shape)
            X_test_scaled = scaler.transform(X_test_fold.reshape(-1, X_test_fold.shape[-1])).reshape(X_test_fold.shape)

            # 3. GỌI PIPELINE HUẤN LUYỆN
            pipeline = ComprehensiveTrainingPipeline(config)
            
            # Gán label encoder cho module neural network nếu được sử dụng
            if config.model.enable_neural_models and hasattr(pipeline, 'neural_models'):
                pipeline.neural_models.label_encoder = le
                
            results_dict = pipeline.run_training(X_train_scaled, y_train_fold, X_test_scaled, y_test_fold, data_format=data_format)
            
            if not results_dict or not results_dict.get('best_model_info'): 
                logger.warning(f"Fold {fold+1} không trả về kết quả hợp lệ.")
                continue

            # 4. CẬP NHẬT KẾT QUẢ TỐT NHẤT
            best_model_info_fold = results_dict['best_model_info']
            fold_accuracy = best_model_info_fold['balanced_accuracy']
            fold_scores.append(fold_accuracy)

            if fold_accuracy > best_accuracy_overall:
                best_accuracy_overall = fold_accuracy
                best_model_name_overall = best_model_info_fold['name']
                best_scaler_overall = scaler
                # Lấy đối tượng model đã được huấn luyện từ pipeline
                if config.model.enable_ml_models and hasattr(pipeline, 'ml_models') and best_model_name_overall in pipeline.ml_models.trained_models:
                    best_model_overall = pipeline.ml_models.trained_models[best_model_name_overall]
                elif config.model.enable_neural_models and hasattr(pipeline, 'neural_models') and best_model_name_overall in pipeline.neural_models.trained_models:
                    best_model_overall = pipeline.neural_models.trained_models[best_model_name_overall]

        # 5. IN BÁO CÁO TỔNG KẾT VÀ LƯU MODEL TỐT NHẤT
        logger.info(f"\n{'='*80}\nKẾT QUẢ CUỐI CÙNG (ĐA LỚP TIÊU CHUẨN)\n{'='*80}")
        if not fold_scores:
            logger.error("Không có kết quả nào được ghi nhận sau các fold. Không thể báo cáo.")
            return
            
        logger.info(f"Mean Balanced Accuracy: {np.mean(fold_scores):.4f} (+/- {np.std(fold_scores):.4f})")
        logger.info(f"🏆 Model tốt nhất là '{best_model_name_overall}' với accuracy {best_accuracy_overall:.4f}")

        if best_model_overall is not None and config.output.save_models:
            logger.info(f"--- Đang lưu model tốt nhất: {best_model_name_overall} ---")
            
            output_dir = os.path.join(config.paths.output_dir, config.paths.models_dir_wyo_right) if config.project.target_forearm == 'right' else os.path.join(config.paths.output_dir, config.paths.models_dir_wyo_left)
            os.makedirs(output_dir, exist_ok=True)
            
            suffix = "_grouped" if config.grouping.enable_gesture_grouping else ""
            base_filename = f"{best_model_name_overall}_best{suffix}"
            
            # Lưu scaler
            scaler_path = os.path.join(output_dir, f"{base_filename}_scaler.pkl")
            joblib.dump(best_scaler_overall, scaler_path)
            logger.info(f"Đã lưu Scaler tại: {scaler_path}")
            
            # Lưu model
            if isinstance(best_model_overall, keras.Model):
                model_path = os.path.join(output_dir, f"{base_filename}.keras")
                best_model_overall.save(model_path)
            else:
                model_path = os.path.join(output_dir, f"{base_filename}.pkl")
                joblib.dump(best_model_overall, model_path)
            logger.info(f"Đã lưu Model tại: {model_path}")

            # Lưu label encoder và metadata
            joblib.dump(le, os.path.join(output_dir, f"{base_filename}_label_encoder.pkl"))

            metadata_path = os.path.join(output_dir, f"{base_filename}_metadata.json")
            metadata = {"model_name": best_model_name_overall, "data_format": data_format}
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=4)
            logger.info(f"Đã lưu Metadata tại: {metadata_path}")

        # --- KẾT THÚC PHẦN CODE HUẤN LUYỆN ĐA LỚP GỐC ---

if __name__ == '__main__':
    main()
