# src/products/training_pipeline_wyo.py (PHIÊN BẢN CUỐI CÙNG)

import logging
import pandas as pd
import numpy as np
import os
import joblib
import json
from sklearn.model_selection import GroupKFold, train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from tensorflow import keras
from keras.preprocessing.sequence import pad_sequences
from scipy import signal

from src.config.enhanced_config import get_config
from src.models.enhanced_neural_model import EnhancedNeuralModel
from src.features.advanced_feature_extraction import AdvancedFeatureExtractor 

class Trainer:
    """
    Lớp quản lý quy trình huấn luyện chuyên biệt cho bộ dữ liệu WyoFlex.
    Tự động chuẩn bị đúng loại dữ liệu dựa trên cấu hình thử nghiệm.
    """
    def __init__(self, config):
        self.config = config
        self.logger = logging.getLogger("WyoFlexTrainer")
        self.forearm = self.config.project.target_forearm
        self.feature_extractor = AdvancedFeatureExtractor(config)
        self.data_format = None
        self.logger.info(f"Trainer được khởi tạo cho TAY: '{self.forearm.upper()}'")

    @property
    def model_output_dir(self):
        """Trả về đường dẫn thư mục model tương ứng."""
        if self.forearm == 'right':
            return os.path.join(self.config.paths.output_dir, self.config.paths.models_dir_wyo_right)
        return os.path.join(self.config.paths.output_dir, self.config.paths.models_dir_wyo_left)

    def prepare_data(self):
        """Hàm 'chuyển mạch' - Tự động chọn đúng hàm chuẩn bị dữ liệu."""
        mode = self.config.project.experiment_mode
        self.logger.info(f"Chế độ thử nghiệm được chọn: '{mode}'. Chuẩn bị dữ liệu...")

        if mode == 'spectrogram_cnn':
            self.data_format = 'spectrogram'
            return self._prepare_spectrogram_data()
        elif mode == 'cnn_lstm':
            self.data_format = 'raw_signal_sequence'
            return self._prepare_raw_signal_sequences()
        elif mode == 'lstm_on_features':
            self.data_format = 'feature_sequence'
            return self._prepare_feature_sequences()
        else:
            self.logger.error(f"Chế độ thử nghiệm '{mode}' không được hỗ trợ.")
            return None

    def _prepare_spectrogram_data(self):
        """Tải dữ liệu thô và biến đổi thành các 'ảnh' Spectrogram."""
        self.logger.info("  -> Đang thực thi: Tín hiệu thô -> Spectrogram 2D")
        cfg_spec = self.config.spectrogram
        offset_type = "O1" if self.config.data_prep.use_offset_data else "O2"
        filename = self.config.paths.wyoflex_emg_processed_template.format(forearm=self.forearm, offset_type=offset_type)
        signal_path = os.path.join(self.config.paths.processed_data_dir, filename)

        if not os.path.exists(signal_path):
            self.logger.error(f"File tín hiệu '{signal_path}' không tồn tại.")
            return None
            
        df = pd.read_csv(signal_path)
        df.dropna(inplace=True)
        
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

        X_final = np.array(X_spectrograms, dtype=np.float32)
        self.logger.info(f"Hoàn tất. Đã tạo {len(X_final)} ảnh. Shape dữ liệu: {X_final.shape}")
        return {'X': X_final, 'y': np.array(y_labels), 'groups': np.array(all_groups)}

    def _prepare_raw_signal_sequences(self):
        """Chuẩn bị dữ liệu chuỗi tín hiệu thô (cho CNN-LSTM)."""
        self.logger.info("  -> Đang thực thi: Tín hiệu thô -> Trượt Cửa Sổ Chuỗi Thô")
        cfg = self.config.cnn_lstm
        offset_type = "O1" if self.config.data_prep.use_offset_data else "O2"
        filename = self.config.paths.wyoflex_emg_processed_template.format(forearm=self.forearm, offset_type=offset_type)
        signal_path = os.path.join(self.config.paths.processed_data_dir, filename)
        
        if not os.path.exists(signal_path):
            self.logger.error(f"File tín hiệu '{signal_path}' không tồn tại.")
            return None

        df = pd.read_csv(signal_path)
        df.dropna(inplace=True)
        
        emg_cols = [f'emg_{i+1}' for i in range(4)]
        X_sequences, y_sequences, groups_sequences = [], [], []

        for (p_id, m_id, c_id), group in tqdm(df.groupby(['participant', 'movement', 'cycle']), desc="Creating Raw Sequences"):
            trial_signal = group.iloc[4900:11100][emg_cols].values
            
            for i in range(0, len(trial_signal) - cfg.sequence_length + 1, cfg.sequence_step):
                X_sequences.append(trial_signal[i : i + cfg.sequence_length])
                y_sequences.append(m_id)
                groups_sequences.append(p_id)

        if not X_sequences:
            self.logger.error("Không tạo được mẫu huấn luyện nào.")
            return None

        X_final = np.array(X_sequences)
        self.logger.info(f"Hoàn tất. Đã tạo {len(X_final)} mẫu. Shape dữ liệu: {X_final.shape}")
        return {'X': X_final, 'y': np.array(y_sequences), 'groups': np.array(groups_sequences)}


    def _prepare_feature_sequences(self):
        """Chuẩn bị dữ liệu chuỗi vector đặc trưng (cho LSTM-on-Features)."""
        self.logger.info("  -> Đang thực thi: Tín hiệu thô -> Đặc trưng -> Chuỗi Đặc trưng")
        cfg_prep = self.config.data_prep
        offset_type = "O1" if cfg_prep.use_offset_data else "O2"
        
        feature_filename = self.config.paths.features_template.format(forearm=self.forearm, offset_type=offset_type)
        feature_path = os.path.join(self.config.paths.processed_data_dir, feature_filename)
        
        if not os.path.exists(feature_path):
            self.logger.error(f"File đặc trưng '{feature_path}' không tồn tại. Vui lòng chạy 'run_wyoflex_feature_extraction.py' trước.")
            return None

        self.logger.info(f"Đang tải các vector đặc trưng từ: {feature_path}")
        feature_df = pd.read_csv(feature_path)
        
        X_features = feature_df.drop(columns=['participant', 'label']).values
        y_labels = feature_df['label'].values
        groups = feature_df['participant'].values
        
        self.logger.info(f"Đã tải {len(X_features)} vector. Đang tạo chuỗi...")
        X_seq, y_seq, groups_seq = [], [], []
        SEQ_LENGTH = cfg_prep.sequence_length
        
        for i in range(len(X_features) - SEQ_LENGTH + 1):
            group_of_sequence = groups[i : i + SEQ_LENGTH]
            if len(set(group_of_sequence)) == 1:
                X_seq.append(X_features[i : i + SEQ_LENGTH])
                y_seq.append(y_labels[i + SEQ_LENGTH - 1])
                groups_seq.append(groups[i + SEQ_LENGTH - 1])

        X_final = np.array(X_seq, dtype=np.float32)
        self.logger.info(f"Hoàn tất. Đã tạo {len(X_final)} mẫu. Shape dữ liệu: {X_final.shape}")
        return {'X': X_final, 'y': np.array(y_seq), 'groups': np.array(groups_seq)}
        
    def run(self):
        """Chạy toàn bộ quy trình huấn luyện và đánh giá."""
        data = self.prepare_data()
        if data is None or data['X'].shape[0] == 0:
            self.logger.error("Chuẩn bị dữ liệu thất bại. Dừng quá trình.")
            return

        X, y, groups = data['X'], data['y'], data['groups']

        if self.config.grouping.enable_gesture_grouping:
            self.logger.info("Áp dụng tính năng gom nhóm cử chỉ...")
            y_df = pd.DataFrame(y, columns=['label'])
            y_df['grouped_label'] = y_df['label'].map(self.config.grouping.gesture_mapping)
            valid_indices = y_df['grouped_label'].notna()
            X, y, groups = X[valid_indices], y_df['grouped_label'][valid_indices].values, groups[valid_indices]

        le = LabelEncoder()
        y_encoded = le.fit_transform(y)

        gkf = GroupKFold(n_splits=5)
        fold_scores = []
        best_model_info = {'model': None, 'scaler': None, 'name': 'None', 'le': le, 'accuracy': -1.0}

        for fold, (train_idx, test_idx) in enumerate(gkf.split(X, y_encoded, groups)):
            self.logger.info(f"\n--- Đang xử lý Fold {fold + 1}/{gkf.get_n_splits()} ---")
            X_train_fold, X_test_fold = X[train_idx], X[test_idx]
            y_train_fold, y_test_fold = y_encoded[train_idx], y_encoded[test_idx]

            scaler = StandardScaler()
            original_shape = X_train_fold.shape
            X_train_scaled = scaler.fit_transform(X_train_fold.reshape(-1, X_train_fold.shape[-1])).reshape(original_shape)
            X_test_scaled = scaler.transform(X_test_fold.reshape(-1, X_test_fold.shape[-1])).reshape(X_test_fold.shape)

            nn_handler = EnhancedNeuralModel(self.config)
            nn_handler.label_encoder = le
            
            results = nn_handler.train_and_evaluate_all(X_train_scaled, y_train_fold, X_test_scaled, y_test_fold, self.data_format)
            
            if not results: continue

            best_fold_accuracy = -1.0
            best_model_name_in_fold = ""
            for model_name, result in results.items():
                if result['test_balanced_accuracy'] > best_fold_accuracy:
                    best_fold_accuracy = result['test_balanced_accuracy']
                    best_model_name_in_fold = model_name
            
            fold_scores.append(best_fold_accuracy)
            self.logger.info(f"Fold {fold+1} Best Balanced Accuracy: {best_fold_accuracy:.4f} (from model: {best_model_name_in_fold})")

            if best_fold_accuracy > best_model_info['accuracy']:
                best_model_info.update({
                    'model': nn_handler.trained_models[best_model_name_in_fold],
                    'scaler': scaler,
                    'name': best_model_name_in_fold,
                    'accuracy': best_fold_accuracy
                })
        
        if not fold_scores: return

        self.logger.info(f"\n{'='*80}\nKẾT QUẢ CUỐI CÙNG\n{'='*80}\nMean Balanced Accuracy: {np.mean(fold_scores):.4f} (+/- {np.std(fold_scores):.4f})\n{'='*80}")
        self._save_best_model(best_model_info)

    def _save_best_model(self, model_info):
        """Lưu model và các file phụ trợ."""
        model = model_info.get('model')
        if model is None:
            self.logger.warning("Không có model tốt nhất để lưu.")
            return

        output_dir = self.model_output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        name = model_info.get('name', 'model')
        suffix = "_grouped" if self.config.grouping.enable_gesture_grouping else ""
        base_filename = f"{name}_best{suffix}"
        
        model_path = os.path.join(output_dir, f"{base_filename}.keras")
        scaler_path = os.path.join(output_dir, f"{base_filename}_scaler.pkl")
        le_path = os.path.join(output_dir, f"{base_filename}_label_encoder.pkl")
        metadata_path = os.path.join(output_dir, f"{base_filename}_metadata.json")

        self.logger.info(f"Đang lưu model và các file phụ trợ vào: {output_dir}")
        model.save(model_path)
        joblib.dump(model_info.get('scaler'), scaler_path)
        joblib.dump(model_info.get('le'), le_path)
        
        metadata = {
            "model_name": name,
            "data_format": self.data_format,
            "forearm": self.forearm,
            "is_grouped": self.config.grouping.enable_gesture_grouping,
            "offset_type": "O1" if self.config.data_prep.use_offset_data else "O2"
        }
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=4)

        self.logger.info(f"✅ Đã lưu thành công model '{name}' và các file liên quan.")