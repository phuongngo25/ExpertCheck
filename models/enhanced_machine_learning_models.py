# src/models/enhanced_neural_model.py (PHIÊN BẢN HOÀN CHỈNH)

import numpy as np
import os
import tensorflow as tf
from tensorflow import keras
from keras import layers, callbacks, regularizers
from sklearn.model_selection import train_test_split, GroupKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, accuracy_score, balanced_accuracy_score
from sklearn.utils import class_weight
import logging
import joblib
from pathlib import Path
import optuna

class EnhancedNeuralModel:
    """
    Lớp quản lý việc xây dựng, huấn luyện và tối ưu hóa các mô hình Neural Network.
    Hoạt động linh hoạt dựa trên cấu hình được cung cấp.
    """
    def __init__(self, config):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.label_encoder = LabelEncoder()
        self.trained_models = {}
        tf.random.set_seed(config.model.random_state)
        np.random.seed(config.model.random_state)
        
        # Ánh xạ tên model từ config tới hàm tạo model tương ứng
        self.model_builders = {
            'advanced_cnn': self._create_advanced_cnn_model,
            'cnn_lstm': self._create_cnn_lstm_model,
            'lstm_on_features': self._create_lstm_on_features_model,
            'cnn_2d': self._create_2d_cnn_model,
        }

    # --- CÁC HÀM XÂY DỰNG KIẾN TRÚC ---
    
    def _create_advanced_cnn_model(self, input_shape, num_classes):
        """Kiến trúc CNN đa nhánh cho dữ liệu cửa sổ ngắn (256 mẫu)."""
        self.logger.info(f"--- Xây dựng mô hình Advanced CNN (trên cửa sổ ngắn) ---")
        inputs = keras.Input(shape=input_shape)
        
        branch_a = layers.Conv1D(filters=64, kernel_size=13, padding='same', activation='relu')(inputs)
        branch_a = layers.BatchNormalization()(branch_a)
        branch_b = layers.Conv1D(filters=64, kernel_size=9, padding='same', activation='relu')(inputs)
        branch_b = layers.BatchNormalization()(branch_b)
        branch_c = layers.Conv1D(filters=64, kernel_size=5, padding='same', activation='relu')(inputs)
        branch_c = layers.BatchNormalization()(branch_c)

        x = layers.concatenate([branch_a, branch_b, branch_c])
        x = layers.MaxPooling1D(pool_size=2)(x)
        x = layers.Dropout(0.6)(x)
        x = layers.Conv1D(filters=64, kernel_size=9, padding='same', activation='relu')(x)
        x = layers.BatchNormalization()(x)
        attention_output = layers.Attention()([x, x])
        x = layers.GlobalAveragePooling1D()(attention_output)
        x = layers.Dropout(0.6)(x)
        x = layers.Dense(64, activation='relu')(x)
        outputs = layers.Dense(num_classes, activation='softmax')(x)
        
        model = keras.Model(inputs=inputs, outputs=outputs, name="Advanced_CNN_Model")
        optimizer = keras.optimizers.Adam(learning_rate=0.00078)
        model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        model.summary(print_fn=self.logger.info)
        return model

    def _create_cnn_lstm_model(self, input_shape, num_classes):
        """
        NÂNG CẤP VỚI ATTENTION: Kiến trúc Hybrid CNN-LSTM-Attention linh hoạt.
        - Hoạt động với chuỗi tín hiệu thô hoặc chuỗi vector đặc trưng.
        """
        sequence_length = input_shape[0]
        num_features = input_shape[1]

        if num_features < 10: # Giả định là tín hiệu thô (ít kênh)
            self.logger.info(f"--- Xây dựng mô hình CNN-LSTM-Attention (trên TÍN HIỆU THÔ) ---")
        else: # Giả định là chuỗi đặc trưng (nhiều đặc trưng)
            self.logger.info(f"--- Xây dựng mô hình CNN-LSTM-Attention (trên CHUỖI ĐẶC TRƯNG) ---")

        cfg = self.config.cnn_lstm
        l2_rate = 1e-4
        
        inputs = keras.Input(shape=input_shape, name="Sequence_Input")
        x = layers.BatchNormalization()(inputs)
        
        cnn_kernel_size = 3 if sequence_length < 50 else cfg.cnn_kernel_size

        # Lớp CNN để trích xuất các mẫu con trong chuỗi
        x = layers.Conv1D(filters=64, kernel_size=cnn_kernel_size, padding='same', activation='relu')(x)
        x = layers.MaxPooling1D(pool_size=2, padding='same')(x)
        x = layers.Dropout(0.4)(x)

        x = layers.Conv1D(filters=128, kernel_size=cnn_kernel_size, padding='same', activation='relu')(x)
        x = layers.MaxPooling1D(pool_size=2, padding='same')(x)
        x = layers.Dropout(0.4)(x)
        
        # Lớp LSTM phải trả về chuỗi (return_sequences=True) để lớp Attention có thể hoạt động
        x = layers.LSTM(cfg.lstm_units, return_sequences=True, dropout=0.3, recurrent_dropout=0.3)(x)
        
        # === LỚP ATTENTION ĐƯỢC THÊM VÀO ĐÂY ===
        # Lớp này sẽ học cách gán "trọng số quan trọng" cho mỗi bước thời gian trong chuỗi.
        attention_output = layers.Attention()([x, x])
        
        # Gom thông tin từ chuỗi đã được "chú ý" lại
        x = layers.GlobalAveragePooling1D()(attention_output)
        
        # Lớp phân loại cuối cùng
        x = layers.Dense(64, activation='relu', kernel_regularizer=regularizers.l2(l2_rate))(x)
        x = layers.Dropout(0.5)(x)
        outputs = layers.Dense(num_classes, activation='softmax')(x)
        
        model = keras.Model(inputs=inputs, outputs=outputs, name="Hybrid_CNN_LSTM_Attention_Model")
        optimizer = keras.optimizers.Adam(learning_rate=3e-4)
        model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        model.summary(print_fn=self.logger.info)
        return model
        
    def _create_2d_cnn_model(self, input_shape, num_classes):
        """Xây dựng kiến trúc 2D CNN để học từ Spectrogram."""
        self.logger.info(f"--- Xây dựng mô hình 2D CNN cho Spectrogram ---")
        l2_rate = 1e-5

        inputs = keras.Input(shape=input_shape)
        x = layers.BatchNormalization()(inputs)
        x = layers.Conv2D(32, (3, 3), padding='same', activation='relu', kernel_regularizer=regularizers.l2(l2_rate))(x)
        x = layers.MaxPooling2D((2, 2))(x)
        x = layers.Conv2D(64, (3, 3), padding='same', activation='relu', kernel_regularizer=regularizers.l2(l2_rate))(x)
        x = layers.MaxPooling2D((2, 2))(x)
        x = layers.Dropout(0.4)(x)
        x = layers.Flatten()(x)
        x = layers.Dense(128, activation='relu')(x)
        x = layers.Dropout(0.5)(x)
        outputs = layers.Dense(num_classes, activation='softmax')(x)
        
        model = keras.Model(inputs=inputs, outputs=outputs, name="Spectrogram_CNN_2D_Model")
        optimizer = keras.optimizers.Adam(learning_rate=3e-4)
        model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        model.summary(print_fn=self.logger.info)
        return model

    def _create_lstm_on_features_model(self, input_shape, num_classes):
        """Kiến trúc LSTM cho chuỗi vector đặc trưng."""
        self.logger.info("--- Xây dựng mô hình LSTM trên Đặc Trưng ---")
        inputs = keras.Input(shape=input_shape, name="Feature_Sequence_Input")
        x = layers.BatchNormalization()(inputs)
        x = layers.LSTM(128, return_sequences=False, dropout=0.4, recurrent_dropout=0.4)(x)
        x = layers.Dense(64, activation='relu')(x)
        x = layers.Dropout(0.5)(x)
        outputs = layers.Dense(num_classes, activation='softmax')(x)
        
        model = keras.Model(inputs=inputs, outputs=outputs, name="LSTM_on_Features_Model")
        optimizer = keras.optimizers.Adam(learning_rate=5e-4)
        model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        model.summary(print_fn=self.logger.info)
        return model

    # --- CÁC HÀM TIỆN ÍCH VÀ ĐIỀU PHỐI ---

    def tune_hyperparameters(self, X, y, groups):
        """Chạy Optuna để tìm siêu tham số tốt nhất cho kiến trúc CNN-LSTM."""
        self.logger.info("--- BẮT ĐẦU TỐI ƯU HÓA SIÊU THAM SỐ CHO CNN-LSTM BẰNG OPTUNA ---")
        
        le = LabelEncoder()
        y_encoded = le.fit_transform(y)
        num_classes = len(le.classes_)
        
        def objective(trial):
            gkf = GroupKFold(n_splits=5)
            train_idx, val_idx = next(gkf.split(X, y_encoded, groups))
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y_encoded[train_idx], y_encoded[val_idx]

            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train.reshape(-1, X.shape[-1])).reshape(X_train.shape)
            X_val = scaler.transform(X_val.reshape(-1, X.shape[-1])).reshape(X_val.shape)
            
            params = {
                'cnn_filters_1': trial.suggest_categorical('cnn_filters_1', [16, 32, 64]),
                'cnn_filters_2': trial.suggest_categorical('cnn_filters_2', [32, 64, 128]),
                'cnn_kernel_size': trial.suggest_categorical('cnn_kernel_size', [7, 11, 15]),
                'lstm_units': trial.suggest_categorical('lstm_units', [32, 64, 128]),
                'dropout': trial.suggest_float('dropout', 0.3, 0.6),
                'learning_rate': trial.suggest_float('learning_rate', 1e-4, 5e-3, log=True)
            }
            
            inputs = keras.Input(shape=(X.shape[1], X.shape[2]))
            x = inputs
            x = layers.Conv1D(filters=params['cnn_filters_1'], kernel_size=params['cnn_kernel_size'], padding='same', activation='relu')(x)
            x = layers.BatchNormalization()(x)
            x = layers.MaxPooling1D(pool_size=2)(x)
            x = layers.Dropout(params['dropout'])(x)
            x = layers.Conv1D(filters=params['cnn_filters_2'], kernel_size=params['cnn_kernel_size'], padding='same', activation='relu')(x)
            x = layers.BatchNormalization()(x)
            x = layers.MaxPooling1D(pool_size=2)(x)
            x = layers.Dropout(params['dropout'])(x)
            x = layers.LSTM(params['lstm_units'], return_sequences=False, dropout=params['dropout'], recurrent_dropout=params['dropout'])(x)
            x = layers.Dense(64, activation='relu')(x)
            x = layers.Dropout(params['dropout'])(x)
            outputs = layers.Dense(num_classes, activation='softmax')(x)
            model = keras.Model(inputs=inputs, outputs=outputs)
            
            optimizer = keras.optimizers.Adam(learning_rate=params['learning_rate'])
            model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])

            early_stopping = keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=10, mode='max', restore_best_weights=True)
            history = model.fit(X_train, y_train, batch_size=32, epochs=30, validation_data=(X_val, y_val),
                                callbacks=[early_stopping], verbose=0)
            
            return max(history.history['val_accuracy'])

        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=self.config.tuning.n_trials)
        
        self.logger.info(f"\n{'='*80}\nKẾT QUẢ TỐI ƯU HÓA\n{'='*80}")
        self.logger.info(f"Độ chính xác tốt nhất (ước tính): {study.best_value:.4f}")
        self.logger.info("BỘ SIÊU THAM SỐ TỐT NHẤT:")
        for key, value in study.best_params.items():
            self.logger.info(f"    '{key}': {value},")
        self.logger.info("="*80)

    def train_and_evaluate_all(self, X_train_fold, y_train_fold, X_test_fold, y_test_fold, data_format):
        """Lặp qua các model trong config và huấn luyện chúng."""
        X_train, X_val, y_train, y_val = train_test_split(
            X_train_fold, y_train_fold, 
            test_size=0.2, 
            random_state=self.config.model.random_state, 
            stratify=y_train_fold
        )
        
        results = {}
        for model_name in self.config.model.neural_architectures:
            # Kiểm tra tính tương thích của dữ liệu với model
            # ... (Thêm lại logic kiểm tra data_format nếu cần)
            
            self.logger.info(f"\n>>> Đang huấn luyện model: {model_name.upper()} <<<")
            result = self._train_single(model_name, X_train, y_train, X_val, y_val, X_test_fold, y_test_fold)
            if result:
                results[model_name] = result
        
        return results

    def _train_single(self, model_name, X_train, y_train, X_val, y_val, X_test, y_test):
        """Hàm huấn luyện một mô hình duy nhất."""
        input_shape = X_train.shape[1:]
        num_classes = len(self.label_encoder.classes_)
        
        builder = self.model_builders.get(model_name)
        if builder is None:
            self.logger.error(f"Không tìm thấy hàm tạo cho model '{model_name}'.")
            return None
            
        model = builder(input_shape, num_classes)
        
        class_weights = class_weight.compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
        class_weights_dict = dict(enumerate(class_weights))
        
        reduce_lr = callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=1e-6, verbose=1)
        early_stopping = callbacks.EarlyStopping(monitor='val_loss', patience=self.config.model.nn_config.patience, restore_best_weights=True, verbose=1)
        
        model.fit(X_train, y_train,
                  epochs=self.config.model.nn_config.epochs,
                  batch_size=self.config.model.nn_config.batch_size,
                  validation_data=(X_val, y_val),
                  callbacks=[early_stopping, reduce_lr],
                  class_weight=class_weights_dict,
                  verbose=2)
        
        self.trained_models[model.name] = model
        test_pred_proba = model.predict(X_test)
        test_pred = np.argmax(test_pred_proba, axis=1)
        
        return {
            'test_accuracy': accuracy_score(y_test, test_pred),
            'test_balanced_accuracy': balanced_accuracy_score(y_test, test_pred),
            'classification_report': classification_report(y_test, test_pred, output_dict=True, zero_division=0, target_names=[str(c) for c in self.label_encoder.classes_])
        }

    def save_models(self, directory: str):
        """Lưu các mô hình đã huấn luyện và label encoder."""
        self.logger.info(f"Đang lưu các mô hình Neural Network vào: {directory}")
        os.makedirs(directory, exist_ok=True)
        
        for name, model in self.trained_models.items():
            try:
                model_name_to_save = name.replace(' ', '_')
                suffix = "_grouped" if self.config.grouping.enable_gesture_grouping else ""
                model_path = Path(directory) / f"{model_name_to_save}{suffix}.keras"
                model.save(model_path)
                self.logger.info(f"   -> Đã lưu mô hình '{name}' tại '{model_path}'")
            except Exception as e:
                self.logger.error(f"Lỗi khi lưu mô hình '{name}': {e}")
        
        try:
            encoder_path = Path(directory) / "label_encoder.pkl"
            if not os.path.exists(encoder_path):
                joblib.dump(self.label_encoder, encoder_path)
                self.logger.info(f"   -> Đã lưu Label Encoder tại '{encoder_path}'")
        except Exception as e:
            self.logger.error(f"Lỗi khi lưu Label Encoder: {e}")
