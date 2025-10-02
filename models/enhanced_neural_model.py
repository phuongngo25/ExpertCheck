# enhanced_neural_model.py

import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras import layers, callbacks, regularizers
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, balanced_accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import class_weight
import logging
from pathlib import Path
import joblib

class PositionalEncoding(layers.Layer):
    def __init__(self, position, d_model):
        super(PositionalEncoding, self).__init__()
        self.pos_encoding = self.positional_encoding(position, d_model)
    def get_angles(self, position, i, d_model):
        angles = 1 / tf.pow(10000, (2 * (i // 2)) / tf.cast(d_model, tf.float32))
        return position * angles
    def positional_encoding(self, position, d_model):
        angle_rads = self.get_angles(
            tf.range(position, dtype=tf.float32)[:, tf.newaxis],
            tf.range(d_model, dtype=tf.float32)[tf.newaxis, :],
            d_model
        )
        sines = tf.math.sin(angle_rads[:, 0::2])
        cosines = tf.math.cos(angle_rads[:, 1::2])
        pos_encoding = tf.concat([sines, cosines], axis=-1)
        pos_encoding = pos_encoding[tf.newaxis, ...]
        return tf.cast(pos_encoding, tf.float32)
    def call(self, inputs):
        return inputs + self.pos_encoding[:, :tf.shape(inputs)[1], :]

class EnhancedNeuralModel:
    def __init__(self, config):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.label_encoder = LabelEncoder()
        self.trained_models = {}
        tf.random.set_seed(self.config.model.random_state)
        np.random.seed(self.config.model.random_state)
        self.model_architectures = self._initialize_models()

    def _initialize_models(self) -> dict:
        self.logger.info("Khởi tạo các kiến trúc mạng nơ-ron.")
        return {
            'mlp_model': self._create_residual_mlp_model,
            'cnn_2d_model': self._create_2d_cnn_model,
            'advanced_cnn': self._create_advanced_cnn_model,
        }

    # =============================================================================
    # CÁC HÀM TẠO KIẾN TRÚC MODEL
    # =============================================================================
    
    def _residual_block(self, x, units, dropout_rate=0.4, l2_reg=1e-5):
        # ... (giữ nguyên)
        pass

    def _create_residual_mlp_model(self, input_shape: tuple, num_classes: int, steps_per_epoch: int) -> keras.Model:
        # ... (giữ nguyên)
        pass

    def _create_2d_cnn_model(self, input_shape: tuple, num_classes: int, steps_per_epoch: int) -> keras.Model:
        # ... (giữ nguyên)
        pass
        
    def _create_advanced_cnn_model(self, input_shape: tuple, num_classes: int, steps_per_epoch: int) -> keras.Model:
        self.logger.info(f"Xây dựng kiến trúc CNN Đa nhánh + Attention với input shape: {input_shape}")
        inputs = keras.Input(shape=input_shape)
        
        # --- Nhánh 1: Kernel nhỏ ---
        branch_a = layers.Conv1D(filters=32, kernel_size=3, padding='same', activation='relu')(inputs)
        branch_a = layers.BatchNormalization()(branch_a)
        
        # --- Nhánh 2: Kernel vừa ---
        branch_b = layers.Conv1D(filters=32, kernel_size=9, padding='same', activation='relu')(inputs)
        branch_b = layers.BatchNormalization()(branch_b)
        
        # --- Nhánh 3: Kernel lớn ---
        branch_c = layers.Conv1D(filters=32, kernel_size=15, padding='same', activation='relu')(inputs)
        branch_c = layers.BatchNormalization()(branch_c)

        # Kết hợp các nhánh
        x = layers.concatenate([branch_a, branch_b, branch_c])
        x = layers.MaxPooling1D(pool_size=2)(x)
        x = layers.Dropout(0.4)(x)

        x = layers.Conv1D(filters=128, kernel_size=7, padding='same', kernel_regularizer=regularizers.l2(2e-4))(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)

        # --- Cơ chế Chú ý (Self-Attention) ---
        attention_output = layers.Attention()([x, x])
        
        x = layers.GlobalAveragePooling1D()(attention_output)
        x = layers.Dropout(0.5)(x)
        
        x = layers.Dense(128, activation='relu', kernel_regularizer=regularizers.l2(2e-4))(x)
        x = layers.Dropout(0.5)(x)
        
        outputs = layers.Dense(num_classes, activation='softmax')(x)
        model = keras.Model(inputs=inputs, outputs=outputs, name="Advanced_CNN_Model")
        
        lr_schedule = keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=1e-3, decay_steps=steps_per_epoch * 10, decay_rate=0.9)
        optimizer = keras.optimizers.Adam(learning_rate=lr_schedule)
        
        model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        model.summary(print_fn=self.logger.info)
        return model

    # =============================================================================
    # HÀM HUẤN LUYỆN VÀ ĐÁNH GIÁ (ĐÃ SỬA LỖI)
    # =============================================================================

    def train_and_evaluate_all(self, X_train, y_train, X_test, y_test):
        results = {}
        y_train_encoded = self.label_encoder.fit_transform(y_train)
        y_test_encoded = self.label_encoder.transform(y_test)

        # Dùng một phần nhỏ của tập train làm validation set nội bộ
        X_train_fit, X_val, y_train_fit, y_val = train_test_split(
            X_train, y_train_encoded,
            test_size=0.2, # 20% for validation
            random_state=self.config.model.random_state,
            stratify=y_train_encoded
        )
        
        steps_per_epoch = max(1, len(X_train_fit) // self.config.model.nn_config.batch_size)
            
        for model_name in self.config.model.neural_architectures:
            if model_name in self.model_architectures:
                self.logger.info(f"--- Đang huấn luyện mô hình {model_name.upper()} ---")
                create_fn = self.model_architectures[model_name]
                
                # =============================================================================
                # <<< SỬA LỖI: Logic xử lý input_shape cho từng loại model >>>
                # =============================================================================
                if model_name == 'cnn_2d_model':
                    self.logger.info("Chuẩn bị dữ liệu 4D cho 2D-CNN (trên đặc trưng)...")
                    if X_train_fit.ndim != 2:
                        raise ValueError(f"Model '{model_name}' yêu cầu dữ liệu 2D (đặc trưng), nhưng nhận được {X_train_fit.ndim}D.")
                    
                    num_channels = self.config.wyo_processing.max_channels
                    num_features = X_train_fit.shape[1] // num_channels
                    
                    X_train_iter = X_train_fit.reshape((-1, num_channels, num_features, 1))
                    X_val_iter = X_val.reshape((-1, num_channels, num_features, 1))
                    X_test_iter = X_test.reshape((-1, num_channels, num_features, 1))
                    input_shape_iter = (num_channels, num_features, 1)

                elif model_name == 'advanced_cnn':
                    self.logger.info("Sử dụng dữ liệu tín hiệu thô (3D) cho mô hình 1D-CNN nâng cao...")
                    if X_train_fit.ndim != 3:
                        raise ValueError(f"Model '{model_name}' yêu cầu dữ liệu 3D (tín hiệu thô), nhưng nhận được {X_train_fit.ndim}D.")
                    
                    X_train_iter, X_val_iter, X_test_iter = X_train_fit, X_val, X_test
                    input_shape_iter = (X_train_fit.shape[1], X_train_fit.shape[2]) # Shape là (timesteps, channels), vd: (256, 4)

                else: # Dành cho các mô hình 1D trên đặc trưng phẳng như MLP
                    self.logger.info("Sử dụng dữ liệu 2D (đặc trưng phẳng) cho mô hình 1D...")
                    if X_train_fit.ndim != 2:
                        raise ValueError(f"Model '{model_name}' yêu cầu dữ liệu 2D (đặc trưng phẳng), nhưng nhận được {X_train_fit.ndim}D.")
                    
                    X_train_iter, X_val_iter, X_test_iter = X_train_fit, X_val, X_test
                    input_shape_iter = (X_train_fit.shape[1],) # Shape là (số đặc trưng,), vd: (48,)

                results[model_name] = self._train_single(
                    create_fn, X_train_iter, y_train_fit, X_val_iter, y_val, X_test_iter, y_test_encoded, 
                    input_shape_iter, steps_per_epoch
                )
        return results

    def _train_single(self, create_fn, X_train, y_train, X_val, y_val, X_test, y_test, input_shape, steps_per_epoch):
        num_classes = len(self.label_encoder.classes_)
        model = create_fn(input_shape, num_classes, steps_per_epoch)
        
        class_weights = class_weight.compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
        class_weights_dict = dict(enumerate(class_weights))
        
        early_stopping = callbacks.EarlyStopping(monitor='val_loss', patience=self.config.model.nn_config.patience, restore_best_weights=True, verbose=1)
        
        model.fit(X_train, y_train,
                  epochs=self.config.model.nn_config.epochs,
                  batch_size=self.config.model.nn_config.batch_size,
                  validation_data=(X_val, y_val),
                  callbacks=[early_stopping],
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
        self.logger.info("Đang lưu các mô hình Neural Network...")
        nn_models_dir = Path(directory) / 'neural'
        nn_models_dir.mkdir(parents=True, exist_ok=True)
        for name, model in self.trained_models.items():
            try:
                model_name_to_save = name.replace(' ', '_')
                model_path = nn_models_dir / f"{model_name_to_save}.keras"
                model.save(model_path)
                self.logger.info(f"   -> Đã lưu mô hình '{name}' tại '{model_path}'")
            except Exception as e:
                self.logger.error(f"Lỗi khi lưu mô hình '{name}': {e}")
        try:
            encoder_path = nn_models_dir / "label_encoder.pkl"
            joblib.dump(self.label_encoder, encoder_path)
            self.logger.info(f"   -> Đã lưu Label Encoder tại '{encoder_path}'")
        except Exception as e:
            self.logger.error(f"Lỗi khi lưu Label Encoder: {e}")