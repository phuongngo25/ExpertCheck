
# XAI.py

import logging
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import LabelEncoder
from tensorflow import keras
import joblib
import shap
import matplotlib.pyplot as plt

from src.config.enhanced_config import get_config

def main():
    config = get_config()
    if not config.xai.enable_xai_analysis:
        print("Phân tích XAI đã bị tắt trong file config. Bỏ qua.")
        return

    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logger = logging.getLogger("XAI_Analyzer")
    
    TARGET_FOREARM = 'right' # Tạm thời hardcode, có thể đưa vào config nếu cần
    
    logger.info(f">>> BẮT ĐẦU PHÂN TÍCH XAI CHO TAY: {TARGET_FOREARM.upper()} <<<")
    logger.info(f"Model cần phân tích: {config.xai.target_model_name} (Type: {config.xai.target_model_type})")

    # --- 1. Tải model và scaler tương ứng ---
    model_dir = os.path.join('results', f'models_wyo_{TARGET_FOREARM}')
    model_type_upper = config.xai.target_model_type.upper()
    model_name = config.xai.target_model_name

    try:
        if model_type_upper == 'NN':
            model_path = os.path.join(model_dir, f"{model_name}.keras")
            scaler_path = os.path.join(model_dir, '1d_cnn_best_scaler.pkl')
            model = keras.models.load_model(model_path)
        elif model_type_upper == 'ML':
            model_path = os.path.join(model_dir, 'ML', f"{model_name}.pkl")
            scaler_path = os.path.join(model_dir, 'final_scaler.pkl')
            model = joblib.load(model_path)
        else:
            raise ValueError("xai_model_type trong config phải là 'ML' hoặc 'NN'")
        
        scaler = joblib.load(scaler_path)
        logger.info(f"Đã tải model từ: {model_path}")
        logger.info(f"Đã tải scaler từ: {scaler_path}")

    except FileNotFoundError as e:
        logger.error(f"Lỗi không tìm thấy file: {e}. Hãy chắc chắn bạn đã chạy pipeline huấn luyện tương ứng.")
        return

    # --- 2. Xử lý dựa trên loại model ---
    if model_type_upper == 'NN':
        # --- Logic cho NN: Phân tích trên tín hiệu thô ---
        signal_file = f'wyoflex_emg_processed_{TARGET_FOREARM}.csv'
        signal_path = os.path.join('processed_data', signal_file)
        df = pd.read_csv(signal_path)
        df.dropna(inplace=True)

        WINDOW_SIZE = model.input_shape[1]
        NUM_CHANNELS = model.input_shape[2]
        
        X, y = [], []
        emg_cols = [f'emg_{i+1}' for i in range(NUM_CHANNELS)]
        emg_data = df[emg_cols].values
        for i in range(0, len(emg_data) - WINDOW_SIZE + 1, WINDOW_SIZE // 4):
            X.append(emg_data[i : i + WINDOW_SIZE])
        
        X = np.array(X)
        le = LabelEncoder().fit(df['label'].unique())
        class_names = [str(int(c)) for c in le.classes_]

        logger.info("Chuẩn bị dữ liệu nền và dữ liệu cần giải thích cho NN...")
        background_indices = np.random.choice(X.shape[0], config.xai.n_background_samples, replace=False)
        background_data = X[background_indices]
        background_data_scaled = scaler.transform(background_data.reshape(-1, NUM_CHANNELS)).reshape(background_data.shape)
        
        explain_indices = np.random.choice(X.shape[0], config.xai.n_explain_samples, replace=False)
        explain_data = X[explain_indices]
        explain_data_scaled = scaler.transform(explain_data.reshape(-1, NUM_CHANNELS)).reshape(explain_data.shape)

        logger.info("Khởi tạo SHAP GradientExplainer...")
        explainer = shap.GradientExplainer(model, background_data_scaled)
        
        logger.info(f"Đang tính toán giá trị SHAP cho {config.xai.n_explain_samples} mẫu...")
        shap_values = explainer.shap_values(explain_data_scaled)
        
        output_dir = os.path.join('results', config.xai.output_dir, 'NN')
        os.makedirs(output_dir, exist_ok=True)
        
        logger.info(f"Đang vẽ và lưu biểu đồ vào thư mục '{output_dir}'...")
        shap_values_reshaped = np.transpose(np.array(shap_values), (1, 2, 3, 0))
        samples_reshaped = np.transpose(explain_data_scaled, (0, 2, 1))[:, :, :, np.newaxis]
        
        shap.image_plot(shap_values_reshaped, -samples_reshaped, labels=np.array([f"Class {c}" for c in class_names]), show=False)
        
        save_path = os.path.join(output_dir, f'{model_name}_shap_summary.png')
        plt.savefig(save_path)
        plt.close()
        logger.info(f"Đã lưu biểu đồ giải thích cho NN tại: {save_path}")

    elif model_type_upper == 'ML':
        # --- Logic cho ML: Phân tích trên bộ đặc trưng ---
        feature_file = f'features_{TARGET_FOREARM}_with_participant.csv'
        feature_path = os.path.join('processed_data', feature_file)
        df = pd.read_csv(feature_path)
        df.dropna(inplace=True)

        feature_cols = [col for col in df.columns if col not in ['participant', 'label']]
        X = df[feature_cols]
        
        n_samples_for_summary = 500
        if len(X) > n_samples_for_summary:
            logger.info(f"Lấy {n_samples_for_summary} mẫu ngẫu nhiên từ dữ liệu để phân tích...")
            X_subset = X.sample(n=n_samples_for_summary, random_state=42)
        else:
            X_subset = X

        X_subset_np = X_subset.values
        X_subset_scaled_np = scaler.transform(X_subset_np)
        X_subset_scaled_df = pd.DataFrame(X_subset_scaled_np, columns=feature_cols)

        logger.info("Khởi tạo SHAP TreeExplainer...")
        explainer = shap.TreeExplainer(model)
        
        logger.info("Đang tính toán giá trị SHAP trên tập con...")
        shap_values = explainer.shap_values(X_subset_scaled_df)
        
        output_dir = os.path.join('results', config.xai.output_dir, 'ML')
        os.makedirs(output_dir, exist_ok=True)
        
        logger.info("Đang vẽ và lưu biểu đồ...")
        plt.figure()
        shap.summary_plot(shap_values, X_subset_scaled_df, plot_type="bar", show=False, class_names=model.classes_)
        plt.title(f'SHAP Feature Importance for {model_name}')
        plt.tight_layout()
        save_path = os.path.join(output_dir, f'{model_name}_shap_summary.png')
        plt.savefig(save_path)
        plt.close()
        logger.info(f"Đã lưu biểu đồ giải thích cho ML tại: {save_path}")

if __name__ == '__main__':
    main()