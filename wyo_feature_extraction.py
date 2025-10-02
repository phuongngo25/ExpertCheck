# run_wyoflex_feature_extraction.py 

import os
import pandas as pd
import numpy as np
import logging
import time

import sys
project_root = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
if project_root not in sys.path:
    sys.path.append(project_root)

from src.config.enhanced_config import get_config
from src.features.advanced_feature_extraction import AdvancedFeatureExtractor

# =============================================================================
# Chọn tay 
TARGET_FOREARM = 'right'
# =============================================================================

def main():
    start_time = time.time()
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logger = logging.getLogger("WyoFlexFeatureExtractor")
      
    logger.info(f">>> BẮT ĐẦU PIPELINE TRÍCH XUẤT ĐẶC TRƯNG (ĐỒNG BỘ C) CHO TAY: {TARGET_FOREARM.upper()} <<<")

    try:
        config = get_config()
        
        # 
        if TARGET_FOREARM == 'right':
            input_csv = config.paths.wyoflex_emg_csv_right
            output_file = 'features_right_with_participant.csv'
        elif TARGET_FOREARM == 'left':
            input_csv = config.paths.wyoflex_emg_csv_left
            output_file = 'features_left_with_participant.csv'
        else:
            raise ValueError("TARGET_FOREARM phải là 'right' hoặc 'left'")

        input_csv_path = os.path.join(config.paths.processed_data_dir, input_csv)
        output_features_path = os.path.join(config.paths.processed_data_dir, output_file)

        if not os.path.exists(input_csv_path):
            logger.error(f"LỖI: Không tìm thấy file '{input_csv_path}'. Vui lòng chạy 'process_wyodata.py' trước.")
            return

        # --- BƯỚC 1: Tải dữ liệu ---
        df = pd.read_csv(input_csv_path)
        emg_cols = [f'emg_{i+1}' for i in range(config.wyo_processing.max_channels)]

        # --- BƯỚC 2: Đồng bộ hóa thang đo tín hiệu về [-1.65V, +1.65V] ---
        logger.info("--- Bước 2: Đồng bộ hóa thang đo tín hiệu về khoảng [-1.65, 1.65] V ---")
        global_max_abs_val = df[emg_cols].abs().max().max()
        if global_max_abs_val > 0:
            df[emg_cols] = (df[emg_cols] / global_max_abs_val) * 1.65
            logger.info(f"   Đã áp dụng scaling (dựa trên max_abs={global_max_abs_val:.4f}).")
        
        # --- BƯỚC 3: Làm sạch dữ liệu cơ bản ---
        logger.info("--- Bước 3: Làm sạch các giá trị không hợp lệ (NaN, Inf) ---")
        initial_rows = len(df)
        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        df.dropna(inplace=True)
        rows_removed = initial_rows - len(df)
        logger.info(f"   Đã loại bỏ {rows_removed} dòng.")

        if df.empty:
            logger.error("DataFrame rỗng sau khi làm sạch. Dừng chương trình.")
            return

        # --- BƯỚC 4: Phân đoạn và trích xuất đặc trưng (KHÔNG QUA BỘ LỌC FIR) ---
        logger.info("--- Bước 4: Phân đoạn và trích xuất đặc trưng ---")
        feature_extractor = AdvancedFeatureExtractor(config)
        feature_df = feature_extractor.segment_and_extract(df)

        if feature_df.empty:
            logger.error("Trích xuất đặc trưng thất bại.")
            return

        feature_df.to_csv(output_features_path, index=False)
        logger.info(f"Done, find at: '{output_features_path}'")

    except Exception as e:
        logger.error(f"Error: {e}", exc_info=True)
    finally:
        logger.info(f"End {time.time() - start_time:.2f} <<<")

if __name__ == '__main__':
    main()