
# run_wyoflex_feature_extraction.py (NÂNG CẤP CUỐI CÙNG)

import os
import pandas as pd
import numpy as np
import logging
import time
import sys

# Đảm bảo có thể import từ src
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.append(project_root)

from src.config.enhanced_config import get_config
# === THAY ĐỔI 1: IMPORT BỘ XỬ LÝ DỮ LIỆU ===
from src.data.advanced_data_processor import AdvancedDataProcessor
from src.features.advanced_feature_extraction import AdvancedFeatureExtractor

def main():
    """
    Pipeline chuyên dụng để trích xuất đặc trưng thủ công.
    Tự động đọc cấu hình, áp dụng bộ lọc đồng bộ và trích xuất đặc trưng.
    """
    start_time = time.time()
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logger = logging.getLogger("WyoFlexFeatureExtractor")
    
    try:
        config = get_config()
        
        # --- ĐỌC CẤU HÌNH TỰ ĐỘNG (Giữ nguyên) ---
        target_forearm = config.project.target_forearm
        use_offset = config.data_prep.use_offset_data
        offset_type = "O1" if use_offset else "O2"
        offset_str = "có offset" if use_offset else "không có offset"
      
        logger.info(f">>> BẮT ĐẦU TRÍCH XUẤT ĐẶC TRƯNG CHO TAY: {target_forearm.upper()} | DỮ LIỆU: {offset_type} ({offset_str}) <<<")

        input_filename = config.paths.wyoflex_emg_processed_template.format(forearm=target_forearm, offset_type=offset_type)
        output_filename = config.paths.features_template.format(forearm=target_forearm, offset_type=offset_type)

        input_path = os.path.join(config.paths.processed_data_dir, input_filename)
        output_path = os.path.join(config.paths.processed_data_dir, output_filename)

        if not os.path.exists(input_path):
            logger.error(f"LỖI: Không tìm thấy file '{input_path}'. Vui lòng chạy 'process_wyodata.py' trước.")
            return

        # --- BƯỚC 1: Tải dữ liệu (Giữ nguyên) ---
        logger.info(f"--- Bước 1: Tải dữ liệu từ '{input_path}' ---")
        df = pd.read_csv(input_path)
        
        # === THAY ĐỔI 2: THÊM BƯỚC TIỀN XỬ LÝ VÀ LỌC TÍN HIỆU ===
        logger.info("--- Bước 2: Áp dụng Tiền xử lý (Lọc FIR-IIR, Chuẩn hóa Z-score)... ---")
        data_processor = AdvancedDataProcessor(config)
        processed_df = data_processor.process_dataframe(df)

        if processed_df.empty:
            logger.error("DataFrame rỗng sau khi tiền xử lý. Dừng chương trình.")
            return
        
        # === THAY ĐỔI 3: LOẠI BỎ CÁC BƯỚC XỬ LÝ THỦ CÔNG CŨ ===
        # Các bước làm sạch và chuẩn hóa cũ ("Đồng bộ hóa thang đo", "Làm sạch dữ liệu") 
        # đã được thực hiện trong AdvancedDataProcessor, vì vậy chúng ta không cần chúng ở đây nữa.

        # --- BƯỚC 3: Phân đoạn và trích xuất đặc trưng ---
        logger.info("--- Bước 3: Phân đoạn và trích xuất đặc trưng ---")
        feature_extractor = AdvancedFeatureExtractor(config)
        # Sử dụng dataframe đã được xử lý đúng cách
        feature_df = feature_extractor.segment_and_extract(processed_df)

        if feature_df.empty:
            logger.error("Trích xuất đặc trưng thất bại.")
            return

        # --- BƯỚC 4: Lưu kết quả (Giữ nguyên) ---
        feature_df.to_csv(output_path, index=False)
        logger.info(f" Hoàn tất. Đã lưu file đặc trưng tại: '{output_path}'")

    except Exception as e:
        logger.error(f"Đã xảy ra lỗi nghiêm trọng: {e}", exc_info=True)
    finally:
        logger.info(f">>> Pipeline kết thúc sau {time.time() - start_time:.2f} giây <<<")

if __name__ == '__main__':
    main()
