# process_wyodata.py (NÂNG CẤP)

import os
import pandas as pd
import numpy as np
import logging
from tqdm import tqdm
from collections import defaultdict
import re

# Đảm bảo có thể import từ src
import sys
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.append(project_root)

from src.config.enhanced_config import get_config

def process_wyoflex_data():
    """
    Hàm chính, tự động xử lý và tạo file riêng cho cả dữ liệu O1 và O2.
    """
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logger = logging.getLogger("WyoFlexProcessor")
    
    try:
        config = get_config()
        os.makedirs(config.paths.processed_data_dir, exist_ok=True)
        
        # === THAY ĐỔI: LẶP QUA CẢ O1 VÀ O2 ===
        for offset_type in ['O1', 'O2']:
            offset_str = "có offset" if offset_type == 'O1' else "không có offset"
            logger.info(f"\n>>> BẮT ĐẦU XỬ LÝ DỮ LIỆU {offset_type} ({offset_str}) <<<")

            df_right, df_left = _group_and_process_raw_files(config, offset_type)
            
            # --- Lưu file cho tay phải ---
            if df_right is not None and not df_right.empty:
                # Tạo tên file động
                filename = config.paths.wyoflex_emg_processed_template.format(forearm='right', offset_type=offset_type)
                output_path = os.path.join(config.paths.processed_data_dir, filename)
                df_right.to_csv(output_path, index=False)
                logger.info(f"Đã lưu dữ liệu TAY PHẢI ({offset_type}) tại: '{output_path}'")
            else:
                logger.warning(f"Không tìm thấy dữ liệu TAY PHẢI cho {offset_type}.")

            # --- Lưu file cho tay trái ---
            if df_left is not None and not df_left.empty:
                # Tạo tên file động
                filename = config.paths.wyoflex_emg_processed_template.format(forearm='left', offset_type=offset_type)
                output_path = os.path.join(config.paths.processed_data_dir, filename)
                df_left.to_csv(output_path, index=False)
                logger.info(f"Đã lưu dữ liệu TAY TRÁI ({offset_type}) tại: '{output_path}'")
            else:
                logger.warning(f"Không tìm thấy dữ liệu TAY TRÁI cho {offset_type}.")

        logger.info("\n>>> Xử lý toàn bộ dữ liệu thô hoàn tất <<<")

    except Exception as e:
        logger.error(f"Đã xảy ra lỗi nghiêm trọng: {e}", exc_info=True)

def _group_and_process_raw_files(config, offset_type_to_process):
    """Hàm này giờ sẽ nhận loại offset cần xử lý."""
    logger = logging.getLogger("WyoFlexProcessor")
    base_data_dir = config.paths.raw_wyoflex_dir
    voltage_dir = os.path.join(base_data_dir, 'VOLTAGE DATA')

    trial_files = defaultdict(list)
    logger.info(f"Đang quét và nhóm các file '{offset_type_to_process}'...")
    
    for root, _, files in os.walk(voltage_dir):
        for filename in files:
            # === LOGIC LỌC FILE ĐƠN GIẢN HƠN ===
            # Chỉ xử lý file có chứa đúng loại offset mong muốn
            if offset_type_to_process not in filename:
                continue
            
            if not filename.startswith('P'): continue
            
            try:
                trial_key = re.sub(r'S\d+', '', filename)
                trial_files[trial_key].append(os.path.join(root, filename))
            except (AttributeError, IndexError):
                logger.warning(f"Bỏ qua file không đúng định dạng: {filename}")

    logger.info(f"Đã nhóm thành công {len(trial_files)} trials.")
    
    all_right_dfs = []
    all_left_dfs = []
    
    # Cắt lấy đoạn tín hiệu hoạt động
    start_index = 4900
    end_index = 11100
    
    for key, files in tqdm(trial_files.items(), desc="Đang xử lý các trials"):
        if len(files) != 4:
            logger.warning(f"Bỏ qua trial '{key}' vì không đủ 4 sensor.")
            continue
        
        try:
            p_id = int(re.search(r'P(\d+)', key).group(1))
            c_id = int(re.search(r'C(\d+)', key).group(1))
            m_id = int(re.search(r'M(\d+)', key).group(1))
            f_id = int(re.search(r'F(\d+)', key).group(1)) # 1 = phải, 2 = trái
            label = m_id
            
            channels_data = []
            min_len = float('inf')
            for file_path in sorted(files): 
                channel_values = np.loadtxt(file_path, delimiter=',', dtype=np.float32)
                channels_data.append(channel_values)
                min_len = min(min_len, len(channel_values))
            
            emg_data = np.array([ch[:min_len] for ch in channels_data]).T
            
            if min_len >= end_index:
                segment_to_process = emg_data[start_index:end_index, :]
            else:
                logger.warning(f"Bỏ qua trial {key} vì tín hiệu quá ngắn ({min_len} mẫu).")
                continue
            
            df = pd.DataFrame(segment_to_process, columns=[f'emg_{i+1}' for i in range(4)])
            df['participant'] = p_id
            df['cycle'] = c_id
            df['movement'] = m_id
            df['label'] = label
            
            if f_id == 1:
                all_right_dfs.append(df)
            elif f_id == 2:
                all_left_dfs.append(df)

        except Exception as e:
            logger.error(f"Lỗi xử lý trial '{key}': {e}")
            continue

    df_right = pd.concat(all_right_dfs, ignore_index=True) if all_right_dfs else None
    df_left = pd.concat(all_left_dfs, ignore_index=True) if all_left_dfs else None
    
    return df_right, df_left

if __name__ == '__main__':
    process_wyoflex_data()
