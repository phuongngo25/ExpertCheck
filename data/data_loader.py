# src/data/data_loader.py

import os
import json
import numpy as np
import pandas as pd
import scipy.io
import h5py
from scipy import signal
from tqdm import tqdm
import logging
from typing import Tuple, Optional, List, Dict

class EnhancedDataLoader:
    """
    Lớp chuyên trách cho việc tải và xử lý dữ liệu thô (.json, .mat)
    để tạo ra các file CSV sạch, sẵn sàng cho bước xử lý tiếp theo.
    """
    def __init__(self, config):
        self.config = config
        self.logger = logging.getLogger(__name__)

    def process_raw_data_to_csv(self):
        """
        Hàm chính điều phối toàn bộ quá trình xử lý dữ liệu thô.
        Nó sẽ kiểm tra và chạy xử lý cho cả hai nguồn dữ liệu JSON và MATLAB.
        """
        os.makedirs(self.config.paths.processed_data_dir, exist_ok=True)
        self._process_json_files()
        self._process_matlab_files()

    def _process_json_files(self):
        """Xử lý tất cả các file trong thư mục JSON."""
        cfg_paths = self.config.paths
        cfg_proc = self.config.processing
        
        output_path = os.path.join(cfg_paths.processed_data_dir, cfg_paths.json_emg_csv)
        
        # Nếu không ép buộc xử lý lại và file đã tồn tại, thì bỏ qua
        if not cfg_proc.force_reprocess_raw_data and os.path.exists(output_path):
            self.logger.info(f"File '{output_path}' đã tồn tại, bỏ qua xử lý JSON.")
            return

        self.logger.info(f"--- Bắt đầu xử lý thư mục JSON: {cfg_paths.raw_json_dir} ---")
        all_dfs = []
        json_files = [os.path.join(root, f) for root, _, files in os.walk(cfg_paths.raw_json_dir) for f in files if f.endswith('.json') and 'responses' not in f]
        
        for file_path in tqdm(json_files, desc="Đang xử lý file JSON"):
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = json.load(f)
                df = self._extract_full_data_from_json(content, cfg_proc.max_channels)
                if df is not None:
                    all_dfs.append(df)
            except Exception as e:
                self.logger.warning(f"Bỏ qua file JSON bị lỗi '{file_path}': {e}")
        
        if not all_dfs:
            self.logger.warning("Không tìm thấy dữ liệu JSON hợp lệ để xử lý.")
            return

        json_full_df = pd.concat(all_dfs, ignore_index=True)
        
        # Làm sạch dữ liệu
        rows_before = len(json_full_df)
        json_full_df = json_full_df[json_full_df['label'] != 65535]
        self.logger.info(f"Đã lọc bỏ {rows_before - len(json_full_df)} dòng có label 65535.")
        
        # Lưu file EMG
        emg_cols = ['sample_index', 'label'] + [f'emg_{i+1}' for i in range(cfg_proc.max_channels)]
        json_full_df[emg_cols].to_csv(output_path, index=False)
        self.logger.info(f"✅ Đã lưu file EMG từ JSON vào: {output_path}")
        
        # Lưu file IMU (tùy chọn)
        imu_output_path = os.path.join(cfg_paths.processed_data_dir, cfg_paths.json_imu_csv)
        imu_cols = [col for col in json_full_df.columns if col not in emg_cols or col in ['sample_index', 'label']]
        json_full_df[imu_cols].to_csv(imu_output_path, index=False)
        self.logger.info(f"✅ Đã lưu file IMU từ JSON vào: {imu_output_path}")

    def _process_matlab_files(self):
        """Xử lý tất cả các file trong thư mục MATLAB."""
        cfg_paths = self.config.paths
        cfg_proc = self.config.processing
        
        output_path = os.path.join(cfg_paths.processed_data_dir, cfg_paths.matlab_emg_csv)

        if not cfg_proc.force_reprocess_raw_data and os.path.exists(output_path):
            self.logger.info(f"File '{output_path}' đã tồn tại, bỏ qua xử lý MATLAB.")
            return
            
        self.logger.info(f"--- Bắt đầu xử lý thư mục MATLAB: {cfg_paths.raw_matlab_dir} ---")
        mat_emg_data, mat_labels, mat_tasks = [], [], []
        for root, _, files in os.walk(cfg_paths.raw_matlab_dir):
            for f in files:
                if f.endswith('d.mat'):
                    emg_path = os.path.join(root, f)
                    label_path = os.path.join(root, f.replace('d.mat', 'i.mat'))
                    if os.path.exists(label_path):
                        mat_tasks.append({'emg': emg_path, 'label': label_path})
        
        q = int(cfg_proc.original_matlab_freq / cfg_proc.sampling_rate)
        self.logger.info(f"Hệ số downsample cho dữ liệu MATLAB: {q}")

        for task in tqdm(mat_tasks, desc="Đang xử lý file MATLAB"):
            extracted = self._process_matlab_pair(task['emg'], task['label'], q, cfg_proc.max_channels)
            if extracted:
                mat_emg_data.append(extracted[0])
                mat_labels.append(extracted[1])
        
        if not mat_emg_data:
            self.logger.warning("Không tìm thấy dữ liệu MATLAB hợp lệ để xử lý.")
            return

        mat_emg_df = pd.DataFrame(np.vstack(mat_emg_data), columns=[f'emg_{i+1}' for i in range(cfg_proc.max_channels)])
        mat_emg_df['label'] = np.hstack(mat_labels)
        
        # Làm sạch dữ liệu
        rows_before = len(mat_emg_df)
        mat_emg_df = mat_emg_df[mat_emg_df['label'] != 7]
        self.logger.info(f"Đã lọc bỏ {rows_before - len(mat_emg_df)} dòng có label 7 từ dữ liệu MATLAB.")
        
        mat_emg_df['label'] = mat_emg_df['label'].replace(-1, 0)
        
        mat_emg_df.to_csv(output_path, index=False)
        self.logger.info(f"✅ Đã lưu file EMG từ MATLAB vào: {output_path}")

    def _extract_full_data_from_json(self, json_data: dict, num_channels: int) -> Optional[pd.DataFrame]:
        all_dfs: List[pd.DataFrame] = []
        try:
            training_samples = json_data.get('trainingSamples')
            if not training_samples or not isinstance(training_samples, dict): return None
            for sample_key, sample_content in training_samples.items():
                if not ('myoDetection' in sample_content and 'emg' in sample_content and
                        'ch1' in sample_content['emg'] and sample_content['emg']['ch1']): continue
                
                table_data = {}
                emg = sample_content.get('emg', {})
                accel = sample_content.get('accelerometer', {})
                gyro = sample_content.get('gyroscope', {})
                quat = sample_content.get('quaternion', {})

                for i in range(num_channels):
                    table_data[f'emg_{i+1}'] = emg.get(f'ch{i+1}', [])
                
                table_data.update({
                    'acc_x': accel.get('x', []), 'acc_y': accel.get('y', []), 'acc_z': accel.get('z', []),
                    'gyro_x': gyro.get('x', []), 'gyro_y': gyro.get('y', []), 'gyro_z': gyro.get('z', []),
                    'quat_w': quat.get('w', []), 'quat_x': quat.get('x', []), 'quat_y': quat.get('y', []), 'quat_z': quat.get('z', []),
                    'label': sample_content.get('myoDetection', [])
                })

                min_len = min((len(v) for v in table_data.values() if isinstance(v, list) and v), default=0)
                if min_len == 0: continue
                
                for key in table_data: table_data[key] = table_data[key][:min_len]
                
                df = pd.DataFrame(table_data)
                df['sample_index'] = sample_key
                all_dfs.append(df)
            
            if not all_dfs: return None
            return pd.concat(all_dfs, ignore_index=True)
        except Exception as e:
            self.logger.error(f"Lỗi khi trích xuất JSON: {e}")
            return None
    
    def _robust_load_mat(self, file_path: str) -> Optional[dict]:
        try:
            return scipy.io.loadmat(file_path)
        except NotImplementedError:
            with h5py.File(file_path, 'r') as f:
                return {k: v[()] for k, v in f.items()}
        except Exception:
            return None

    def _process_matlab_pair(self, emg_path: str, label_path: str, q: int, num_channels: int) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        try:
            emg_mat, label_mat = self._robust_load_mat(emg_path), self._robust_load_mat(label_path)
            if not emg_mat or not label_mat: return None
            
            emg, motion_labels_raw, start_indices_raw = emg_mat.get('x'), label_mat.get('motion'), label_mat.get('data_indx')
            if emg is None or motion_labels_raw is None or start_indices_raw is None: return None
            
            if emg.shape[0] < emg.shape[1]: emg = emg.T
            if emg.shape[1] < num_channels: return None
            emg = emg[:, :num_channels]
            
            motion_labels, start_indices = motion_labels_raw.flatten(), start_indices_raw.flatten()
            if len(start_indices) > len(motion_labels): start_indices = start_indices[1:-1]
            
            min_len = min(len(start_indices), len(motion_labels))
            start_indices, motion_labels = start_indices[:min_len], motion_labels[:min_len]
            
            full_labels = np.full(emg.shape[0], -1, dtype=int)
            for i in range(len(start_indices) - 1):
                start_idx, end_idx = start_indices[i] - 1, start_indices[i+1] - 1
                if start_idx < end_idx: full_labels[start_idx:end_idx] = motion_labels[i]
            
            if len(start_indices) > 0: full_labels[start_indices[-1] - 1:] = motion_labels[-1]
            
            resampled_emg = signal.decimate(emg, q, axis=0, zero_phase=True)
            resampled_labels = full_labels[::q][:resampled_emg.shape[0]]
            return resampled_emg, resampled_labels
        except Exception:
            return None
