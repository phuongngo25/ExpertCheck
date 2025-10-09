# src/features/advanced_feature_extraction.py

import numpy as np
import pandas as pd
from scipy.signal import welch, get_window
import pywt
from sklearn.preprocessing import StandardScaler
import logging
from typing import List

class AdvancedFeatureExtractor:
    def __init__(self, config):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.proc_config = config.wyo_processing
        self.feat_config = config.wyo_features

    def _moving_average(self, data, window_size=3):
        """Hàm tính trung bình trượt để làm mịn tín hiệu trước khi tính ZC."""
        return np.convolve(data, np.ones(window_size), 'valid') / window_size

    def segment_and_extract(self, df: pd.DataFrame) -> pd.DataFrame:
        self.logger.info("Bắt đầu phân đoạn và trích xuất đặc trưng...")
        window_size = self.proc_config.segment_length
        step_size = int(window_size * (1 - self.proc_config.overlap_ratio))
        
        all_feature_rows = []

        for p_id, p_group in df.groupby('participant'):
            for label, label_group in p_group.groupby('label'):
                emg_cols = [col for col in label_group.columns if 'emg' in col]
                emg_data = label_group[emg_cols].values
                
                for i in range(0, emg_data.shape[0] - window_size + 1, step_size):
                    window = emg_data[i : i + window_size]
                    
                    if window.shape[0] == window_size:
                        features = self._extract_features_from_window(window)
                        feature_row = features + [int(p_id), int(label)]
                        all_feature_rows.append(feature_row)

        if not all_feature_rows:
            self.logger.warning("Không trích xuất được vector đặc trưng nào.")
            return pd.DataFrame()

        feature_names = self._get_feature_column_names()
        final_columns = feature_names + ['participant', 'label']

        feature_df = pd.DataFrame(all_feature_rows, columns=final_columns)
        self.logger.info(f"Trích xuất thành công {len(feature_df)} bộ đặc trưng.")
        return feature_df
        
    def _get_feature_column_names(self) -> List[str]:
        final_column_names = []
        num_channels = self.proc_config.max_channels
        
        for ch in range(1, num_channels + 1):
            for name in self.feat_config.time_features: final_column_names.append(f"{name.upper()}_EMG_{ch}")
            for band in self.feat_config.freq_bands.keys(): final_column_names.append(f"PSD_{band}_EMG_{ch}")
            levels = self.feat_config.wavelet_levels
            wavelet_names = [f'WAV_Energy_A{levels}'] + [f'WAV_Energy_D{i}' for i in range(levels, 0, -1)]
            for name in wavelet_names: final_column_names.append(f"{name}_EMG_{ch}")
        return final_column_names

    def _extract_features_from_window(self, window: np.ndarray) -> List[float]:
        features = []
        for ch in range(window.shape[1]):
            signal_ch = window[:, ch]
            length = len(signal_ch)
            
            for feature_name in self.feat_config.time_features:
                if feature_name == 'mav': features.append(np.mean(np.abs(signal_ch)))
                elif feature_name == 'rms': features.append(np.sqrt(np.mean(np.square(signal_ch))))
                elif feature_name == 'var': features.append(np.var(signal_ch, ddof=0))
                elif feature_name == 'zc':
                    zc_threshold = 0.001 
                    # Tính ZC trực tiếp trên tín hiệu đã lọc (giống hệt logic C)
                    zc_count = np.sum(
                        ( (signal_ch[:-1] * signal_ch[1:]) < 0 ) & 
                        ( np.abs(signal_ch[:-1] - signal_ch[1:]) > zc_threshold )
                    )
                    features.append(zc_count)

            nperseg = 256
            win = get_window('hann', nperseg)
            freqs, psd = welch(signal_ch, fs=self.proc_config.sampling_rate, window=win, nperseg=nperseg, scaling='spectrum')
            
            for band_name, (f_min, f_max) in self.feat_config.freq_bands.items():
                mask = (freqs >= f_min) & (freqs <= f_max)
                if np.any(mask):
                    df_freq = freqs[1] - freqs[0]
                    band_power = np.sum(psd[mask]) * df_freq
                    features.append(band_power)
                else: features.append(0.0)

            try:
                if length % 16 != 0:
                    pad_len = 16 - (length % 16)
                    signal_padded = np.pad(signal_ch, (0, pad_len), 'constant', constant_values=0)
                else: signal_padded = signal_ch
                coeffs = pywt.wavedec(signal_padded, self.feat_config.wavelet_name, level=self.feat_config.wavelet_levels)
                for c in coeffs:
                    energy = np.sum(np.square(c)) if len(c) > 0 else 0.0
                    features.append(energy)
            except ValueError:
                self.logger.warning(f"Lỗi phân rã Wavelet cho kênh {ch}. Gán giá trị 0.")
                for _ in range(self.feat_config.wavelet_levels + 1): features.append(0.0)
        return features
