# src/config/enhanced_config.py

import os
import json
import yaml
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List
import logging

@dataclass
class PathsConfig:
    """Quản lý tập trung tất cả các đường dẫn trong dự án."""
    # Đường dẫn dữ liệu thô 
    raw_json_dir: str = os.path.join('data', 'Data1')
    raw_matlab_dir: str = os.path.join('data', 'Data2')
    raw_wyoflex_dir: str = os.path.join('data', 'WyoFlex_Dataset')
    
    # Đường dẫn dữ liệu đã xử lý 
    processed_data_dir: str = 'processed_data'
    json_emg_csv: str = 'json_emg_data.csv'
    matlab_emg_csv: str = 'matlab_emg_data_final.csv'
    
    wyoflex_emg_csv_right: str = 'wyoflex_emg_processed_right.csv'# tay phai 
    wyoflex_emg_csv_left: str = 'wyoflex_emg_processed_left.csv'# tay trai 

    
    # Output dir 
    output_dir: str = 'results'
    models_dir_original: str = 'models_original'
    models_dir_wyo_right: str = 'models_wyo_right'
    models_dir_wyo_left: str = 'models_wyo_left'

@dataclass
class OriginalProcessingConfig:
    """Cấu hình xử lý cho các bộ dữ liệu GỐC (JSON, MATLAB)."""
    segment_length: int = 256
    overlap_ratio: float = 0.5
    sampling_rate: int = 200
    max_channels: int = 6
    filter_order: int = 4
    highpass_freq: float = 20.0
    lowpass_freq: float = 95.0
    notch_freq: float = 50.0
    original_matlab_freq: int = 3000

@dataclass
class WyoFlexProcessingConfig:
    """Cấu hình xử lý RIÊNG cho bộ dữ liệu WyoFlex."""
    segment_length: int = 256 
    overlap_ratio: float = 0.5
    sampling_rate: int = 1000
    max_channels: int = 4
    filter_order: int = 4
    highpass_freq: float = 20.0
    lowpass_freq: float = 220.0
    notch_freq: float = 50.0

@dataclass
class OriginalFeatureConfig:
    """Cấu hình trích xuất đặc trưng cho dữ liệu gốc (200Hz)."""
    time_features: List[str] = field(default_factory=lambda: ['mav', 'rms', 'var', 'zc'])
    freq_bands: Dict[str, tuple] = field(default_factory=lambda: {
        'low': (20, 40), 'mid': (40, 60), 'high': (60, 95)
    })
    wavelet_name: str = 'db5'
    wavelet_levels: int = 4

@dataclass
class WyoFlexFeatureConfig:
    """Cấu hình trích xuất đặc trưng cho dữ liệu WyoFlex (1000Hz)."""
    time_features: List[str] = field(default_factory=lambda: ['mav', 'rms', 'var', 'zc'])
    freq_bands: Dict[str, tuple] = field(default_factory=lambda: {
        'low': (20, 80), 'mid': (80, 140), 'high': (140, 200)
    })
    wavelet_name: str = 'db5'
    wavelet_levels: int = 4

@dataclass
class NeuralNetworkConfig:
    """Cấu hình chi tiết cho các kiến trúc Neural Network."""
    epochs: int = 100
    batch_size: int = 256
    patience: int = 50

@dataclass
class GroupingConfig:
    """Cấu hình cho việc gom nhóm các cử chỉ tương tự."""
    enable_gesture_grouping: bool = True  # Bật/Tắt tính năng gom nhóm

    gesture_mapping: Dict[int, int] = field(default_factory=lambda: {
        # Nhóm 0: Nắm Mạnh (Power)
        5: 0, # Hook grip
        6: 0, # Power grip
        7: 0, # Spherical grip
        # Nhóm 1: Nắm Tinh Xảo (Precision)
        8: 1, # Precision grip
        9: 1, # Lateral grip
        10: 1, # Pinch grip
        # Nhóm 2: Gập/Duỗi Cổ Tay (Flex/Extend)
        1: 2, # Flexion
        2: 2, # Extension
        # Nhóm 3: Xoay Cổ Tay (Deviate)
        3: 3, # Ulnar deviation
        4: 3, # Radial deviation
    })
    
    new_class_names: Dict[int, str] = field(default_factory=lambda: {
        0: "Power_Grip_Group",
        1: "Precision_Grip_Group",
        2: "Flex_Extend_Group",
        3: "Deviation_Group",
    })

@dataclass
class XAIConfig:
    """Cấu hình cho việc chạy phân tích và giải thích mô hình (XAI)."""
    enable_xai_analysis: bool = False  # Bật/Tắt toàn bộ quá trình phân tích
    
    # model cần phân tích
    target_model_type: str = 'ML'      
    target_model_name: str = 'random_forest_model' # file model 
    
  
    n_background_samples: int = 100    
    n_explain_samples: int = 10        
    
    # lưu kết quả
    output_dir: str = 'xai_reports'
@dataclass
class ModelConfig:
    """Cấu hình huấn luyện mô hình tổng thể."""
    random_state: int = 42
    test_size: float = 0.2

    # Kích hoạt các nhóm mô hình để huấn luyện 
    enable_ml_models: bool = False         
    enable_neural_models: bool = True         
    # enable_dqn: bool = False 
    # enable_dt: bool = False +))))
    # enable_ppo: bool = False 

    # mo hinh ML 
    ml_models: List[str] = field(default_factory=lambda: [
        'lightgbm',                 
        'random_forest',
        'hist_gradient_boosting'
    ])
    
    # mo hinh NN 
    ml_models_with_class_weight: List[str] = field(default_factory=lambda: [
        'random_forest', 
        'hist_gradient_boosting'
    ])


    neural_architectures: List[str] = field(default_factory=lambda: [
        # 'mlp_model', 
        # 'transformer', 
        # 'conformer', 
        # 'cnn_2d_model',
        'advanced_cnn'
    ])
    nn_config: NeuralNetworkConfig = field(default_factory=NeuralNetworkConfig)



@dataclass
class OutputConfig:
    """Cấu hình đầu ra."""
    save_models: bool = True

class Config:
    def __init__(self, config_path: Optional[str] = None):
        self.paths = PathsConfig()
        self.wyo_processing = WyoFlexProcessingConfig()
        self.wyo_features = WyoFlexFeatureConfig()
        self.model = ModelConfig()
        self.output = OutputConfig()
        self.xai = XAIConfig()
        self.grouping = GroupingConfig() # gom nhóm
        
        self.original_processing = OriginalProcessingConfig()
        
        if config_path: 
            self.load_config(config_path)

    def load_config(self, config_path: str):
        config_path = Path(config_path)
        if not config_path.exists(): raise FileNotFoundError(f"Config file not found: {config_path}")
        if config_path.suffix.lower() == '.json':
            with open(config_path, 'r', encoding='utf-8') as f: config_data = json.load(f)
        elif config_path.suffix.lower() in ['.yml', '.yaml']:
            with open(config_path, 'r', encoding='utf-8') as f: config_data = yaml.safe_load(f)
        else: raise ValueError(f"Unsupported config format: {config_path.suffix}")
        self._update_config(config_data)

    def _update_config(self, config_data: Dict[str, Any]):
        for section, values in config_data.items():
            if hasattr(self, section) and isinstance(values, dict):
                section_obj = getattr(self, section)
                for key, value in values.items():
                    if hasattr(section_obj, key): setattr(section_obj, key, value)

    def save_config(self, config_path: str):
        config_data = {
            'paths': self.paths.__dict__,
            'wyo_processing': self.wyo_processing.__dict__,
            'wyo_features': self.wyo_features.__dict__,
            'model': self.model.__dict__,
            'output': self.output.__dict__,
            'xai': self.xai.__dict__
        }
        config_path = Path(config_path)
        config_path.parent.mkdir(parents=True, exist_ok=True)
        if config_path.suffix.lower() == '.json':
            with open(config_path, 'w', encoding='utf-8') as f: json.dump(config_data, f, indent=4, default=lambda o: o.__dict__)
        elif config_path.suffix.lower() in ['.yml', '.yaml']:
            with open(config_path, 'w', encoding='utf-8') as f: yaml.dump(config_data, f, default_flow_style=False)
        else: raise ValueError(f"Unsupported config format: {config_path.suffix}")

def get_config(config_path: Optional[str] = None) -> Config:
    return Config(config_path)

def get_motion_mappings() -> Dict[str, Dict[int, Any]]:
    unified_gestures = {
        0: 'Rest', 1: 'Hand_Open', 2: 'Hand_Close', 3: 'Wrist_Flexion',
        4: 'Wrist_Extension', 5: 'Pinch', 6: 'Supination', 7: 'Pronation'
    }
    raw_json_to_unified = {0: 0, 1: 2, 2: 3, 3: 4, 4: 1, 5: 5}
    raw_matlab_to_unified = {0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 6, 6: 7, 7: 0}
    return {
        "unified": unified_gestures,
        "json_map": raw_json_to_unified,
        "matlab_map": raw_matlab_to_unified
    }
def get_wyoflex_gestures() -> Dict[int, str]:
    """
    Trả về từ điển định nghĩa 10 cử chỉ gốc của bộ dữ liệu WyoFlex.
    Key là Movement ID (1-10) và value là tên cử chỉ.
    """
    return {
        1: 'Flexion',
        2: 'Extension',
        3: 'Ulnar deviation',
        4: 'Radial deviation',
        5: 'Hook grip',
        6: 'Power grip',
        7: 'Spherical grip',
        8: 'Precision grip',
        9: 'Lateral grip',
        10: 'Pinch grip'
    }
def setup_logging(level: str = 'INFO', log_file: Optional[str] = None):
    logging_config = {
        'level': getattr(logging, level.upper()),
        'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        'datefmt': '%Y-%m-%d %H:%M:%S'
    }
    if log_file:
        logging_config['filename'] = log_file
        logging_config['filemode'] = 'a'
    logging.basicConfig(**logging_config)