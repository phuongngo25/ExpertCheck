import numpy as np
import json
import scipy.io
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import pandas as pd
from scipy import signal
from scipy.stats import kurtosis, skew
from typing import Dict, List, Tuple, Any, Optional
import logging

class DataExplorer:
    """Comprehensive EMG data exploration and visualization"""
    
    def __init__(self,config):
        self.logger = logging.getLogger(__name__)
        plt.style.use('default')
        sns.set_palette("husl")
    
    def analyze_emg_data(self, data: np.ndarray, labels: np.ndarray, 
                        output_dir: str = "results") -> Dict[str, Any]:
        """Comprehensive analysis of EMG data"""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        analysis_results = {}
        
        try:
            # Basic statistics
            analysis_results['basic_stats'] = self._compute_basic_statistics(data, labels)
            
            # Signal quality analysis
            analysis_results['signal_quality'] = self._analyze_signal_quality(data)
            
            # Class distribution analysis
            analysis_results['class_analysis'] = self._analyze_class_distribution(labels)
            
            # Generate visualizations
            self._create_visualizations(data, labels, output_path)
            
            # Save analysis results
            self._save_analysis_results(analysis_results, output_path / "data_analysis.json")
            
            self.logger.info(f"Data exploration completed. Results saved to {output_dir}")
            
        except Exception as e:
            self.logger.error(f"Data exploration failed: {e}")
            analysis_results['error'] = str(e)
        
        return analysis_results
    
    def _compute_basic_statistics(self, data: np.ndarray, labels: np.ndarray) -> Dict[str, Any]:
        """Compute basic statistical measures"""
        stats = {
            'n_samples': len(data),
            'data_shape': list(data.shape),
            'n_classes': len(np.unique(labels)),
            'unique_labels': np.unique(labels).tolist(),
            'data_range': [float(np.min(data)), float(np.max(data))],
            'data_mean': float(np.mean(data)),
            'data_std': float(np.std(data)),
            'data_median': float(np.median(data)),
            'data_skewness': float(skew(data.flatten())),
            'data_kurtosis': float(kurtosis(data.flatten()))
        }
        
        # Per-class statistics
        class_stats = {}
        for label in np.unique(labels):
            class_data = data[labels == label]
            class_stats[str(label)] = {
                'n_samples': len(class_data),
                'mean': float(np.mean(class_data)),
                'std': float(np.std(class_data)),
                'median': float(np.median(class_data))
            }
        
        stats['per_class_stats'] = class_stats
        return stats
    
    def _analyze_signal_quality(self, data: np.ndarray) -> Dict[str, Any]:
        """Analyze signal quality metrics"""
        quality_metrics = {}
        
        try:
            # Signal-to-noise ratio estimation
            if data.ndim >= 2:
                # Use first channel for analysis
                signal_data = data[0] if data.ndim == 3 else data[0, :]
            else:
                signal_data = data
            
            # Estimate SNR using signal power vs noise power
            signal_power = np.mean(signal_data**2)
            noise_estimate = np.std(np.diff(signal_data))  # High-frequency noise estimate
            snr_estimate = 10 * np.log10(signal_power / (noise_estimate**2 + 1e-10))
            
            quality_metrics['estimated_snr_db'] = float(snr_estimate)
            quality_metrics['signal_power'] = float(signal_power)
            quality_metrics['noise_estimate'] = float(noise_estimate)
            
            # Zero-crossing rate
            zero_crossings = np.sum(np.diff(np.sign(signal_data)) != 0)
            quality_metrics['zero_crossing_rate'] = float(zero_crossings / len(signal_data))
            
            # Dynamic range
            quality_metrics['dynamic_range_db'] = float(20 * np.log10(np.max(np.abs(signal_data)) / (np.min(np.abs(signal_data[signal_data != 0])) + 1e-10)))
            
        except Exception as e:
            self.logger.warning(f"Signal quality analysis failed: {e}")
            quality_metrics['error'] = str(e)
        
        return quality_metrics
    
    def _analyze_class_distribution(self, labels: np.ndarray) -> Dict[str, Any]:
        """Analyze class distribution and balance"""
        unique_labels, counts = np.unique(labels, return_counts=True)
        
        class_analysis = {
            'class_counts': dict(zip(unique_labels.astype(str), counts.astype(int))),
            'class_percentages': dict(zip(unique_labels.astype(str), (counts / len(labels) * 100).astype(float))),
            'is_balanced': bool(np.std(counts) / np.mean(counts) < 0.1),  # CV < 10%
            'imbalance_ratio': float(np.max(counts) / np.min(counts))
        }
        
        return class_analysis
    
    def _create_visualizations(self, data: np.ndarray, labels: np.ndarray, output_path: Path):
        """Create comprehensive visualizations"""
        try:
            # 1. Class distribution plot
            plt.figure(figsize=(10, 6))
            unique_labels, counts = np.unique(labels, return_counts=True)
            plt.bar(unique_labels.astype(str), counts)
            plt.title('Class Distribution')
            plt.xlabel('Gesture Class')
            plt.ylabel('Number of Samples')
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig(output_path / 'class_distribution.png', dpi=300, bbox_inches='tight')
            plt.close()
            
            # 2. Data distribution histogram
            plt.figure(figsize=(12, 8))
            plt.subplot(2, 2, 1)
            plt.hist(data.flatten(), bins=50, alpha=0.7, edgecolor='black')
            plt.title('Overall Data Distribution')
            plt.xlabel('Amplitude')
            plt.ylabel('Frequency')
            
            # 3. Per-class data distribution
            plt.subplot(2, 2, 2)
            for label in np.unique(labels):
                class_data = data[labels == label]
                plt.hist(class_data.flatten(), bins=30, alpha=0.5, label=f'Class {label}')
            plt.title('Per-Class Data Distribution')
            plt.xlabel('Amplitude')
            plt.ylabel('Frequency')
            plt.legend()
            
            # 4. Box plot per class
            plt.subplot(2, 2, 3)
            class_data_list = []
            class_labels_list = []
            for label in np.unique(labels):
                class_data = data[labels == label]
                # Sample data for visualization (to avoid memory issues)
                sample_size = min(1000, len(class_data.flatten()))
                sampled_data = np.random.choice(class_data.flatten(), sample_size, replace=False)
                class_data_list.extend(sampled_data)
                class_labels_list.extend([f'Class {label}'] * sample_size)
            
            df = pd.DataFrame({'Value': class_data_list, 'Class': class_labels_list})
            sns.boxplot(data=df, x='Class', y='Value')
            plt.title('Data Distribution by Class')
            plt.xticks(rotation=45)
            
            # 5. Sample signals
            plt.subplot(2, 2, 4)
            n_samples_to_plot = min(5, len(data))
            for i in range(n_samples_to_plot):
                if data.ndim == 3:
                    # Multi-channel data - plot first channel
                    signal_data = data[i, 0, :]
                elif data.ndim == 2:
                    signal_data = data[i, :]
                else:
                    signal_data = data
                
                plt.plot(signal_data, alpha=0.7, label=f'Sample {i+1} (Class {labels[i]})')
            
            plt.title('Sample EMG Signals')
            plt.xlabel('Time Points')
            plt.ylabel('Amplitude')
            plt.legend()
            
            plt.tight_layout()
            plt.savefig(output_path / 'data_exploration.png', dpi=300, bbox_inches='tight')
            plt.close()
            
            # 6. Correlation matrix (if multi-channel)
            if data.ndim == 3 and data.shape[1] > 1:
                plt.figure(figsize=(10, 8))
                # Compute correlation between channels
                n_channels = data.shape[1]
                correlation_matrix = np.zeros((n_channels, n_channels))
                
                for i in range(n_channels):
                    for j in range(n_channels):
                        # Use a sample of data for correlation calculation
                        sample_size = min(1000, data.shape[0])
                        sample_indices = np.random.choice(data.shape[0], sample_size, replace=False)
                        
                        channel_i_data = data[sample_indices, i, :].flatten()
                        channel_j_data = data[sample_indices, j, :].flatten()
                        
                        correlation_matrix[i, j] = np.corrcoef(channel_i_data, channel_j_data)[0, 1]
                
                sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0,
                           xticklabels=[f'Ch{i+1}' for i in range(n_channels)],
                           yticklabels=[f'Ch{i+1}' for i in range(n_channels)])
                plt.title('Channel Correlation Matrix')
                plt.tight_layout()
                plt.savefig(output_path / 'channel_correlation.png', dpi=300, bbox_inches='tight')
                plt.close()
            
            self.logger.info("Visualizations created successfully")
            
        except Exception as e:
            self.logger.error(f"Visualization creation failed: {e}")
    
    def _save_analysis_results(self, results: Dict[str, Any], filepath: Path):
        """Save analysis results to JSON file"""
        try:
            # Convert numpy types to Python types for JSON serialization
            def convert_numpy_types(obj):
                if isinstance(obj, np.integer):
                    return int(obj)
                elif isinstance(obj, np.floating):
                    return float(obj)
                elif isinstance(obj, np.ndarray):
                    return obj.tolist()
                elif isinstance(obj, dict):
                    return {key: convert_numpy_types(value) for key, value in obj.items()}
                elif isinstance(obj, list):
                    return [convert_numpy_types(item) for item in obj]
                else:
                    return obj
            
            json_results = convert_numpy_types(results)
            
            with open(filepath, 'w') as f:
                json.dump(json_results, f, indent=2)
                
        except Exception as e:
            self.logger.error(f"Failed to save analysis results: {e}")
    
    def load_and_explore_data(self, matlab_path: Optional[str] = None, 
                             json_path: Optional[str] = None,
                             output_dir: str = "results") -> Dict[str, Any]:
        """Load data and perform exploration"""
        # This would integrate with data loading functionality
        # For now, return placeholder
        return {"status": "Data exploration functionality ready"}
