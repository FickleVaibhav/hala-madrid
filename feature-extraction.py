"""
Feature Extraction Module for Respiratory Sound Analysis
======================================================

This module provides comprehensive feature extraction functionality for respiratory
sounds, including MFCC, Mel-spectrogram, Chroma, spectral features, and temporal
characteristics optimized for respiratory disease classification.
"""

import numpy as np
import librosa
import librosa.display
from scipy import stats
from scipy.signal import hilbert, find_peaks
import scipy.signal
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

class FeatureExtractor:
    """
    Comprehensive feature extraction for respiratory sound analysis.
    
    Features extracted:
    - MFCC (Mel-frequency Cepstral Coefficients)
    - Mel-spectrogram features
    - Chroma features
    - Spectral features (centroid, rolloff, bandwidth, etc.)
    - Temporal features (ZCR, energy, rhythm)
    - Statistical features (mean, std, skewness, kurtosis)
    """
    
    def __init__(self, sr=16000):
        """
        Initialize the feature extractor.
        
        Args:
            sr (int): Sample rate for processing
        """
        self.sr = sr
        self.scaler = StandardScaler()
        
        # Feature extraction parameters
        self.n_mfcc = 13
        self.n_fft = 2048
        self.hop_length = 512
        self.n_mels = 128
        self.n_chroma = 12
        
    def extract_mfcc_features(self, audio, n_mfcc=None):
        """
        Extract MFCC features from audio signal.
        
        Args:
            audio (np.ndarray): Input audio signal
            n_mfcc (int): Number of MFCC coefficients
            
        Returns:
            dict: MFCC features and statistics
        """
        if n_mfcc is None:
            n_mfcc = self.n_mfcc
            
        # Extract MFCCs
        mfccs = librosa.feature.mfcc(
            y=audio, 
            sr=self.sr, 
            n_mfcc=n_mfcc,
            n_fft=self.n_fft,
            hop_length=self.hop_length
        )
        
        # Calculate statistics for each MFCC coefficient
        mfcc_features = {}
        
        for i in range(n_mfcc):
            coeff = mfccs[i, :]
            mfcc_features.update({
                f'mfcc_{i}_mean': np.mean(coeff),
                f'mfcc_{i}_std': np.std(coeff),
                f'mfcc_{i}_max': np.max(coeff),
                f'mfcc_{i}_min': np.min(coeff),
                f'mfcc_{i}_median': np.median(coeff),
                f'mfcc_{i}_skewness': stats.skew(coeff),
                f'mfcc_{i}_kurtosis': stats.kurtosis(coeff)
            })
        
        # Delta and delta-delta features
        mfcc_delta = librosa.feature.delta(mfccs)
        mfcc_delta2 = librosa.feature.delta(mfccs, order=2)
        
        # Statistics for deltas
        mfcc_features.update({
            'mfcc_delta_mean': np.mean(mfcc_delta),
            'mfcc_delta_std': np.std(mfcc_delta),
            'mfcc_delta2_mean': np.mean(mfcc_delta2),
            'mfcc_delta2_std': np.std(mfcc_delta2)
        })
        
        return mfcc_features
    
    def extract_mel_spectrogram_features(self, audio):
        """
        Extract Mel-spectrogram based features.
        
        Args:
            audio (np.ndarray): Input audio signal
            
        Returns:
            dict: Mel-spectrogram features
        """
        # Mel-spectrogram
        mel_spec = librosa.feature.melspectrogram(
            y=audio,
            sr=self.sr,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            n_mels=self.n_mels
        )
        
        # Convert to log scale
        log_mel_spec = librosa.power_to_db(mel_spec, ref=np.max)
        
        # Statistical features
        mel_features = {
            'mel_spec_mean': np.mean(log_mel_spec),
            'mel_spec_std': np.std(log_mel_spec),
            'mel_spec_max': np.max(log_mel_spec),
            'mel_spec_min': np.min(log_mel_spec),
            'mel_spec_median': np.median(log_mel_spec),
            'mel_spec_skewness': stats.skew(log_mel_spec.flatten()),
            'mel_spec_kurtosis': stats.kurtosis(log_mel_spec.flatten())
        }
        
        # Spectral contrast
        contrast = librosa.feature.spectral_contrast(
            y=audio, sr=self.sr, hop_length=self.hop_length
        )
        mel_features.update({
            'spectral_contrast_mean': np.mean(contrast),
            'spectral_contrast_std': np.std(contrast)
        })
        
        # Mel-frequency band energies
        mel_bands = np.mean(mel_spec, axis=1)  # Average across time
        for i, energy in enumerate(mel_bands):
            mel_features[f'mel_band_{i}_energy'] = energy
        
        return mel_features
    
    def extract_chroma_features(self, audio):
        """
        Extract chroma features from audio signal.
        
        Args:
            audio (np.ndarray): Input audio signal
            
        Returns:
            dict: Chroma features
        """
        # Chroma STFT
        chroma_stft = librosa.feature.chroma_stft(
            y=audio,
            sr=self.sr,
            hop_length=self.hop_length,
            n_chroma=self.n_chroma
        )
        
        # Chroma CQT (Constant-Q Transform)
        chroma_cqt = librosa.feature.chroma_cqt(
            y=audio,
            sr=self.sr,
            hop_length=self.hop_length,
            n_chroma=self.n_chroma
        )
        
        # Chroma CENS (Chroma Energy Normalized Statistics)
        chroma_cens = librosa.feature.chroma_cens(
            y=audio,
            sr=self.sr,
            hop_length=self.hop_length,
            n_chroma=self.n_chroma
        )
        
        chroma_features = {}
        
        # Statistics for each chroma type
        for chroma_type, chroma_data in [
            ('stft', chroma_stft), 
            ('cqt', chroma_cqt), 
            ('cens', chroma_cens)
        ]:
            chroma_features.update({
                f'chroma_{chroma_type}_mean': np.mean(chroma_data),
                f'chroma_{chroma_type}_std': np.std(chroma_data),
                f'chroma_{chroma_type}_max': np.max(chroma_data),
                f'chroma_{chroma_type}_min': np.min(chroma_data)
            })
            
            # Individual chroma bin statistics
            for i in range(self.n_chroma):
                bin_data = chroma_data[i, :]
                chroma_features[f'chroma_{chroma_type}_bin_{i}_mean'] = np.mean(bin_data)
        
        return chroma_features
    
    def extract_spectral_features(self, audio):
        """
        Extract spectral characteristics from audio signal.
        
        Args:
            audio (np.ndarray): Input audio signal
            
        Returns:
            dict: Spectral features
        """
        spectral_features = {}
        
        # Spectral centroid
        spectral_centroids = librosa.feature.spectral_centroid(
            y=audio, sr=self.sr, hop_length=self.hop_length
        )[0]
        
        # Spectral rolloff
        spectral_rolloff = librosa.feature.spectral_rolloff(
            y=audio, sr=self.sr, hop_length=self.hop_length
        )[0]
        
        # Spectral bandwidth
        spectral_bandwidth = librosa.feature.spectral_bandwidth(
            y=audio, sr=self.sr, hop_length=self.hop_length
        )[0]
        
        # Spectral flatness
        spectral_flatness = librosa.feature.spectral_flatness(
            y=audio, hop_length=self.hop_length
        )[0]
        
        # Zero crossing rate
        zcr = librosa.feature.zero_crossing_rate(
            audio, frame_length=self.n_fft, hop_length=self.hop_length
        )[0]
        
        # Compute statistics for each spectral feature
        for feature_name, feature_data in [
            ('centroid', spectral_centroids),
            ('rolloff', spectral_rolloff),
            ('bandwidth', spectral_bandwidth),
            ('flatness', spectral_flatness),
            ('zcr', zcr)
        ]:
            spectral_features.update({
                f'spectral_{feature_name}_mean': np.mean(feature_data),
                f'spectral_{feature_name}_std': np.std(feature_data),
                f'spectral_{feature_name}_max': np.max(feature_data),
                f'spectral_{feature_name}_min': np.min(feature_data),
                f'spectral_{feature_name}_median': np.median(feature_data),
                f'spectral_{feature_name}_skewness': stats.skew(feature_data),
                f'spectral_{feature_name}_kurtosis': stats.kurtosis(feature_data)
            })
        
        return spectral_features
    
    def extract_temporal_features(self, audio):
        """
        Extract temporal characteristics from audio signal.
        
        Args:
            audio (np.ndarray): Input audio signal
            
        Returns:
            dict: Temporal features
        """
        temporal_features = {}
        
        # Basic temporal statistics
        temporal_features.update({
            'audio_length': len(audio),
            'duration_seconds': len(audio) / self.sr,
            'rms_energy': np.sqrt(np.mean(audio**2)),
            'max_amplitude': np.max(np.abs(audio)),
            'min_amplitude': np.min(np.abs(audio)),
            'mean_amplitude': np.mean(np.abs(audio)),
            'std_amplitude': np.std(audio)
        })
        
        # Envelope features using Hilbert transform
        analytic_signal = hilbert(audio)
        envelope = np.abs(analytic_signal)
        
        temporal_features.update({
            'envelope_mean': np.mean(envelope),
            'envelope_std': np.std(envelope),
            'envelope_max': np.max(envelope),
            'envelope_skewness': stats.skew(envelope),
            'envelope_kurtosis': stats.kurtosis(envelope)
        })
        
        # Rhythm and periodicity features
        # Find peaks in envelope for rhythm analysis
        peaks, _ = find_peaks(envelope, distance=int(self.sr * 0.5))  # Min 0.5s between peaks
        
        if len(peaks) > 1:
            peak_intervals = np.diff(peaks) / self.sr  # Convert to seconds
            temporal_features.update({
                'rhythm_regularity': np.std(peak_intervals) if len(peak_intervals) > 1 else 0,
                'avg_peak_interval': np.mean(peak_intervals),
                'breathing_rate_estimate': 60 / np.mean(peak_intervals) if np.mean(peak_intervals) > 0 else 0,
                'num_breath_cycles': len(peaks)
            })
        else:
            temporal_features.update({
                'rhythm_regularity': 0,
                'avg_peak_interval': 0,
                'breathing_rate_estimate': 0,
                'num_breath_cycles': 0
            })
        
        # Energy distribution over time
        frame_energy = []
        frame_size = int(self.sr * 0.1)  # 100ms frames
        
        for i in range(0, len(audio) - frame_size, frame_size):
            frame = audio[i:i + frame_size]
            frame_energy.append(np.sum(frame**2))
        
        if frame_energy:
            frame_energy = np.array(frame_energy)
            temporal_features.update({
                'energy_variance': np.var(frame_energy),
                'energy_entropy': stats.entropy(frame_energy + 1e-10),
                'energy_concentration': np.max(frame_energy) / (np.sum(frame_energy) + 1e-10)
            })
        
        return temporal_features
    
    def extract_respiratory_specific_features(self, audio):
        """
        Extract features specific to respiratory sound analysis.
        
        Args:
            audio (np.ndarray): Input audio signal
            
        Returns:
            dict: Respiratory-specific features
        """
        resp_features = {}
        
        # Wheeze detection features (high-frequency continuous sounds)
        # Filter for wheeze frequency range (100-2000 Hz)
        nyquist = self.sr / 2
        low = 100 / nyquist
        high = 2000 / nyquist
        
        b, a = scipy.signal.butter(4, [low, high], btype='band')
        wheeze_filtered = scipy.signal.filtfilt(b, a, audio)
        
        resp_features.update({
            'wheeze_energy': np.sum(wheeze_filtered**2),
            'wheeze_ratio': np.sum(wheeze_filtered**2) / (np.sum(audio**2) + 1e-10)
        })
        
        # Crackle detection features (transient, high-frequency sounds)
        # High-pass filter for crackles (>250 Hz)
        high_freq = 250 / nyquist
        b, a = scipy.signal.butter(4, high_freq, btype='high')
        crackle_filtered = scipy.signal.filtfilt(b, a, audio)
        
        # Find transient events (potential crackles)
        envelope = np.abs(hilbert(crackle_filtered))
        peaks, properties = find_peaks(
            envelope, 
            height=np.std(envelope) * 2,
            distance=int(self.sr * 0.01)  # Min 10ms between crackles
        )
        
        resp_features.update({
            'crackle_count': len(peaks),
            'crackle_density': len(peaks) / (len(audio) / self.sr),
            'crackle_energy': np.sum(crackle_filtered**2),
            'crackle_ratio': np.sum(crackle_filtered**2) / (np.sum(audio**2) + 1e-10)
        })
        
        # Stridor detection (inspiratory high-pitched sound)
        # Focus on higher frequencies (200-2000 Hz)
        stridor_low = 200 / nyquist
        stridor_high = 2000 / nyquist
        
        b, a = scipy.signal.butter(4, [stridor_low, stridor_high], btype='band')
        stridor_filtered = scipy.signal.filtfilt(b, a, audio)
        
        resp_features.update({
            'stridor_energy': np.sum(stridor_filtered**2),
            'stridor_ratio': np.sum(stridor_filtered**2) / (np.sum(audio**2) + 1e-10)
        })
        
        # Rhonchus detection (low-frequency continuous sounds)
        # Low-pass filter for rhonchus (<300 Hz)
        rhonchus_freq = 300 / nyquist
        b, a = scipy.signal.butter(4, rhonchus_freq, btype='low')
        rhonchus_filtered = scipy.signal.filtfilt(b, a, audio)
        
        resp_features.update({
            'rhonchus_energy': np.sum(rhonchus_filtered**2),
            'rhonchus_ratio': np.sum(rhonchus_filtered**2) / (np.sum(audio**2) + 1e-10)
        })
        
        # Pleural friction rub detection (grating sound)
        # Look for irregular, broadband transients
        stft = librosa.stft(audio, hop_length=self.hop_length)
        magnitude = np.abs(stft)
        
        # Measure spectral irregularity across frequency bins
        spectral_irregularity = []
        for t in range(magnitude.shape[1]):
            spectrum = magnitude[:, t]
            if np.sum(spectrum) > 0:
                # Calculate roughness as variation in adjacent frequency bins
                roughness = np.mean(np.abs(np.diff(spectrum)) / (spectrum[1:] + 1e-10))
                spectral_irregularity.append(roughness)
        
        if spectral_irregularity:
            resp_features.update({
                'pleural_rub_indicator': np.mean(spectral_irregularity),
                'spectral_roughness': np.std(spectral_irregularity)
            })
        
        return resp_features
    
    def extract_comprehensive_features(self, audio):
        """
        Extract all features from audio signal.
        
        Args:
            audio (np.ndarray): Input audio signal
            
        Returns:
            dict: Comprehensive feature dictionary
        """
        all_features = {}
        
        # Extract all feature types
        all_features.update(self.extract_mfcc_features(audio))
        all_features.update(self.extract_mel_spectrogram_features(audio))
        all_features.update(self.extract_chroma_features(audio))
        all_features.update(self.extract_spectral_features(audio))
        all_features.update(self.extract_temporal_features(audio))
        all_features.update(self.extract_respiratory_specific_features(audio))
        
        # Convert to numpy array for ML models
        feature_names = list(all_features.keys())
        feature_values = np.array(list(all_features.values())).reshape(1, -1)
        
        return {
            'features': feature_values,
            'feature_names': feature_names,
            'feature_dict': all_features,
            'n_features': len(feature_names)
        }
    
    def extract_features_from_cycles(self, respiratory_cycles):
        """
        Extract features from multiple respiratory cycles and aggregate.
        
        Args:
            respiratory_cycles (list): List of audio arrays for each cycle
            
        Returns:
            dict: Aggregated features across all cycles
        """
        if not respiratory_cycles:
            return self.extract_comprehensive_features(np.array([]))
        
        cycle_features = []
        
        # Extract features from each cycle
        for cycle in respiratory_cycles:
            if len(cycle) > 0:
                features = self.extract_comprehensive_features(cycle)
                cycle_features.append(features['feature_dict'])
        
        if not cycle_features:
            return self.extract_comprehensive_features(np.array([]))
        
        # Aggregate features across cycles
        aggregated_features = {}
        feature_names = cycle_features[0].keys()
        
        for feature_name in feature_names:
            feature_values = [cycle[feature_name] for cycle in cycle_features]
            
            # Calculate statistics across cycles
            aggregated_features.update({
                f'{feature_name}_cycle_mean': np.mean(feature_values),
                f'{feature_name}_cycle_std': np.std(feature_values),
                f'{feature_name}_cycle_max': np.max(feature_values),
                f'{feature_name}_cycle_min': np.min(feature_values),
                f'{feature_name}_cycle_range': np.max(feature_values) - np.min(feature_values)
            })
        
        # Add cycle-level statistics
        aggregated_features.update({
            'num_cycles': len(cycle_features),
            'avg_cycle_length': np.mean([len(cycle) for cycle in respiratory_cycles]),
            'cycle_length_variability': np.std([len(cycle) for cycle in respiratory_cycles])
        })
        
        # Convert to format expected by ML models
        feature_names = list(aggregated_features.keys())
        feature_values = np.array(list(aggregated_features.values())).reshape(1, -1)
        
        return {
            'features': feature_values,
            'feature_names': feature_names,
            'feature_dict': aggregated_features,
            'n_features': len(feature_names),
            'individual_cycles': cycle_features
        }