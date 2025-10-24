"""
Audio Preprocessing Module for Respiratory Sound Analysis
=======================================================

This module provides comprehensive audio preprocessing functionality including
noise reduction, normalization, segmentation, and quality enhancement for 
respiratory sound analysis.
"""

import numpy as np
import librosa
import scipy.signal
from scipy.signal import butter, sosfilt, find_peaks
import noisereduce as nr
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import warnings
warnings.filterwarnings('ignore')

class AudioPreprocessor:
    """
    Comprehensive audio preprocessing pipeline for respiratory sounds.
    
    Features:
    - Noise reduction using spectral subtraction and Wiener filtering
    - Audio normalization and standardization
    - Respiratory cycle segmentation
    - Quality assessment and enhancement
    - Format conversion and resampling
    """
    
    def __init__(self, target_sr=16000, target_duration=None):
        """
        Initialize the audio preprocessor.
        
        Args:
            target_sr (int): Target sample rate for processing
            target_duration (float): Target duration in seconds (None for variable)
        """
        self.target_sr = target_sr
        self.target_duration = target_duration
        self.scaler = StandardScaler()
        
    def load_audio(self, file_path, sr=None, offset=0.0, duration=None):
        """
        Load audio file with proper error handling.
        
        Args:
            file_path (str): Path to audio file
            sr (int): Sample rate (None for original)
            offset (float): Start time in seconds
            duration (float): Duration to load in seconds
            
        Returns:
            tuple: (audio_data, sample_rate)
        """
        try:
            audio, sr = librosa.load(
                file_path, 
                sr=sr or self.target_sr,
                offset=offset,
                duration=duration,
                mono=True
            )
            return audio, sr
        except Exception as e:
            raise ValueError(f"Error loading audio file {file_path}: {str(e)}")
    
    def resample_audio(self, audio, original_sr, target_sr=None):
        """
        Resample audio to target sample rate.
        
        Args:
            audio (np.ndarray): Input audio signal
            original_sr (int): Original sample rate
            target_sr (int): Target sample rate
            
        Returns:
            np.ndarray: Resampled audio
        """
        if target_sr is None:
            target_sr = self.target_sr
            
        if original_sr != target_sr:
            audio = librosa.resample(audio, orig_sr=original_sr, target_sr=target_sr)
        
        return audio
    
    def trim_silence(self, audio, sr, top_db=20, frame_length=2048, hop_length=512):
        """
        Remove silence from beginning and end of audio.
        
        Args:
            audio (np.ndarray): Input audio signal
            sr (int): Sample rate
            top_db (float): Threshold for silence detection
            frame_length (int): Frame length for analysis
            hop_length (int): Hop length for analysis
            
        Returns:
            np.ndarray: Trimmed audio
        """
        # Trim silence
        audio_trimmed, _ = librosa.effects.trim(
            audio, 
            top_db=top_db,
            frame_length=frame_length,
            hop_length=hop_length
        )
        
        return audio_trimmed
    
    def normalize_audio(self, audio, method='rms'):
        """
        Normalize audio signal.
        
        Args:
            audio (np.ndarray): Input audio signal
            method (str): Normalization method ('rms', 'peak', 'lufs')
            
        Returns:
            np.ndarray: Normalized audio
        """
        if method == 'rms':
            # RMS normalization
            rms = np.sqrt(np.mean(audio**2))
            if rms > 0:
                audio = audio / rms * 0.1  # Target RMS of 0.1
        
        elif method == 'peak':
            # Peak normalization
            peak = np.max(np.abs(audio))
            if peak > 0:
                audio = audio / peak * 0.8  # Target peak of 0.8
        
        elif method == 'lufs':
            # LUFS-inspired normalization (simplified)
            # Apply A-weighting approximation
            audio_weighted = self._apply_a_weighting(audio)
            lufs_approx = np.sqrt(np.mean(audio_weighted**2))
            if lufs_approx > 0:
                audio = audio / lufs_approx * 0.1
        
        return audio
    
    def _apply_a_weighting(self, audio, sr=16000):
        """
        Apply A-weighting filter (approximation).
        
        Args:
            audio (np.ndarray): Input audio signal
            sr (int): Sample rate
            
        Returns:
            np.ndarray: A-weighted audio
        """
        # Simplified A-weighting (high-pass emphasis)
        nyquist = sr / 2
        high_freq = 1000 / nyquist  # Emphasize frequencies above 1kHz
        
        b, a = butter(2, high_freq, btype='high')
        weighted_audio = scipy.signal.filtfilt(b, a, audio)
        
        return weighted_audio
    
    def reduce_noise(self, audio, sr, method='spectral_subtraction', noise_duration=1.0):
        """
        Apply noise reduction to audio signal.
        
        Args:
            audio (np.ndarray): Input audio signal
            sr (int): Sample rate
            method (str): Noise reduction method
            noise_duration (float): Duration for noise estimation (seconds)
            
        Returns:
            np.ndarray: Denoised audio
        """
        if method == 'spectral_subtraction':
            # Use noisereduce library for spectral subtraction
            reduced_noise = nr.reduce_noise(y=audio, sr=sr, stationary=False)
            return reduced_noise
        
        elif method == 'wiener_filter':
            return self._wiener_filter(audio, sr)
        
        elif method == 'bandpass_filter':
            return self._bandpass_filter(audio, sr)
        
        else:
            raise ValueError(f"Unknown noise reduction method: {method}")
    
    def _wiener_filter(self, audio, sr, noise_factor=0.1):
        """
        Apply Wiener filtering for noise reduction.
        
        Args:
            audio (np.ndarray): Input audio signal
            sr (int): Sample rate
            noise_factor (float): Noise estimation factor
            
        Returns:
            np.ndarray: Filtered audio
        """
        # Estimate noise from first/last segments
        noise_samples = int(sr * 0.5)  # 0.5 second for noise estimation
        noise_est = np.concatenate([audio[:noise_samples], audio[-noise_samples:]])
        noise_power = np.var(noise_est)
        
        # Apply Wiener filter in frequency domain
        audio_fft = np.fft.fft(audio)
        audio_power = np.abs(audio_fft)**2
        
        # Wiener filter coefficient
        wiener_coef = audio_power / (audio_power + noise_power * noise_factor)
        filtered_fft = audio_fft * wiener_coef
        
        filtered_audio = np.real(np.fft.ifft(filtered_fft))
        
        return filtered_audio
    
    def _bandpass_filter(self, audio, sr, low_freq=20, high_freq=2000):
        """
        Apply bandpass filter to focus on respiratory frequency range.
        
        Args:
            audio (np.ndarray): Input audio signal
            sr (int): Sample rate
            low_freq (float): Lower cutoff frequency (Hz)
            high_freq (float): Upper cutoff frequency (Hz)
            
        Returns:
            np.ndarray: Filtered audio
        """
        nyquist = sr / 2
        low = low_freq / nyquist
        high = high_freq / nyquist
        
        # Design 5th order Butterworth bandpass filter
        sos = butter(5, [low, high], btype='band', output='sos')
        filtered_audio = sosfilt(sos, audio)
        
        return filtered_audio
    
    def segment_respiratory_cycles(self, audio, sr, min_cycle_duration=2.0, max_cycle_duration=8.0):
        """
        Segment audio into individual respiratory cycles.
        
        Args:
            audio (np.ndarray): Input audio signal
            sr (int): Sample rate
            min_cycle_duration (float): Minimum cycle duration (seconds)
            max_cycle_duration (float): Maximum cycle duration (seconds)
            
        Returns:
            list: List of audio segments for each respiratory cycle
        """
        # Calculate envelope using Hilbert transform
        analytic_signal = scipy.signal.hilbert(audio)
        envelope = np.abs(analytic_signal)
        
        # Smooth envelope
        window_size = int(sr * 0.1)  # 100ms smoothing window
        envelope_smooth = scipy.signal.savgol_filter(envelope, window_size, 3)
        
        # Find peaks in envelope (potential breath starts)
        min_distance = int(sr * min_cycle_duration)
        peaks, _ = find_peaks(
            envelope_smooth, 
            distance=min_distance,
            prominence=np.std(envelope_smooth) * 0.5
        )
        
        # Extract segments between peaks
        cycles = []
        for i in range(len(peaks) - 1):
            start_idx = peaks[i]
            end_idx = peaks[i + 1]
            
            cycle_duration = (end_idx - start_idx) / sr
            
            # Filter by duration
            if min_cycle_duration <= cycle_duration <= max_cycle_duration:
                cycle = audio[start_idx:end_idx]
                cycles.append(cycle)
        
        return cycles
    
    def pad_or_truncate(self, audio, target_length):
        """
        Pad or truncate audio to target length.
        
        Args:
            audio (np.ndarray): Input audio signal
            target_length (int): Target length in samples
            
        Returns:
            np.ndarray: Processed audio
        """
        current_length = len(audio)
        
        if current_length < target_length:
            # Pad with zeros
            padding = target_length - current_length
            audio = np.pad(audio, (0, padding), mode='constant')
        elif current_length > target_length:
            # Truncate
            audio = audio[:target_length]
        
        return audio
    
    def augment_audio(self, audio, sr, augmentation_type='noise'):
        """
        Apply audio augmentation techniques.
        
        Args:
            audio (np.ndarray): Input audio signal
            sr (int): Sample rate
            augmentation_type (str): Type of augmentation
            
        Returns:
            np.ndarray: Augmented audio
        """
        if augmentation_type == 'noise':
            # Add white noise
            noise_factor = 0.005
            noise = np.random.normal(0, noise_factor, len(audio))
            return audio + noise
        
        elif augmentation_type == 'time_stretch':
            # Time stretching
            stretch_factor = np.random.uniform(0.8, 1.2)
            return librosa.effects.time_stretch(audio, rate=stretch_factor)
        
        elif augmentation_type == 'pitch_shift':
            # Pitch shifting
            pitch_shift = np.random.uniform(-2, 2)  # semitones
            return librosa.effects.pitch_shift(audio, sr=sr, n_steps=pitch_shift)
        
        elif augmentation_type == 'volume':
            # Volume scaling
            volume_factor = np.random.uniform(0.7, 1.3)
            return audio * volume_factor
        
        else:
            return audio
    
    def assess_quality(self, audio, sr):
        """
        Assess audio quality metrics.
        
        Args:
            audio (np.ndarray): Input audio signal
            sr (int): Sample rate
            
        Returns:
            dict: Quality metrics
        """
        # Signal-to-noise ratio estimation
        # Use spectral analysis to estimate SNR
        stft = librosa.stft(audio)
        magnitude = np.abs(stft)
        
        # Estimate noise floor (bottom 10th percentile)
        noise_floor = np.percentile(magnitude, 10)
        signal_power = np.mean(magnitude)
        
        snr_estimate = 20 * np.log10(signal_power / (noise_floor + 1e-10))
        
        # Dynamic range
        dynamic_range = 20 * np.log10(np.max(np.abs(audio)) / (np.std(audio) + 1e-10))
        
        # Zero crossing rate (indicative of signal complexity)
        zcr = librosa.feature.zero_crossing_rate(audio)[0]
        avg_zcr = np.mean(zcr)
        
        # Spectral centroid (brightness measure)
        spectral_centroids = librosa.feature.spectral_centroid(y=audio, sr=sr)[0]
        avg_brightness = np.mean(spectral_centroids)
        
        return {
            'snr_estimate_db': snr_estimate,
            'dynamic_range_db': dynamic_range,
            'avg_zero_crossing_rate': avg_zcr,
            'avg_spectral_centroid_hz': avg_brightness,
            'duration_seconds': len(audio) / sr,
            'sample_rate': sr,
            'peak_amplitude': np.max(np.abs(audio)),
            'rms_amplitude': np.sqrt(np.mean(audio**2))
        }
    
    def preprocess_pipeline(self, audio, sr, apply_noise_reduction=True, 
                          normalize=True, segment_cycles=False):
        """
        Complete preprocessing pipeline.
        
        Args:
            audio (np.ndarray): Input audio signal
            sr (int): Sample rate
            apply_noise_reduction (bool): Whether to apply noise reduction
            normalize (bool): Whether to normalize audio
            segment_cycles (bool): Whether to segment respiratory cycles
            
        Returns:
            dict: Processed audio data and metadata
        """
        original_audio = audio.copy()
        
        # Step 1: Resample if needed
        if sr != self.target_sr:
            audio = self.resample_audio(audio, sr, self.target_sr)
            sr = self.target_sr
        
        # Step 2: Trim silence
        audio = self.trim_silence(audio, sr)
        
        # Step 3: Apply noise reduction
        if apply_noise_reduction:
            audio = self.reduce_noise(audio, sr)
        
        # Step 4: Normalize
        if normalize:
            audio = self.normalize_audio(audio)
        
        # Step 5: Pad or truncate if target duration specified
        if self.target_duration:
            target_samples = int(self.target_duration * sr)
            audio = self.pad_or_truncate(audio, target_samples)
        
        # Step 6: Segment respiratory cycles if requested
        cycles = []
        if segment_cycles:
            cycles = self.segment_respiratory_cycles(audio, sr)
        
        # Step 7: Quality assessment
        quality_metrics = self.assess_quality(audio, sr)
        
        return {
            'processed_audio': audio,
            'original_audio': original_audio,
            'sample_rate': sr,
            'respiratory_cycles': cycles,
            'quality_metrics': quality_metrics,
            'preprocessing_applied': {
                'noise_reduction': apply_noise_reduction,
                'normalization': normalize,
                'cycle_segmentation': segment_cycles
            }
        }