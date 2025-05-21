
import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
from typing import Tuple, List, Optional, Dict, Any
from sklearn.ensemble import IsolationForest
import time
import json
import hashlib
import base64
import struct
import pandas as pd
import os
import logging
from functools import lru_cache
from pathlib import Path

# Configure logging with configurable levels
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class TCCLogger:
    """Transaction Chain Code Logger for auditable, hash-linked logs."""
    def __init__(self, log_level: str = "INFO"):
        self.tcc_log: List['TCCLogEntry'] = []
        self.step_counter: int = 0
        self.log_level: int = getattr(logging, log_level.upper(), logging.INFO)

    def log(self, operation: str, input_data: bytes, output_data: bytes,
            metadata: Optional[Dict[str, Any]] = None,
            log_level: str = "INFO", error_code: str = "NONE") -> None:
        if getattr(logging, log_level.upper(), logging.INFO) >= self.log_level:
            try:
                entry = TCCLogEntry(
                    self.step_counter, operation, input_data, output_data,
                    metadata or {}, log_level, error_code, prev_hash=self._compute_prev_hash()
                )
                self.tcc_log.append(entry)
                self.step_counter += 1
                logger.log(self.log_level, f"Logged operation: {operation}, step: {self.step_counter}")
            except Exception as e:
                logger.error(f"Logging failed for operation {operation}: {str(e)}")

    def _compute_prev_hash(self) -> bytes:
        if not self.tcc_log:
            return b'\x00' * 32
        last_entry = self.tcc_log[-1]
        return hashlib.sha256(last_entry.to_bytes()).digest()

    def save_log(self, filename: str) -> None:
        try:
            with open(filename, 'w') as f:
                for entry in self.tcc_log:
                    f.write(json.dumps(entry.to_json()) + '\n')
            logger.info(f"Log saved to {filename}")
        except IOError as e:
            logger.error(f"Failed to save log to {filename}: {str(e)}")
            raise

class TCCLogEntry:
    """A single log entry with hash-linking for integrity."""
    def __init__(self, step: int, operation: str, input_data: bytes, output_data: bytes,
                 metadata: Dict[str, Any], log_level: str, error_code: str, prev_hash: bytes):
        self.step: int = step
        self.operation: str = operation
        self.input_data: bytes = input_data
        self.output_data: bytes = output_data
        self.metadata: Dict[str, Any] = metadata
        self.log_level: str = log_level
        self.error_code: str = error_code
        self.prev_hash: bytes = prev_hash
        self.operation_id: str = hashlib.sha256(f"{step}:{operation}:{time.time_ns()}".encode()).hexdigest()[:32]
        self.timestamp: int = time.time_ns()
        self.execution_time_ns: int = 0

    def to_bytes(self) -> bytes:
        start_time = time.time_ns()
        try:
            step_bytes = struct.pack('>I', self.step)
            op_bytes = self.operation.encode('utf-8').ljust(32, b'\x00')[:32]
            input_len_bytes = struct.pack('>I', len(self.input_data))
            output_len_bytes = struct.pack('>I', len(self.output_data))
            meta_bytes = json.dumps(self.metadata).encode('utf-8').ljust(128, b'\x00')[:128]
            level_bytes = self.log_level.encode('utf-8').ljust(16, b'\x00')[:16]
            error_bytes = self.error_code.encode('utf-8').ljust(16, b'\x00')[:16]
            op_id_bytes = self.operation_id.encode('utf-8').ljust(32, b'\x00')[:32]
            ts_bytes = struct.pack('>Q', self.timestamp)
            exec_time_bytes = struct.pack('>Q', self.execution_time_ns)
            result = (
                step_bytes + op_bytes + input_len_bytes + self.input_data +
                output_len_bytes + self.output_data + meta_bytes + level_bytes +
                error_bytes + self.prev_hash + op_id_bytes + ts_bytes + exec_time_bytes
            )
            self.execution_time_ns = time.time_ns() - start_time
            return result
        except Exception as e:
            logger.error(f"Failed to serialize log entry: {str(e)}")
            raise

    def to_json(self) -> Dict[str, Any]:
        return {
            "step": self.step,
            "operation": self.operation,
            "input_data": base64.b64encode(self.input_data).decode('utf-8'),
            "output_data": base64.b64encode(self.output_data).decode('utf-8'),
            "metadata": self.metadata,
            "log_level": self.log_level,
            "error_code": self.error_code,
            "prev_hash": base64.b64encode(self.prev_hash).decode('utf-8'),
            "operation_id": self.operation_id,
            "timestamp": self.timestamp,
            "execution_time_ns": self.execution_time_ns
        }

class BioInspiredSecuritySystem:
    """Bio-inspired anomaly detection system with FFT, Isolation Forest, adaptive thresholding, and robust audit logging."""
    def __init__(self, sample_rate: int = 1000, window_size: int = 100,
                 base_threshold: float = 0.1, adaptive_threshold_factor: float = 1.5,
                 log_level: str = "INFO"):
        self.sample_rate: int = sample_rate
        self.window_size: int = window_size
        self.base_threshold: float = base_threshold
        self.adaptive_threshold: float = base_threshold
        self.adaptive_threshold_factor: float = adaptive_threshold_factor
        self.execution_data: np.ndarray = np.array([])
        self.time_points: np.ndarray = np.array([])
        self.baseline_frequencies: Optional[np.ndarray] = None
        self.isolation_forest: IsolationForest = IsolationForest(contamination=0.1, random_state=42)
        self.anomaly_history: List[float] = []
        self.logger: TCCLogger = TCCLogger(log_level=log_level)
        self.start_time: float = time.time()

    def _generate_signal(self, is_malicious: bool = False, seed: Optional[int] = None) -> np.ndarray:
        """Generate synthetic time-domain signal with optional anomaly."""
        start_time = time.time_ns()
        try:
            if seed is not None:
                np.random.seed(seed)
            t = np.linspace(0, self.window_size / self.sample_rate, self.window_size)
            base_signal = (
                np.sin(2 * np.pi * 5 * t) +
                0.5 * np.sin(2 * np.pi * 10 * t) +
                0.2 * np.sin(2 * np.pi * 20 * t)
            )
            if is_malicious:
                anomaly = 0.8 * np.sin(2 * np.pi * np.random.uniform(40, 60) * t) * np.random.normal(1, 0.3, self.window_size)
                signal_out = base_signal + anomaly
            else:
                signal_out = base_signal + np.random.normal(0, 0.15, self.window_size)
            self.execution_data = signal_out
            self.time_points = t
            self.logger.log(
                "generate_signal", b"", signal_out.tobytes(),
                {"is_malicious": is_malicious, "seed": seed, "execution_time_ns": time.time_ns() - start_time}
            )
            return signal_out
        except Exception as e:
            self.logger.log(
                "generate_signal", b"", b"",
                {"error": str(e), "execution_time_ns": time.time_ns() - start_time}, "ERROR", "SIGNAL_GENERATION_FAILED"
            )
            raise

    @lru_cache(maxsize=32)
    def _compute_fft(self, data: tuple) -> Tuple[np.ndarray, np.ndarray]:
        """Compute FFT with windowing for frequency analysis, cached for performance."""
        start_time = time.time_ns()
        try:
            window = signal.windows.hann(self.window_size)
            fft_result = np.fft.fft(np.array(data) * window)
            freqs = np.fft.fftfreq(self.window_size, d=1 / self.sample_rate)
            magnitudes = np.abs(fft_result)
            mask = freqs > 0
            self.logger.log(
                "analyze_frequency", np.array(data).tobytes(), magnitudes[mask].tobytes(),
                {"freq_count": len(freqs[mask]), "execution_time_ns": time.time_ns() - start_time}
            )
            return freqs[mask], magnitudes[mask]
        except Exception as e:
            self.logger.log(
                "analyze_frequency", b"", b"",
                {"error": str(e), "execution_time_ns": time.time_ns() - start_time}, "ERROR", "FFT_COMPUTATION_FAILED"
            )
            raise

    def analyze_frequency_signature(self) -> Tuple[np.ndarray, np.ndarray]:
        """Wrapper for cached FFT computation."""
        return self._compute_fft(tuple(self.execution_data))

    def establish_baseline(self, num_samples: int = 10) -> np.ndarray:
        """Establish baseline using multiple samples and train Isolation Forest."""
        start_time = time.time_ns()
        try:
            all_magnitudes = []
            for i in range(num_samples):
                self._generate_signal(is_malicious=False, seed=i)
                _, magnitudes = self.analyze_frequency_signature()
                all_magnitudes.append(magnitudes)
            self.baseline_frequencies = np.mean(all_magnitudes, axis=0)
            self.isolation_forest.fit(np.array(all_magnitudes))
            self.logger.log(
                "establish_baseline", b"", self.baseline_frequencies.tobytes(),
                {"num_samples": num_samples, "execution_time_ns": time.time_ns() - start_time}
            )
            return self.baseline_frequencies
        except Exception as e:
            self.logger.log(
                "establish_baseline", b"", b"",
                {"error": str(e), "execution_time_ns": time.time_ns() - start_time}, "ERROR", "BASELINE_FAILED"
            )
            raise

    def _explain_anomaly(self, anomaly_score: float, freqs: np.ndarray, magnitudes: np.ndarray, baseline: np.ndarray) -> str:
        """Rule-based explanation for detected anomalies."""
        try:
            dominant_idx = np.argmax(np.abs(magnitudes - baseline))
            dominant_freq = freqs[dominant_idx]
            if anomaly_score > 2.0:
                return f"Severe anomaly: Unusual energy at {dominant_freq:.2f} Hz. Possible attack or malfunction detected."
            elif anomaly_score > 1.0:
                return f"Moderate anomaly: Elevated activity at {dominant_freq:.2f} Hz. Potential security issue or environmental change."
            elif anomaly_score > 0.5:
                return f"Mild anomaly: Slight deviation at {dominant_freq:.2f} Hz. Monitor for further irregularities."
            return "No significant anomaly detected."
        except Exception as e:
            logger.error(f"Anomaly explanation failed: {str(e)}")
            return "Anomaly explanation unavailable due to processing error."

    def detect_anomaly(self) -> Tuple[bool, float, str]:
        """Detect anomalies and provide explanation."""
        if self.baseline_frequencies is None:
            self.logger.log(
                "detect_anomaly", b"", b"",
                {"error": "Baseline not established"}, "ERROR", "NO_BASELINE"
            )
            raise ValueError("Baseline not established. Run establish_baseline() first.")

        start_time = time.time_ns()
        try:
            freqs, current_magnitudes = self.analyze_frequency_signature()
            deviation = np.abs(current_magnitudes - self.baseline_frequencies)
            fft_score = np.mean(deviation / (self.baseline_frequencies + 1e-8))
            ml_score = -self.isolation_forest.score_samples([current_magnitudes])[0]
            combined_score = 0.7 * fft_score + 0.3 * ml_score

            if self.anomaly_history:
                self.adaptive_threshold = np.mean(self.anomaly_history[-10:]) * self.adaptive_threshold_factor
            self.anomaly_history.append(combined_score)

            is_anomaly = combined_score > max(self.base_threshold, self.adaptive_threshold)
            explanation = self._explain_anomaly(combined_score, freqs, current_magnitudes, self.baseline_frequencies) if is_anomaly else ""

            self.logger.log(
                "detect_anomaly", current_magnitudes.tobytes(), np.array([is_anomaly, combined_score]).tobytes(),
                {
                    "anomaly_score": combined_score,
                    "threshold": self.adaptive_threshold,
                    "is_anomaly": is_anomaly,
                    "explanation": explanation,
                    "execution_time_ns": time.time_ns() - start_time
                }
            )
            return is_anomaly, combined_score, explanation
        except Exception as e:
            self.logger.log(
                "detect_anomaly", b"", b"",
                {"error": str(e), "execution_time_ns": time.time_ns() - start_time}, "ERROR", "ANOMALY_DETECTION_FAILED"
            )
            raise

    def apply_interference(self) -> np.ndarray:
        """Apply adaptive interference to suppress dominant anomaly frequencies."""
        start_time = time.time_ns()
        try:
            freqs, magnitudes = self.analyze_frequency_signature()
            dominant_freq = freqs[np.argmax(magnitudes)]
            interference_amplitude = 0.8 * np.max(magnitudes) / (np.max(self.baseline_frequencies + 1e-8))
            interference = -interference_amplitude * np.sin(2 * np.pi * dominant_freq * self.time_points + np.pi)
            self.execution_data += interference
            self.logger.log(
                "apply_interference", self.execution_data.tobytes(), interference.tobytes(),
                {
                    "dominant_freq": dominant_freq,
                    "amplitude": interference_amplitude,
                    "execution_time_ns": time.time_ns() - start_time
                }
            )
            return interference
        except Exception as e:
            self.logger.log(
                "apply_interference", b"", b"",
                {"error": str(e), "execution_time_ns": time.time_ns() - start_time}, "ERROR", "INTERFERENCE_FAILED"
            )
            raise

    def real_time_monitor(self, duration: float = 10.0, interval: float = 1.0) -> pd.DataFrame:
        """Simulate real-time monitoring with periodic signal analysis."""
        start_time = time.time_ns()
        results = []
        monitor_start = time.time()
        try:
            while time.time() - monitor_start < duration:
                is_malicious = np.random.choice([True, False], p=[0.2, 0.8])
                self._generate_signal(is_malicious=is_malicious)
                is_anomaly, score, explanation = self.detect_anomaly()
                if is_anomaly:
                    self.apply_interference()
                results.append({
                    'timestamp': time.time() - self.start_time,
                    'anomaly_detected': is_anomaly,
                    'anomaly_score': score,
                    'is_malicious': is_malicious,
                    'explanation': explanation
                })
                self.plot_results(f"Monitor_Snapshot_{len(results)}")
                time.sleep(interval)
            self.logger.log(
                "real_time_monitor", b"", b"",
                {
                    "duration": duration,
                    "interval": interval,
                    "total_snapshots": len(results),
                    "execution_time_ns": time.time_ns() - start_time
                }
            )
            return pd.DataFrame(results)
        except Exception as e:
            self.logger.log(
                "real_time_monitor", b"", b"",
                {"error": str(e), "execution_time_ns": time.time_ns() - start_time}, "ERROR", "MONITORING_FAILED"
            )
            raise

    def plot_results(self, title: str = "System Analysis") -> None:
        """Enhanced visualization with anomaly annotations and overwrite protection."""
        start_time = time.time_ns()
        try:
            freqs, magnitudes = self.analyze_frequency_signature()
            filename = title.lower().replace(" ", "_") + ".png"
            filepath = Path(filename)
            counter = 1
            while filepath.exists():
                filepath = Path(f"{title.lower().replace(' ', '_')}_{counter}.png")
                counter += 1
            plt.figure(figsize=(14, 10))
            plt.subplot(2, 1, 1)
            plt.plot(self.time_points, self.execution_data, label="Signal", color="dodgerblue")
            plt.title(f"{title} - Time Domain")
            plt.xlabel("Time (s)")
            plt.ylabel("Amplitude")
            plt.grid(True)
            plt.legend()
            plt.subplot(2, 1, 2)
            plt.plot(freqs, magnitudes, label="Observed", color="darkorange")
            if self.baseline_frequencies is not None:
                plt.plot(freqs, self.baseline_frequencies, linestyle="--", label="Baseline", color="gray")
                anomaly_mask = np.abs(magnitudes - self.baseline_frequencies) > self.adaptive_threshold * (self.baseline_frequencies + 1e-8)
                plt.scatter(freqs[anomaly_mask], magnitudes[anomaly_mask], color="red", label="Anomaly Peaks", zorder=5)
            plt.title(f"{title} - Frequency Domain")
            plt.xlabel("Frequency (Hz)")
            plt.ylabel("Magnitude")
            plt.grid(True)
            plt.legend()
            plt.tight_layout()
            plt.savefig(filepath)
            plt.close()
            self.logger.log(
                "plot_results", b"", str(filepath).encode('utf-8'),
                {"title": title, "filename": str(filepath), "execution_time_ns": time.time_ns() - start_time}
            )
        except Exception as e:
            self.logger.log(
                "plot_results", b"", b"",
                {"error": str(e), "execution_time_ns": time.time_ns() - start_time}, "ERROR", "PLOT_FAILED"
            )
            raise

    def save_log(self, filename: str = "security_system_log.jsonl") -> None:
        """Save audit log to file with error handling."""
        start_time = time.time_ns()
        try:
            filepath = Path(filename)
            if filepath.exists():
                counter = 1
                while Path(f"{filepath.stem}_{counter}{filepath.suffix}").exists():
                    counter += 1
                filepath = Path(f"{filepath.stem}_{counter}{filepath.suffix}")
            self.logger.save_log(str(filepath))
            self.logger.log(
                "save_log", b"", str(filepath).encode('utf-8'),
                {"filename": str(filepath), "execution_time_ns": time.time_ns() - start_time}
            )
        except Exception as e:
            self.logger.log(
                "save_log", b"", b"",
                {"error": str(e), "execution_time_ns": time.time_ns() - start_time}, "ERROR", "LOG_SAVE_FAILED"
            )
            raise

if __name__ == "__main__":
    try:
        system = BioInspiredSecuritySystem(log_level="INFO")
        system.establish_baseline()
        system._generate_signal(is_malicious=True, seed=42)
        is_anomaly, score, explanation = system.detect_anomaly()
        if is_anomaly:
            system.apply_interference()
        system.plot_results("Initial Analysis")
        results_df = system.real_time_monitor(duration=5.0)
        system.save_log()
        print(results_df)
        print(f"Anomaly explanation: {explanation}")
    except Exception as e:
        logger.error(f"Main execution failed: {str(e)}")
