import csv
import os
from datetime import datetime


class Logger:
    """
    Lightweight training logger.

    Prints messages to the console (with a timestamp) and appends
    per-step metrics to a CSV file under outputs/logs/, so training
    curves can be plotted later without needing TensorBoard/W&B.

    Args:
        log_dir  : directory where the log file will be written
        filename : name of the CSV log file
    """

    def __init__(self, log_dir: str, filename: str = "train_log.csv"):
        os.makedirs(log_dir, exist_ok=True)
        self.log_path = os.path.join(log_dir, filename)
        self._header_written = os.path.isfile(self.log_path)

    def log(self, message: str) -> None:
        """Prints a timestamped message to the console."""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"[{timestamp}] {message}")

    def log_metrics(self, epoch: int, step: int, metrics: dict) -> None:
        """
        Appends a row of metrics to the CSV log file, creating the file
        (and header) on the first call.

        Args:
            epoch   : current epoch number
            step    : current step/iteration number
            metrics : dict of metric name 
        """
        fieldnames = ["epoch", "step"] + list(metrics.keys())
        row = {"epoch": epoch, "step": step, **metrics}

        write_header = not self._header_written
        with open(self.log_path, mode="a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            if write_header:
                writer.writeheader()
                self._header_written = True
            writer.writerow(row)