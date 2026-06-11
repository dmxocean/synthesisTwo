# -*- coding: utf-8 -*-
"""
Hardware detection and device selection for PyTorch-based pipeline scripts

Supports CUDA (NVIDIA), MPS (Apple Silicon), and CPU fallback
The SLURM detection enables cluster-aware DataLoader worker configuration
"""

import os
import torch

BASE_PATH = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

class DeviceManager:
    """
    Orchestrates hardware acceleration and environment-specific parallelization
    """
    @staticmethod
    def get_device():
        """
        Identify the best available compute device
        """
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    @staticmethod
    def is_slurm():
        """
        Check if the script is running inside a SLURM job
        """
        return "SLURM_JOB_ID" in os.environ

    @staticmethod
    def get_optimal_workers():
        """
        Calculate the recommended number of DataLoader workers for the current environment
        """
        if DeviceManager.is_slurm():
            return int(os.environ.get("SLURM_CPUS_PER_TASK", 4))
        return 0 # Local inference is faster on the main thread

    @staticmethod
    def print_hardware_summary():
        """
        Log a concise summary of the active compute environment to the console
        """
        device = DeviceManager.get_device()
        env = "SLURM cluster" if DeviceManager.is_slurm() else "local machine"
        
        print(f"[*] Device: {device} | {env}")
        if device.type == "cuda":
            name = torch.cuda.get_device_name(0)
            vram = torch.cuda.get_device_properties(0).total_memory / 1024**3
            print(f"[*] GPU: {name} | VRAM: {vram:.2f} GB")
