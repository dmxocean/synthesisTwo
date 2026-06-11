# -*- coding: utf-8 -*-
"""
Common training utilities for segmentation models
"""


import os

BASE_PATH = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def count_parameters(model):
    """
    Return (trainable_count, total_count) parameter counts for logging
    """
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in model.parameters())
    return trainable, total


class EarlyStopping:
    """
    Terminate training when a monitored metric stops improving

    Args:
        patience (int): how many epochs to wait after last time the metric improved
        min_delta (float): minimum change in the monitored metric to qualify as an improvement
        mode (str): one of ['min', 'max'] defining whether to monitor for stall in decrease or increase
    """
    def __init__(self, patience=10, min_delta=0, mode='min'):
        self.patience  = patience
        self.min_delta = min_delta
        self.mode      = mode
        self.counter   = 0
        self.best_score = None
        self.early_stop = False

        if mode not in ['min', 'max']:
            raise ValueError("EarlyStopping mode must be 'min' or 'max'")

    def __call__(self, current_score):
        if self.best_score is None:
            self.best_score = current_score
        elif self.mode == 'min':
            if current_score > self.best_score - self.min_delta:
                self.counter += 1
            else:
                self.best_score = current_score
                self.counter = 0
        else: # Mode == 'max'
            if current_score < self.best_score + self.min_delta:
                self.counter += 1
            else:
                self.best_score = current_score
                self.counter = 0

        if self.counter >= self.patience:
            self.early_stop = True
        return self.early_stop
