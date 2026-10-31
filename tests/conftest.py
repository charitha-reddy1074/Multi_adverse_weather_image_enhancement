"""Avoid CPU thread oversubscription on tiny correctness fixtures."""
import torch


def pytest_sessionstart(session):
    torch.set_num_threads(min(2, torch.get_num_threads()))
