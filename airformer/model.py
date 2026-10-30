"""Backward-compatible import for the modular AIRFormer implementation."""
from models.airformer import AIRFormer
from models.supplementary_prior import dwt2 as haar_dwt, idwt2

__all__ = ["AIRFormer", "haar_dwt", "idwt2"]
