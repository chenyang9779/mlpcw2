import torch
import pytest
from model_architectures import *

def test_batch_normalization_block():
    input_shape = (1, 3, 64, 64)  # Batch size 1, 3 channels, 64x64 image
    x = torch.randn(input_shape)
    block = ConvolutionalBNProcessingBlock(input_shape=input_shape, num_filters=16, kernel_size=3, padding=1, bias=False, dilation=1, use_residual=False)
    try:
        output = block(x)
        print("BatchNormalizationBlock forward pass successful. Output shape:", output.shape)
    except Exception as e:
        print("BatchNormalizationBlock forward pass failed with error:", e)

def test_batch_normalization_residual_connections_block():
    input_shape = (1, 3, 64, 64)  # Batch size 1, 3 channels, 64x64 image
    x = torch.randn(input_shape)
    block = ConvolutionalBNDimensionalityReductionBlock(input_shape=input_shape, num_filters=16, kernel_size=3, padding=1, bias=False, dilation=1, reduction_factor=2)
    try:
        output = block(x)
        print("ResidualConnectionsBlock forward pass successful. Output shape:", output.shape)
    except Exception as e:
        print("ResidualConnectionsBlock forward pass failed with error:", e)

if __name__ == "__main__":
    test_batch_normalization_block()
    test_batch_normalization_residual_connections_block()