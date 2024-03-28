import torch
from typing import Dict
from pydantic import BaseModel


class QuantizationArgs(BaseModel):
    n_bits: int
    group_size: int
    symmetric: bool
    observer: str
    observer_kwargs: Dict

# ref: https://pytorch.org/docs/stable/generated/torch.ao.quantization.fake_quantize.FakeQuantize.html
def fake_quantize(
    x: torch.Tensor,
    scale: torch.Tensor,
    zero_point: torch.Tensor,
    args: QuantizationArgs
) -> torch.Tensor:
    quant_min = 0
    quant_max = 2 ** args.n_bits - 1
    x_q = quantize(x, scale, zero_point, quant_min, quant_max)
    return dequantize(x_q, scale, zero_point )
    


def quantize(
    x: torch.Tensor,
    scale: torch.Tensor,
    zero_point: torch.Tensor,
    quant_min: torch.Tensor,
    quant_max: torch.Tensor,
) -> torch.Tensor:
    return torch.clamp(
        round(
            x / scale + zero_point, 
            quant_min,
            quant_max,
        )
    )
    
    
def dequantize(
    x_q: torch.Tensor,
    scale: torch.Tensor,
    zero_point: torch.Tensor,
) -> torch.Tensor:
    return (x_q - zero_point )* scale