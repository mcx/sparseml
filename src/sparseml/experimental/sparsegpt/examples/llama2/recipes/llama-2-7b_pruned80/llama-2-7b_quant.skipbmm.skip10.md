---
# Quantization variables

quantization_modifiers:
 - !QuantizationModifier
    ignore:
      - LlamaRotaryEmbedding
      - LlamaRMSNorm
      - SiLUActivation
      - QuantizableMatMul
      - MatMulLeftInput_QK
      - MatMulRightInput_QK
      - MatMulOutput_QK
      - MatMulLeftInput_PV
      - MatMulRightInput_PV
      - MatMulOutput_PV
      - model.layers.0.mlp.down_proj
      - model.layers.18.mlp.down_proj
      - model.layers.29.mlp.down_proj
      - model.layers.15.mlp.down_proj
      - model.layers.4.mlp.down_proj 
      - model.layers.19.mlp.down_proj
      - model.layers.16.mlp.down_proj
      - model.layers.31.mlp.down_proj
      - model.layers.1.mlp.down_proj
      - model.layers.30.mlp.down_proj
    scheme_overrides:
      Embedding:
        input_activations: null
        weights:
          num_bits: 8
          symmetric: False
---