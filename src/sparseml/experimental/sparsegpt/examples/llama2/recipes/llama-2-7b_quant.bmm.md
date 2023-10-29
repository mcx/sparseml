---
quantization_modifiers:
 - !QuantizationModifier
    ignore:
      - LlamaRotaryEmbedding
      - LlamaRMSNorm
      - SiLUActivation
      - QuantizableMatMul
      - model.layers.0.mlp.down_proj
      - model.layers.1.mlp.down_proj
      - model.layers.2.mlp.down_proj
      - model.layers.3.mlp.down_proj
      - model.layers.4.mlp.down_proj
      - model.layers.5.mlp.down_proj
      - model.layers.6.mlp.down_proj
      - model.layers.7.mlp.down_proj
      - model.layers.8.mlp.down_proj
      - model.layers.9.mlp.down_proj
      - model.layers.10.mlp.down_proj
      - model.layers.11.mlp.down_proj
      - model.layers.12.mlp.down_proj
      - model.layers.13.mlp.down_proj
      - model.layers.14.mlp.down_proj
      - model.layers.15.mlp.down_proj
      - model.layers.16.mlp.down_proj
      - model.layers.17.mlp.down_proj
      - model.layers.18.mlp.down_proj
      - model.layers.19.mlp.down_proj
      - model.layers.20.mlp.down_proj
      - model.layers.21.mlp.down_proj
      - model.layers.22.mlp.down_proj
      - model.layers.23.mlp.down_proj
      - model.layers.24.mlp.down_proj
      - model.layers.25.mlp.down_proj
      - model.layers.26.mlp.down_proj
      - model.layers.27.mlp.down_proj
      - model.layers.28.mlp.down_proj
      - model.layers.29.mlp.down_proj
      - model.layers.30.mlp.down_proj
      - model.layers.31.mlp.down_proj
    scheme_overrides:
      Embedding:
        input_activations: null
        weights:
          num_bits: 8
          symmetric: False
      MatMulLeftInput_QK:
        output_activations: null
        input_activations:
          num_bits: 8
          symmetric: False
      MatMulRightInput_QK:
        output_activations: null
        input_activations:
          num_bits: 8
          symmetric: True
      MatMulOutput_QK:
        input_activations: null
        output_activations: null
      MatMulLeftInput_PV:
        output_activations: null
        input_activations:
          num_bits: 8
          symmetric: False
      MatMulRightInput_PV:
        output_activations: null
        input_activations:
          num_bits: 8
          symmetric: True
      MatMulOutput_PV:
        input_activations: null
        output_activations: null

---