---

quantization_modifiers:
 - !QuantizationModifier
    ignore:
      - LlamaRotaryEmbedding
      - LlamaRMSNorm
      - SiLUActivation
      - QuantizableMatMul
      - model.layers.2.mlp.down_proj
      - model.layers.17.mlp.down_proj
      - model.layers.15.mlp.down_proj
      - model.layers.4.mlp.down_proj
      - model.layers.29.mlp.down_proj 
      - model.layers.19.mlp.down_proj
      - model.layers.16.mlp.down_proj
      - model.layers.31.mlp.down_proj
      - model.layers.1.mlp.down_proj
      - model.layers.30.mlp.down_proj
    scheme_overrides:
      Linear:
        weights:
          num_bits: 8
          symmetric: true
          strategy: channel
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