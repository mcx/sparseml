https://docs.google.com/spreadsheets/d/159NqprZ3gOVAZTb_uQUx_qegfeRdnMlRKA2i6X8MXto/edit#gid=865567375

Include down_proj whose max_abs are <5 in the above spreadsheed
---

quantization_modifiers:
 - !QuantizationModifier
    start_epoch: 0
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
      - model.layers.1.mlp.down_proj
      - model.layers.4.mlp.down_proj
      - model.layers.15.mlp.down_proj
      - model.layers.16.mlp.down_proj
      - model.layers.19.mlp.down_proj
      - model.layers.29.mlp.down_proj
      - model.layers.30.mlp.down_proj
      - model.layers.31.mlp.down_proj
    scheme_overrides:
      Embedding:
        input_activations: null
        weights:
          num_bits: 8
          symmetric: False
---