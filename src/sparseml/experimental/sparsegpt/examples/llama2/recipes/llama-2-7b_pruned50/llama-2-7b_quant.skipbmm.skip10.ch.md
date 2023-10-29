---
# Quantization variables
observer_freeze_epoch: 1
bn_freeze_epoch: 1
qat_start_epoch: 0

quantization_modifiers:
 - !QuantizationModifier
    start_epoch: eval(qat_start_epoch)
    disable_quantization_observer_epoch: eval(observer_freeze_epoch)
    freeze_bn_stats_epoch: eval(bn_freeze_epoch)
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
      - model.layers.1.mlp.down_proj
      - model.layers.2.mlp.down_proj
      - model.layers.4.mlp.down_proj
      - model.layers.15.mlp.down_proj 
      - model.layers.16.mlp.down_proj
      - model.layers.19.mlp.down_proj
      - model.layers.29.mlp.down_proj
      - model.layers.30.mlp.down_proj
      - model.layers.31.mlp.down_proj
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
---