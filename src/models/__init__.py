"""New-LLM Models - Residual Standard Architecture Only

Unified to Residual Standard architecture with CVFP (Context Vector Fixed-Point) training.
All other architectures (Sequential, Layerwise, Flexible) have been deprecated.
"""

from .new_llm_residual import NewLLMResidual

__all__ = ['NewLLMResidual']
