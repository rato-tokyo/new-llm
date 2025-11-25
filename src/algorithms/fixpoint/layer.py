"""
CVFP Layer - 1トークンの基本処理

責任:
- モデルの順伝播呼び出し
- コンテキスト更新の最小単位
"""

import torch


class Layer:
    """
    CVFP処理の最小単位 - 1トークンのコンテキスト更新

    モデルの_update_context_one_step()をラップし、
    gradient managementを明確にする。
    """

    def __init__(self, model):
        """
        Args:
            model: NewLLMResidualモデル
        """
        self.model = model

    def forward(self, token_embed, context, detach_context=True):
        """
        1トークンのコンテキスト更新

        Args:
            token_embed: [1, embed_dim]
            context: [1, context_dim]
            detach_context: コンテキストをdetachするか（トークン間勾配遮断用）

        Returns:
            new_context: [1, context_dim]
        """
        if detach_context:
            context = context.detach()

        return self.model._update_context_one_step(token_embed, context)
