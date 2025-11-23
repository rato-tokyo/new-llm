"""
CVFP (Context Vector Fixed-Point) モジュール

CVFPアーキテクチャの中核コンポーネント:
- CVFPLayer: 基本的なコンテキスト更新ユニット
- CVFPBlock: 複数のCVFPLayerをグループ化
"""

from .layer import CVFPLayer
from .block import CVFPBlock

__all__ = ['CVFPLayer', 'CVFPBlock']
