"""
Memory Mixins - メモリ機能の拡張用Mixin

後方互換性のために残すが、現在はCompressiveMemoryに統合済み。
将来的な拡張用に空のMixinを用意。
"""


class FreezableMemoryMixin:
    """Freeze/Unfreeze機能のMixin

    Note: 現在はCompressiveMemoryに直接実装済み。
    将来的な拡張のためのプレースホルダー。
    """
    pass


class MemoryExportMixin:
    """Export/Import機能のMixin

    Note: 現在はCompressiveMemoryに直接実装済み。
    将来的な拡張のためのプレースホルダー。
    """
    pass
