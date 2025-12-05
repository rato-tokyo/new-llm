"""
New-LLM: MLA-Pythia for KV Cache Compression

Pythia-70MベースのMLA (Multi-head Latent Attention) 実装。
KVキャッシュを87.5%削減。

主要コンポーネント:
- UnifiedPythiaModel: 位置エンコーディング切替可能な統一モデル (RoPE/ALiBi/NoPE)
- MLAPythiaModel: KVキャッシュ圧縮モデル (ALiBi)
- PythiaModel: ベースラインモデル (RoPE)
"""
