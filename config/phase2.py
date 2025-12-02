"""
Phase 2 設定 (Phase 2 Configuration)

トークン予測学習の設定。
キャッシュ方式による高速化対応。
"""

import torch


class Phase2Config:
    """Phase 2: トークン予測学習の設定"""

    # ========== 学習パラメータ ==========
    learning_rate = 0.001            # 学習率
    epochs = 40                      # 訓練エポック数
    patience = 1                     # Early stopping patience
    gradient_clip = 1.0              # 勾配クリッピング値

    # ========== PPL改善閾値 ==========
    min_ppl_improvement = 0.4        # PPL改善がこの値未満なら早期停止
                                     # 例: 133.0 → 132.6 (0.4改善) はギリギリ継続
                                     # 例: 133.0 → 132.8 (0.2改善) は停止

    # ========== バッチサイズ ==========
    batch_size = None                # Noneで自動計算（GPUメモリベース）
    min_batch_size = 256             # 最小バッチサイズ
    max_batch_size = 16384           # 最大バッチサイズ

    # ========== メモリ管理 ==========
    memory_safety_factor = 0.5       # メモリ安全係数（0.0-1.0）
                                     # 0.5: 推奨（空きメモリの50%を使用）
                                     # 0.7: 積極的（OOMリスク増）
                                     # 0.3: 保守的（遅いが安全）

    # ========== Embedding凍結 ==========
    freeze_embedding = True          # Embedding凍結（標準採用）
                                     # PPL 66-72%改善、Accuracy 53-63%改善
                                     # Weight Tying時はOutput Headも凍結される

    @classmethod
    def get_effective_batch_size(cls, config_batch_size=None):
        """Phase 2のバッチサイズを取得（GPUメモリベース自動計算）"""
        if config_batch_size is not None:
            return config_batch_size

        # GPUメモリに基づく自動計算
        if torch.cuda.is_available():
            gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            # 目安: 1GB あたり 500トークン、安全係数0.8
            batch_size = int(gpu_memory_gb * 500 * 0.8)
        else:
            # CPU: 固定値
            batch_size = 512

        return batch_size
