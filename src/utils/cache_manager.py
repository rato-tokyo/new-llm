"""Cache Manager for Fixed-Point Contexts

Manages storage and retrieval of fixed-point contexts for multiple samples.
"""

import torch
import hashlib
import json
import os
from pathlib import Path


class FixedContextCache:
    """Cache manager for fixed-point contexts

    Directory structure:
        cache/fixed_contexts/
        ├── index.json
        └── [architecture]_[sample_hash].pt

    Args:
        cache_dir: Directory to store cache files (default: './cache/fixed_contexts')
    """

    def __init__(self, cache_dir='./cache/fixed_contexts'):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.index_path = self.cache_dir / 'index.json'
        self.index = self._load_index()

    def _load_index(self):
        """Load cache index"""
        if self.index_path.exists():
            with open(self.index_path, 'r') as f:
                return json.load(f)
        return {'entries': []}

    def _save_index(self):
        """Save cache index"""
        with open(self.index_path, 'w') as f:
            json.dump(self.index, f, indent=2)

    def get_sample_hash(self, token_ids):
        """Generate hash for token sequence

        Args:
            token_ids: Tensor of token IDs

        Returns:
            str: 8-character hash
        """
        if isinstance(token_ids, torch.Tensor):
            token_ids = token_ids.cpu().numpy()
        token_str = ','.join(map(str, token_ids.flatten()))
        return hashlib.md5(token_str.encode()).hexdigest()[:8]

    def get_cache_path(self, architecture, sample_hash):
        """Get cache file path

        Args:
            architecture: Architecture description (e.g., "Mixed [2,2]")
            sample_hash: Sample hash

        Returns:
            Path: Cache file path
        """
        # Sanitize architecture name for filename
        arch_name = architecture.replace(' ', '_').replace('[', '').replace(']', '').replace(',', '_')
        return self.cache_dir / f"{arch_name}_{sample_hash}.pt"

    def save(self, token_ids, fixed_contexts, converged, architecture, model_config):
        """Save fixed-point contexts to cache

        Args:
            token_ids: Tensor of token IDs
            fixed_contexts: Fixed-point contexts
            converged: Convergence flags
            architecture: Architecture description
            model_config: Model configuration dict
        """
        sample_hash = self.get_sample_hash(token_ids)
        cache_path = self.get_cache_path(architecture, sample_hash)

        cache_data = {
            'token_ids': token_ids.cpu(),
            'fixed_contexts': fixed_contexts.cpu(),
            'converged': converged.cpu(),
            'architecture': architecture,
            'model_config': model_config,
            'sample_hash': sample_hash
        }

        torch.save(cache_data, cache_path)

        # Update index
        entry = {
            'sample_hash': sample_hash,
            'architecture': architecture,
            'path': str(cache_path),
            'num_tokens': len(token_ids),
            'convergence_rate': converged.float().mean().item()
        }

        # Remove old entry if exists
        self.index['entries'] = [e for e in self.index['entries']
                                 if not (e['sample_hash'] == sample_hash and
                                        e['architecture'] == architecture)]
        self.index['entries'].append(entry)
        self._save_index()

        print(f"  Cached fixed contexts: {cache_path.name}")

    def load(self, token_ids, architecture, model_config):
        """Load fixed-point contexts from cache

        Args:
            token_ids: Tensor of token IDs
            architecture: Architecture description
            model_config: Model configuration dict

        Returns:
            tuple: (fixed_contexts, converged) or (None, None) if not found
        """
        sample_hash = self.get_sample_hash(token_ids)
        cache_path = self.get_cache_path(architecture, sample_hash)

        if not cache_path.exists():
            return None, None

        cache_data = torch.load(cache_path)

        # Verify token_ids match
        if not torch.equal(cache_data['token_ids'], token_ids.cpu()):
            print(f"  Warning: Token IDs mismatch, ignoring cache")
            return None, None

        # Verify architecture matches
        if cache_data['architecture'] != architecture:
            print(f"  Warning: Architecture mismatch, ignoring cache")
            return None, None

        # Verify critical config matches
        for key in ['context_dim', 'hidden_dim', 'vocab_size', 'embed_dim']:
            if key in model_config and key in cache_data['model_config']:
                if model_config[key] != cache_data['model_config'][key]:
                    print(f"  Warning: Config mismatch ({key}), ignoring cache")
                    return None, None

        print(f"  Loaded cached fixed contexts: {cache_path.name}")
        return cache_data['fixed_contexts'], cache_data['converged']

    def exists(self, token_ids, architecture):
        """Check if cache exists for given token sequence and architecture

        Args:
            token_ids: Tensor of token IDs
            architecture: Architecture description

        Returns:
            bool: True if cache exists
        """
        sample_hash = self.get_sample_hash(token_ids)
        cache_path = self.get_cache_path(architecture, sample_hash)
        return cache_path.exists()

    def clear(self, architecture=None):
        """Clear cache

        Args:
            architecture: If specified, only clear caches for this architecture
        """
        if architecture is None:
            # Clear all
            for entry in self.index['entries']:
                cache_path = Path(entry['path'])
                if cache_path.exists():
                    cache_path.unlink()
            self.index['entries'] = []
        else:
            # Clear specific architecture
            remaining = []
            for entry in self.index['entries']:
                if entry['architecture'] == architecture:
                    cache_path = Path(entry['path'])
                    if cache_path.exists():
                        cache_path.unlink()
                else:
                    remaining.append(entry)
            self.index['entries'] = remaining

        self._save_index()
        print(f"Cache cleared: {architecture if architecture else 'all'}")

    def list_cached(self):
        """List all cached entries

        Returns:
            list: List of cache entries
        """
        return self.index['entries']

    def stats(self):
        """Print cache statistics"""
        print("\n" + "="*70)
        print("Fixed Context Cache Statistics")
        print("="*70)

        if not self.index['entries']:
            print("Cache is empty")
            return

        by_arch = {}
        for entry in self.index['entries']:
            arch = entry['architecture']
            if arch not in by_arch:
                by_arch[arch] = []
            by_arch[arch].append(entry)

        for arch, entries in by_arch.items():
            print(f"\n{arch}:")
            total_tokens = sum(e['num_tokens'] for e in entries)
            avg_conv = sum(e['convergence_rate'] for e in entries) / len(entries)
            print(f"  Cached samples: {len(entries)}")
            print(f"  Total tokens: {total_tokens:,}")
            print(f"  Avg convergence: {avg_conv:.1%}")

        print("="*70)
