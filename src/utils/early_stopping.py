"""Early Stopping utilities for Phase 1 and Phase 2 training"""

import torch
import copy


class Phase1EarlyStopping:
    """Early Stopping for Phase 1 (Fixed-Point Context Learning)

    Stopping criteria:
    - Convergence rate >= threshold (e.g., 95% of tokens converged)
    - OR no improvement in convergence rate for `patience` epochs

    Args:
        convergence_threshold: Target convergence rate (default: 0.95)
        patience: Number of epochs to wait for improvement (default: 3)
        min_delta: Minimum change to qualify as improvement (default: 0.01)
    """

    def __init__(self, convergence_threshold=0.95, patience=3, min_delta=0.01):
        self.convergence_threshold = convergence_threshold
        self.patience = patience
        self.min_delta = min_delta

        self.best_convergence = 0.0
        self.epochs_no_improve = 0
        self.should_stop = False

    def __call__(self, convergence_rate):
        """Check if training should stop

        Args:
            convergence_rate: Current convergence rate (0.0 - 1.0)

        Returns:
            bool: True if training should stop
        """
        # Check if target convergence reached
        if convergence_rate >= self.convergence_threshold:
            self.should_stop = True
            return True

        # Check if convergence improved
        if convergence_rate > self.best_convergence + self.min_delta:
            self.best_convergence = convergence_rate
            self.epochs_no_improve = 0
        else:
            self.epochs_no_improve += 1

        # Check patience
        if self.epochs_no_improve >= self.patience:
            self.should_stop = True
            return True

        return False

    def state_dict(self):
        """Return state for checkpoint saving"""
        return {
            'best_convergence': self.best_convergence,
            'epochs_no_improve': self.epochs_no_improve,
            'should_stop': self.should_stop
        }

    def load_state_dict(self, state_dict):
        """Load state from checkpoint"""
        self.best_convergence = state_dict['best_convergence']
        self.epochs_no_improve = state_dict['epochs_no_improve']
        self.should_stop = state_dict['should_stop']


class Phase2EarlyStopping:
    """Early Stopping for Phase 2 (Token Prediction Training)

    Stopping criteria:
    - Validation loss doesn't improve for `patience` epochs
    - OR validation perplexity <= ppl_threshold (if specified)

    Args:
        patience: Number of epochs to wait for improvement (default: 5)
        min_delta: Minimum change to qualify as improvement (default: 0.001)
        ppl_threshold: Target perplexity (optional, stops when reached)
        restore_best: Restore model to best checkpoint when stopping (default: True)
    """

    def __init__(self, patience=5, min_delta=0.001, ppl_threshold=None, restore_best=True):
        self.patience = patience
        self.min_delta = min_delta
        self.ppl_threshold = ppl_threshold
        self.restore_best = restore_best

        self.best_loss = float('inf')
        self.best_ppl = float('inf')
        self.epochs_no_improve = 0
        self.best_model_state = None
        self.should_stop = False

    def __call__(self, val_loss, val_ppl, model):
        """Check if training should stop

        Args:
            val_loss: Current validation loss
            val_ppl: Current validation perplexity
            model: Model to save/restore (nn.Module)

        Returns:
            bool: True if training should stop
        """
        # Check if target perplexity reached
        if self.ppl_threshold and val_ppl <= self.ppl_threshold:
            self.should_stop = True
            return True

        # Check if validation loss improved
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.best_ppl = val_ppl
            self.epochs_no_improve = 0

            # Save best model state
            if self.restore_best:
                self.best_model_state = copy.deepcopy(model.state_dict())
        else:
            self.epochs_no_improve += 1

        # Check patience
        if self.epochs_no_improve >= self.patience:
            self.should_stop = True

            # Restore best model
            if self.restore_best and self.best_model_state is not None:
                model.load_state_dict(self.best_model_state)
                print(f"  → Restored best model (val_loss={self.best_loss:.4f}, val_ppl={self.best_ppl:.2f})")

            return True

        return False

    def state_dict(self):
        """Return state for checkpoint saving"""
        return {
            'best_loss': self.best_loss,
            'best_ppl': self.best_ppl,
            'epochs_no_improve': self.epochs_no_improve,
            'best_model_state': self.best_model_state,
            'should_stop': self.should_stop
        }

    def load_state_dict(self, state_dict):
        """Load state from checkpoint"""
        self.best_loss = state_dict['best_loss']
        self.best_ppl = state_dict['best_ppl']
        self.epochs_no_improve = state_dict['epochs_no_improve']
        self.best_model_state = state_dict['best_model_state']
        self.should_stop = state_dict['should_stop']


class CombinedEarlyStopping:
    """Combined Early Stopping for both Phase 1 and Phase 2

    Convenience class that wraps both Phase1 and Phase2 early stopping.

    Args:
        phase1_kwargs: Keyword arguments for Phase1EarlyStopping
        phase2_kwargs: Keyword arguments for Phase2EarlyStopping
    """

    def __init__(self, phase1_kwargs=None, phase2_kwargs=None):
        phase1_kwargs = phase1_kwargs or {}
        phase2_kwargs = phase2_kwargs or {}

        self.phase1 = Phase1EarlyStopping(**phase1_kwargs)
        self.phase2 = Phase2EarlyStopping(**phase2_kwargs)

    def check_phase1(self, convergence_rate):
        """Check Phase 1 stopping"""
        return self.phase1(convergence_rate)

    def check_phase2(self, val_loss, val_ppl, model):
        """Check Phase 2 stopping"""
        return self.phase2(val_loss, val_ppl, model)

    def state_dict(self):
        """Return state for checkpoint saving"""
        return {
            'phase1': self.phase1.state_dict(),
            'phase2': self.phase2.state_dict()
        }

    def load_state_dict(self, state_dict):
        """Load state from checkpoint"""
        self.phase1.load_state_dict(state_dict['phase1'])
        self.phase2.load_state_dict(state_dict['phase2'])
