"""
Exponential Moving Average (EMA) for model weights.

Based on DINOv2 and state-of-the-art practices:
- Maintains a slow-moving average of model parameters
- Often provides better generalization than the final trained model
- Typical decay rates: 0.999-0.9999 for large models

Reference:
- DINOv2: https://arxiv.org/abs/2304.07193
- Mean teachers are better role models: https://arxiv.org/abs/1703.01780
"""

import torch
import torch.nn as nn
from copy import deepcopy


class ModelEMA:
    """
    Exponential Moving Average of model parameters.

    Args:
        model: The model to track
        decay: EMA decay rate (default: 0.9999)
        device: Device to store EMA model on
    """

    def __init__(self, model, decay=0.9999, device=None):
        self.model = deepcopy(model).eval()  # Create a copy of the model
        self.decay = decay
        self.device = device

        if self.device is not None:
            self.model.to(device)

        # Freeze EMA parameters - we update them manually
        for param in self.model.parameters():
            param.requires_grad = False

        self.updates = 0  # Number of EMA updates performed

    def update(self, model):
        """
        Update EMA parameters using current model parameters.

        EMA formula: ema_param = decay * ema_param + (1 - decay) * current_param
        """
        with torch.no_grad():
            self.updates += 1

            # Use dynamic decay that starts lower and increases
            # This helps at the beginning of training
            decay = min(self.decay, (1 + self.updates) / (10 + self.updates))

            # Update EMA parameters
            for ema_param, model_param in zip(self.model.parameters(), model.parameters()):
                if self.device is not None:
                    model_param = model_param.to(self.device)
                ema_param.data.mul_(decay).add_(model_param.data, alpha=1 - decay)

    def update_buffers(self, model):
        """
        Update buffer values (like batch norm running mean/var).
        These should be copied directly, not averaged.
        """
        with torch.no_grad():
            for ema_buffer, model_buffer in zip(self.model.buffers(), model.buffers()):
                if self.device is not None:
                    model_buffer = model_buffer.to(self.device)
                ema_buffer.copy_(model_buffer)

    def state_dict(self):
        """Return EMA model state dict."""
        return {
            'model': self.model.state_dict(),
            'updates': self.updates,
            'decay': self.decay
        }

    def load_state_dict(self, state_dict):
        """Load EMA model state dict."""
        self.model.load_state_dict(state_dict['model'])
        self.updates = state_dict['updates']
        self.decay = state_dict['decay']

    def eval(self):
        """Set EMA model to eval mode."""
        return self.model.eval()

    def __call__(self, *args, **kwargs):
        """Forward pass through EMA model."""
        return self.model(*args, **kwargs)


class ModelEMAv2:
    """
    Improved EMA with support for different decay schedules.

    This version supports:
    - Constant decay
    - Linear warmup of decay
    - Separate tracking of batch norm statistics

    Args:
        model: The model to track
        decay: EMA decay rate (default: 0.9999)
        warmup_steps: Number of steps to linearly increase decay from 0 to target (default: 2000)
        device: Device to store EMA model on
    """

    def __init__(self, model, decay=0.9999, warmup_steps=2000, device=None):
        self.ema_model = deepcopy(model).eval()
        self.decay = decay
        self.warmup_steps = warmup_steps
        self.device = device
        self.updates = 0

        if self.device is not None:
            self.ema_model.to(device)

        # Freeze EMA parameters
        for param in self.ema_model.parameters():
            param.requires_grad = False

    def _get_decay(self, step):
        """Calculate decay rate with optional warmup."""
        if step < self.warmup_steps:
            # Linear warmup from 0 to target decay
            return self.decay * (step / self.warmup_steps)
        return self.decay

    def update(self, model, step=None):
        """Update EMA model parameters."""
        with torch.no_grad():
            self.updates += 1
            current_step = step if step is not None else self.updates
            decay = self._get_decay(current_step)

            # Update all parameters
            for ema_param, model_param in zip(self.ema_model.parameters(), model.parameters()):
                if self.device is not None:
                    model_param = model_param.to(self.device)
                ema_param.data.mul_(decay).add_(model_param.data, alpha=1.0 - decay)

            # Update batch norm buffers (running mean/var)
            for ema_buffer, model_buffer in zip(self.ema_model.buffers(), model.buffers()):
                if self.device is not None:
                    model_buffer = model_buffer.to(self.device)
                # For buffers, we typically just copy (not average)
                # But for running stats, we can use the same decay
                ema_buffer.data.mul_(decay).add_(model_buffer.data, alpha=1.0 - decay)

    def state_dict(self):
        """Save EMA state."""
        return {
            'ema_model': self.ema_model.state_dict(),
            'decay': self.decay,
            'warmup_steps': self.warmup_steps,
            'updates': self.updates
        }

    def load_state_dict(self, state_dict):
        """Load EMA state."""
        self.ema_model.load_state_dict(state_dict['ema_model'])
        self.decay = state_dict.get('decay', self.decay)
        self.warmup_steps = state_dict.get('warmup_steps', self.warmup_steps)
        self.updates = state_dict.get('updates', 0)

    def eval(self):
        """Set EMA model to eval mode."""
        return self.ema_model.eval()

    def __call__(self, *args, **kwargs):
        """Forward pass through EMA model."""
        return self.ema_model(*args, **kwargs)

    @property
    def model(self):
        """Access the underlying EMA model."""
        return self.ema_model


# Example usage
if __name__ == "__main__":
    # Create a simple test model
    test_model = nn.Sequential(
        nn.Linear(10, 50),
        nn.ReLU(),
        nn.Linear(50, 2)
    )

    # Create EMA
    ema = ModelEMA(test_model, decay=0.999)

    # Simulate training
    for step in range(100):
        # Fake training step - modify model parameters
        for param in test_model.parameters():
            param.data += torch.randn_like(param.data) * 0.01

        # Update EMA
        ema.update(test_model)

        if step % 20 == 0:
            print(f"Step {step}: EMA updates = {ema.updates}")

    print("\nEMA tracking completed!")
