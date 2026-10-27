"""Deterministic grayscale probabilities for the PMT transition.

Steps count consumed batches (including AMP skipped updates), as the sampler does.
"""
from dataclasses import dataclass
import math


@dataclass(frozen=True)
class GraySchedule:
    mode: str
    stage1_steps: int
    window_steps: int
    initial_probability: float = 1.0

    def __post_init__(self):
        if self.mode not in ('off', 'linear', 'fixed_expected_exposure'):
            raise ValueError('Unknown grayscale schedule mode')
        if self.stage1_steps < 0 or self.window_steps < 2:
            raise ValueError('stage1_steps must be nonnegative and window_steps >= 2')
        if not math.isfinite(self.initial_probability) or not 0 <= self.initial_probability <= 1:
            raise ValueError('initial_probability must be between 0 and 1')

    def __call__(self, global_step: int) -> float:
        if not isinstance(global_step, int) or global_step < 0:
            raise ValueError('global_step must be a nonnegative integer')
        if global_step < self.stage1_steps:
            return 1.0
        r = global_step - self.stage1_steps
        if self.mode == 'off' or r >= self.window_steps:
            return 0.0
        if self.mode == 'fixed_expected_exposure':
            return self.initial_probability / 2
        return self.initial_probability * (1 - r / (self.window_steps - 1))


def from_config(config, batches_per_epoch):
    """Build a stateless schedule from the checkpointed config and sampler size."""
    if not isinstance(batches_per_epoch, int) or batches_per_epoch < 1:
        raise ValueError('batches_per_epoch must be a positive integer')
    smooth = config['smooth']
    return GraySchedule(smooth['mode'], config['training']['stage1_epochs'] * batches_per_epoch,
                        smooth['window_epochs'] * batches_per_epoch, smooth['initial_probability'])


if __name__ == '__main__':
    for mode in ('off', 'linear', 'fixed_expected_exposure'):
        schedule = GraySchedule(mode, stage1_steps=2, window_steps=5)
        print(mode, [round(schedule(step), 3) for step in range(9)])
