# C: Smooth Grayscale Transition

## Method
The original PMT baseline switches RGB training images from grayscale
to colour immediately at the start of stage 2. The proposed modification
gradually reduces the grayscale probability during a transition window.

For stage-2 batch r within a window of W batches:

p(r) = p0 × (1 − r / (W − 1))

The probability becomes zero after the window. A fixed-probability
control uses p0/2 over the same window, matching the expected grayscale
exposure of the linear schedule. Actual grayscale counts may differ
because the conversion is sampled randomly.

Only RGB images receive this augmentation. IR images and the PMT loss
stage-switching rules remain unchanged.

## Implementation and Checks
- Implemented the schedule in src/schedules.py.
- Connected M1 and F configurations to the training entry point.
- All 53 baseline, data and schedule tests passed on Mac.
- A simulated sampler check produced:
  - p=1.0: RGB grayscale 4/4, IR grayscale 0/4.
  - p=0.5: RGB grayscale 1/4, IR grayscale 0/4.
  - p=0.0: RGB grayscale 0/4, IR grayscale 0/4.
- Schedule checks covered probability boundaries, equal expected
  exposure and reconstruction from the saved configuration and step.

## Conclusion and Limitations
The checks support the correctness of the probability schedule and its
connection to the sampler. The implementation provides a controlled way
to compare an immediate switch, a gradual transition and a fixed-
probability transition.

These are implementation checks, not evidence of improved retrieval
performance. End-to-end training, checkpoint-resume verification and
Rank-1/mAP comparisons between M0, M1 and F remain to be completed.
