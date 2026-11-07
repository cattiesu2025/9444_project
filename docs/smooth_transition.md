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

## Results and limitations

On the internal SYSU-MM01 validation split (All-search, training seed 17,
averaged over the same 10 fixed galleries), M0 achieved 34.38% mAP and
32.07% Rank-1. The linear transition (M1) reached 34.09% and 31.35%, a
drop of 0.29 and 0.72 percentage points, respectively. The fixed-probability
control (F) reached 34.39% and 31.67%, effectively matching M0 in mAP.
M1's paired mAP difference versus M0 ranged from about −1.0 to +0.5
points across galleries. M0, M1 and F also showed similar drops under
synthetic occlusion and blur. These runs show no clear benefit from
the tested smooth transition. The 10 galleries are repeated evaluations
of one training seed, so they do not establish whether the method
helps or hurts across training runs. See [evaluation.md](evaluation.md).

One possible explanation is that PMT already has a grayscale stage and
uses cross-modal alignment losses. A short taper changes when RGB training
images lose colour, but does not introduce a new alignment objective or
new channel views. The taper may also delay exposure to colour cues at
the start of stage 2. These explanations are hypotheses; the retrieval
metrics alone cannot distinguish them.

The team's corrected CAJ baseline (CAJ_fix) achieved approximately
34.4% mAP and 33.1% Rank-1 on the same internal evaluation. CAJ_fix uses
channel exchange and channel-adaptive grayscale augmentation in its
training pipeline; S changes only the grayscale probability during
PMT's stage-2 transition. CAJ_fix also differs in backbone, losses and
training settings, so its scores cannot isolate the effect of channel
augmentation. The original CAJ run collapsed and was excluded from
the comparison; see the [weekly checkpoint notes](../012_Weekly_Checkpoint_Deliverables.md).
