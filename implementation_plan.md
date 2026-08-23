# Sideward Glance False-Positive Fix Plan

This plan addresses the CRITICAL false-positive issue where normal seated students are flagged as `MEDIUM - SIDEWARD GLANCE`. It incorporates temporal smoothing, per-person relative baselining, strict UI separation for confirmed events, and evidence pipeline preservation.

## User Review Required

Please review the proposed baseline locking conditions and smoothing window. Because we must eliminate pose jitter in low-resolution wide-angle CCTV, I propose a rolling median window of 7 frames (approx 0.7s at 10FPS). Is this acceptable, or would you prefer a wider window?

## Proposed Changes

### `config.py`
We will introduce new configuration parameters to govern the smoothing and baseline locking safely.

#### [MODIFY] `config.py`
- Add `GLANCE_SMOOTHING_FRAMES = 7`: The rolling window size for median smoothing of the raw nose-shoulder ratio.
- Adjust `NOSE_SHOULDER_GLANCE_THRESHOLD = 0.15`: A 1-pixel jitter on a 15-pixel shoulder width equals `0.06`. A threshold of `0.035` is physically impossible to satisfy reliably on small subjects. `0.15` represents roughly a 2-3 pixel deviation (a true head turn).

### `behaviour.py`
This is the core of the fix, completely replacing the absolute geometric thresholding with a robust, smoothed, relative system.

#### [MODIFY] `behaviour.py`
- **Diagnostic Phase Results**: The median raw nose-shoulder ratio across your test set is `0.9971`, not `0`. Because the CCTV camera looks down at an angle, the 2D projection of the head is pushed outside the shoulders. The current code has an absolute fallback: `if nsr > (THRESHOLD + 0.05)`, which evaluates to `if 0.99 > 0.085`, flagging literally everyone as turning *before* the baseline even locks.
- **Remove Absolute Fallback**: I will remove the absolute threshold entirely. If a track's baseline is not yet locked, it cannot trigger a deviation anomaly.
- **Implement Rolling Smoothing**: I will add a `history_raw_yaw` deque to the track state (independent of the calibration history). For every frame, the raw `nsr` is appended. The *current* `nsr` used for anomaly detection will be the `median(history_raw_yaw)`. This mathematically eliminates 1-frame/2-frame pose jitter spikes.
- **Tighten Baseline Locking**: The current lock condition is `mad_yaw < 0.10`. Since jitter alone is `~0.06`, this is fine, but I will ensure the baseline itself uses the smoothed median.
- **Strict UI Rendering**: The current code renders the `MEDIUM - SIDEWARD GLANCE` highlight based on `turn_act`, which correctly only becomes `True` *after* the persistence timer. The UI logic is fundamentally sound, but it was being fed garbage data because every single frame spiked above the `0.035` deviation due to jitter, causing the persistence timer to complete immediately.

## Verification Plan

### Automated Tests
- I will run a script across the CCTV test frames to prove the 99th percentile of smoothed deviations during normal posture.
- I will verify that the unassociated phone detection remains fully functional and forces a `HIGH` severity.

### Manual Verification
- I will provide the final report detailing exactly how the pipeline evaluates normal vs sideward glances.
- You will run `app.py` on the live CCTV footage to confirm that normal students remain completely unflagged, while sustained head turns are correctly labeled `MEDIUM`.
