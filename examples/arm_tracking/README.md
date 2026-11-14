# Record visible right-arm motion for offline PDE/GICP evaluation

This fork adds a NumPy-only recorder for the existing Assistive Gym policy loop.
It does not change the policy, simulator physics, camera setup, or garment
visibility, and does not apply corrected actions. Upstream simulation code is
otherwise unchanged.

The numerical tracker, synthetic example, dependencies and tests live in the
[companion package](https://github.com/akira-takakura/dual_arm_dressing/tree/share/assistive-gym-tracking/simulation/arm_tracking).
That package implements shoulder heat diffusion, lowest-96-point ROI selection,
and fixed/policy-reference SE(3)/translation-only GICP comparisons.

## 1. Use the existing simulator environment to record

Do not upgrade the old simulator environment just to install the numerical
tracker. The standalone recorder needs only its existing NumPy installation.
Keep the custom upstream PyBullet build and existing model/asset setup.

```bash
git clone --branch share/arm-tracking-bridge https://github.com/akira-takakura/fmvp_pb.git
cd fmvp_pb
export PYTHONPATH="$PWD/examples/arm_tracking:$PYTHONPATH"
# Continue with the upstream environment/prepare.sh and usual policy command.
```

Add the following to your existing simulation loop (after reset/camera setup):

```python
from record_arm_motion import VisibleArmRecorder

recorder = VisibleArmRecorder("/tmp/arm-recording")
# Optional source frame, before the first action:
recorder.capture(env, simulation_time_s=0.0, policy_update=True)

# In the existing policy loop, after env.step(action):
# Use actual accumulated simulation time, including frame_skip/substeps.
recorder.capture(
    env,
    simulation_time_s=simulation_time_s,
    policy_update=new_policy_chunk_was_predicted,
)
```

`policy_update` refers to the frame used to predict a new action chunk; record
that source frame when it occurs. Do not mark every servo substep as a new
policy observation. Do not step/reset the environment to make a recording.

The recorder dynamically reads `env.human.body`, `right_upperarm`,
`right_forearm`, `right_hand`, and `j_right_shoulder_x`. It calls the existing
`env.get_camera_image_depth()` and `human.get_pos_orient()` methods. It retains
the current clothing occlusions. Unlike the upstream point-cloud function, it
does not hide the cloth or assume the human's Bullet body ID is 2.

## 2. Replay in a separate numerical environment

```bash
git clone --branch share/assistive-gym-tracking https://github.com/akira-takakura/dual_arm_dressing.git
cd dual_arm_dressing
python3.10 -m venv /tmp/dressing-motion-venv
. /tmp/dressing-motion-venv/bin/activate
pip install -r simulation/arm_tracking/requirements-tested.txt
pip install --no-deps -e simulation/arm_tracking
dressing-motion replay /tmp/arm-recording --out /tmp/arm-comparison.json
```

No simulator models, real-person recordings, or checkpoints are needed for the
companion package's included synthetic example.

## Evaluation and limits

Frames contain visible arm points, per-point link IDs, simulation time, a
simulator shoulder cue and simultaneous world-frame link transforms. Only the
visible arm points and shoulder cue enter tracking. Link poses/IDs provide a
separate rigid-link proxy for evaluating ROI point motion. A deformable or
skinned human surface is not exact rigid-link ground truth.

Visible segmentation here is a simulator oracle, not a learned RGB mask. The
package's synthetic example and primitive headless Bullet adapter test were
validated; a full FMVP clothing rollout has not been validated in this change.
Required upstream dependencies, checkpoints and human/garment assets must be
obtained and configured separately under their original terms.

The recorder is a snapshot of the companion package's
`src/dressing_motion/bridge.py`, plus `VisibleArmRecorder`. Keep the schemas in
sync when updating it. All existing upstream attribution remains applicable.
