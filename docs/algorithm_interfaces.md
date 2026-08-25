# Prior And SLAM Plugin Interfaces

After installing:

```bash
pip install sonogym-reconstruction-core
```

external algorithm code should inherit the core interfaces:

```python
from spinal_surgery.interfaces import (
    GoalCommand,
    PriorEstimate,
    PriorExtractor,
    ReconstructionState,
    SLAMPlanner,
)
```

These interfaces are safe to import without starting Isaac Sim. Running the
environment still requires IsaacLab's `AppLauncher`.

## Prior Extractor

Implement `PriorExtractor.update`. It receives a `ReconstructionState` and
returns a `PriorEstimate`.

```python
import torch
from spinal_surgery.interfaces import PriorEstimate, PriorExtractor, ReconstructionState


class MyPrior(PriorExtractor):
    name = "my_prior"

    def reset(self) -> None:
        self.last_confidence = None

    def update(self, state: ReconstructionState) -> PriorEstimate:
        rec = state.human_rec_volume.float()
        # Replace this with registration, atlas fitting, Bayesian filtering, etc.
        volume = (rec > 0).float()
        confidence = torch.ones((rec.shape[0],), device=rec.device) * 0.5
        self.last_confidence = confidence
        return PriorEstimate(
            volume=volume,
            confidence=confidence,
            metadata={"method": self.name},
        )
```

Expected prior shapes:

```text
volume: (B, X, Y, Z) or (X, Y, Z)
xz:     (B, X, Z)    or (X, Z)
```

`confidence` should be a scalar or `(B,)` tensor in `[0, 1]`. For example,
registration prior can map ICP/CPD fitness to confidence.

## SLAM Planner

Implement `SLAMPlanner.plan`. It receives the current state and the optional
prior estimate and returns a `GoalCommand`.

```python
import torch
from spinal_surgery.interfaces import GoalCommand, PriorEstimate, ReconstructionState, SLAMPlanner


class MySLAM(SLAMPlanner):
    name = "my_slam"

    def plan(
        self,
        state: ReconstructionState,
        prior: PriorEstimate | None = None,
    ) -> GoalCommand:
        cur = state.cur_cmd_state.float()
        goal = cur[:, :4].clone()
        goal[:, 0] = goal[:, 0] + 5.0  # x
        goal[:, 1] = goal[:, 1] + 0.0  # z
        goal[:, 2] = 1.57              # yaw
        goal[:, 3] = 0.0               # roll
        return GoalCommand(goal, metadata={"method": self.name})
```

Goal convention:

```text
(B, 3): x, z, yaw
(B, 4): x, z, yaw, roll
```

## Rollout Usage

Inside an IsaacLab rollout script:

```python
from isaaclab.app import AppLauncher

app = AppLauncher(headless=True).app
import gymnasium as gym
import spinal_surgery
from isaaclab_tasks.utils import parse_env_cfg

from my_algorithms import MyPrior, MySLAM

env_cfg = parse_env_cfg("Isaac-robot-US-reconstruction-v0", device="cuda:0", num_envs=1)
env = gym.make("Isaac-robot-US-reconstruction-v0", cfg=env_cfg)
task_env = env.unwrapped

planner = MySLAM(prior_extractor=MyPrior())
planner.setup(task_env)

_, info = env.reset()
planner.reset()
for step in range(500):
    goal_cmd_pose = planner.goal(info["cur_cmd_state"], step, info)
    # Use the existing core controller to turn goal_cmd_pose into actions.
```

The existing `HeuristicReconstruction.get_action_given_goal(info, goal_cmd_pose)`
can execute the planner goal.

## Recommended Separation

Keep the core environment installed from pip:

```text
sonogym-reconstruction-core
```

Keep experiments in a separate repository:

```text
my_sonogym_algorithms/
  my_algorithms/
    priors.py
    slam.py
  scripts/
    evaluate.py
```

This keeps prior extraction and SLAM iteration independent from the environment
package release cycle.
