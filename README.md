# ROB6323 Go2 Project — Isaac Lab

This repository is the starter code for the NYU Reinforcement Learning and Optimal Control project in which students train a Unitree Go2 walking policy in Isaac Lab starting from a minimal baseline and improve it via reward shaping and robustness strategies. Please read this README fully before starting and follow the exact workflow and naming rules below to ensure your runs integrate correctly with the cluster scripts and grading pipeline.

## Repository policy

- Fork this repository and do not change the repository name in your fork.  
- Your fork must be named rob6323_go2_project so cluster scripts and paths work without modification.

### Prerequisites

- **GitHub Account:** You must have a GitHub account to fork this repository and manage your code. If you do not have one, [sign up here](https://github.com/join).

### Links
1.  **Project Webpage:** [https://machines-in-motion.github.io/RL_class_go2_project/](https://machines-in-motion.github.io/RL_class_go2_project/)
2.  **Project Tutorial:** [https://github.com/machines-in-motion/rob6323_go2_project/blob/master/tutorial/tutorial.md](https://github.com/machines-in-motion/rob6323_go2_project/blob/master/tutorial/tutorial.md)

## Connect to Greene

- Connect to the NYU Greene HPC via SSH; if you are off-campus or not on NYU Wi‑Fi, you must connect through the NYU VPN before SSHing to Greene.  
- The official instructions include example SSH config snippets and commands for greene.hpc.nyu.edu and dtn.hpc.nyu.edu as well as VPN and gateway options: https://sites.google.com/nyu.edu/nyu-hpc/accessing-hpc?authuser=0#h.7t97br4zzvip.

## Clone in $HOME

After logging into Greene, `cd` into your home directory (`cd $HOME`). You must clone your fork into `$HOME` only (not scratch or archive). This ensures subsequent scripts and paths resolve correctly on the cluster. Since this is a private repository, you need to authenticate with GitHub. You have two options:

### Option A: Via VS Code (Recommended)
The easiest way to avoid managing keys manually is to configure **VS Code Remote SSH**. If set up correctly, VS Code forwards your local credentials to the cluster.
- Follow the [NYU HPC VS Code guide](https://sites.google.com/nyu.edu/nyu-hpc/training-support/general-hpc-topics/vs-code) to set up the connection.

> **Tip:** Once connected to Greene in VS Code, you can clone directly without using the terminal:
> 1. **Sign in to GitHub:** Click the "Accounts" icon (user profile picture) in the bottom-left sidebar. If you aren't signed in, click **"Sign in with GitHub"** and follow the browser prompts to authorize VS Code.
> 2. **Clone the Repo:** Open the Command Palette (`Ctrl+Shift+P` or `Cmd+Shift+P`), type **Git: Clone**, and select it.
> 3. **Select Destination:** When prompted, select your home directory (`/home/<netid>/`) as the clone location.
>
> For more details, see the [VS Code Version Control Documentation](https://code.visualstudio.com/docs/sourcecontrol/intro-to-git#_clone-a-repository-locally).

### Option B: Manual SSH Key Setup
If you prefer using a standard terminal, you must generate a unique SSH key on the Greene cluster and add it to your GitHub account:
1. **Generate a key:** Run the `ssh-keygen` command on Greene (follow the official [GitHub documentation on generating a new SSH key](https://docs.github.com/en/authentication/connecting-to-github-with-ssh/generating-a-new-ssh-key-and-adding-it-to-the-ssh-agent#generating-a-new-ssh-key)).
2. **Add the key to GitHub:** Copy the output of your public key (e.g., `cat ~/.ssh/id_ed25519.pub`) and add it to your account settings (follow the [GitHub documentation on adding a new SSH key](https://docs.github.com/en/authentication/connecting-to-github-with-ssh/adding-a-new-ssh-key-to-your-github-account)).

### Execute the Clone
Once authenticated, run the following commands. Replace `<your-git-ssh-url>` with the SSH URL of your fork (e.g., `git@github.com:YOUR_USERNAME/rob6323_go2_project.git`).
```
cd $HOME
git clone <your-git-ssh-url> rob6323_go2_project
```
*Note: You must ensure the target directory is named exactly `rob6323_go2_project`. This ensures subsequent scripts and paths resolve correctly on the cluster.*
## Install environment

- Enter the project directory and run the installer to set up required dependencies and cluster-side tooling.  
```
cd $HOME/rob6323_go2_project
./install.sh
```
Do not skip this step, as it configures the environment expected by the training and evaluation scripts. It will launch a job in burst to set up things and clone the IsaacLab repo inside your greene storage. You must wait until the job in burst is complete before launching your first training. To check the progress of the job, you can run `ssh burst "squeue -u $USER"`, and the job should disappear from there once it's completed. It takes around **30 minutes** to complete. 
You should see something similar to the screenshot below (captured from Greene):

![Example burst squeue output](docs/img/burst_squeue_example.png)

In this output, the **ST** (state) column indicates the job status:
- `PD` = pending in the queue (waiting for resources).
- `CF` = instance is being configured.
- `R`  = job is running.

On burst, it is common for an instance to fail to configure; in that case, the provided scripts automatically relaunch the job when this happens, so you usually only need to wait until the job finishes successfully and no longer appears in `squeue`.

## What to edit

- In this project you'll only have to modify the two files below, which define the Isaac Lab task and its configuration (including PPO hyperparameters).  
  - source/rob6323_go2/rob6323_go2/tasks/direct/rob6323_go2/rob6323_go2_env.py  
  - source/rob6323_go2/rob6323_go2/tasks/direct/rob6323_go2/rob6323_go2_env_cfg.py
PPO hyperparameters are defined in source/rob6323_go2/rob6323_go2/tasks/direct/rob6323_go2/agents/rsl_rl_ppo_cfg.py, but you shouldn't need to modify them.

## How to edit

- Option A (recommended): Use VS Code Remote SSH from your laptop to edit files on Greene; follow the NYU HPC VS Code guide and connect to a compute node as instructed (VPN required off‑campus) (https://sites.google.com/nyu.edu/nyu-hpc/training-support/general-hpc-topics/vs-code). If you set it correctly, it makes the login process easier, among other things, e.g., cloning a private repo.
- Option B: Edit directly on Greene using a terminal editor such as nano.  
```
nano source/rob6323_go2/rob6323_go2/tasks/direct/rob6323_go2/rob6323_go2_env.py
```
- Option C: Develop locally on your machine, push to your fork, then pull changes on Greene within your $HOME/rob6323_go2_project clone.

> **Tip:** Don't forget to regularly push your work to github

## Launch training

- From $HOME/rob6323_go2_project on Greene, submit a training job via the provided script.  
```
cd "$HOME/rob6323_go2_project"
./train.sh
```
- Check job status with SLURM using squeue on the burst head node as shown below.  
```
ssh burst "squeue -u $USER"
```
Be aware that jobs can be canceled and requeued by the scheduler or underlying provider policies when higher-priority work preempts your resources, which is normal behavior on shared clusters using preemptible partitions.

## Where to find results

- When a job completes, logs are written under logs in your project clone on Greene in the structure logs/[job_id]/rsl_rl/go2_flat_direct/[date_time]/.  
- Inside each run directory you will find a TensorBoard events file (events.out.tfevents...), neural network checkpoints (model_[epoch].pt), YAML files with the exact PPO and environment parameters, and a rollout video under videos/play/ that showcases the trained policy.  

## Download logs to your computer

Use `rsync` to copy results from the cluster to your local machine. It is faster and can resume interrupted transfers. Run this on your machine (NOT on Greene):

```
rsync -avzP -e 'ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null' <netid>@dtn.hpc.nyu.edu:/home/<netid>/rob6323_go2_project/logs ./
```

*Explanation of flags:*
- `-a`: Archive mode (preserves permissions, times, and recursive).
- `-v`: Verbose output.
- `-z`: Compresses data during transfer (faster over network).
- `-P`: Shows progress bar and allows resuming partial transfers.

## Visualize with TensorBoard

You can inspect training metrics (reward curves, loss values, episode lengths) using TensorBoard. This requires installing it on your local machine.

1.  **Install TensorBoard:**
    On your local computer (do NOT run this on Greene), install the package:
    ```
    pip install tensorboard
    ```

2.  **Launch the Server:**
    Navigate to the folder where you downloaded your logs and start the server:
    ```
    # Assuming you are in the directory containing the 'logs' folder
    tensorboard --logdir ./logs
    ```

3.  **View Metrics:**
    Open your browser to the URL shown (usually `http://localhost:6006/`).

## Debugging on Burst

Burst storage is accessible only from a job running on burst, not from the burst login node. The provided scripts do not automatically synchronize error logs back to your home directory on Greene. However, you will need access to these logs to debug failed jobs. These error logs differ from the logs in the previous section.

The suggested way to inspect these logs is via the Open OnDemand web interface:

1.  Navigate to [https://ood-burst-001.hpc.nyu.edu](https://ood-burst-001.hpc.nyu.edu).
2.  Select **Files** > **Home Directory** from the top menu.
3.  You will see a list of files, including your `.err` log files.
4.  Click on any `.err` file to view its content directly in the browser.

> **Important:** Do not modify anything inside the `rob6323_go2_project` folder on burst storage. This directory is managed by the job scripts, and manual changes may cause synchronization issues or job failures.

## Project scope reminder

- The assignment expects you to go beyond velocity tracking by adding principled reward terms (posture stabilization, foot clearance, slip minimization, smooth actions, contact and collision penalties), robustness via domain randomization, and clear benchmarking metrics for evaluation as described in the course guidelines.  
- Keep your repository organized, document your changes in the README, and ensure your scripts are reproducible, as these factors are part of grading alongside policy quality and the short demo video deliverable.

## Resources

- [Isaac Lab documentation](https://isaac-sim.github.io/IsaacLab/main/source/setup/ecosystem.html) — Everything you need to know about IsaacLab, and more!
- [Isaac Lab ANYmal C environment](https://github.com/isaac-sim/IsaacLab/tree/main/source/isaaclab_tasks/isaaclab_tasks/direct/anymal_c) — This targets ANYmal C (not Unitree Go2), so use it as a reference and adapt robot config, assets, and reward to Go2.
- [DMO (IsaacGym) Go2 walking project page](https://machines-in-motion.github.io/DMO/) • [Go2 walking environment used by the authors](https://github.com/Jogima-cyber/IsaacGymEnvs/blob/e351da69e05e0433e746cef0537b50924fd9fdbf/isaacgymenvs/tasks/go2_terrain.py) • [Config file used by the authors](https://github.com/Jogima-cyber/IsaacGymEnvs/blob/e351da69e05e0433e746cef0537b50924fd9fdbf/isaacgymenvs/cfg/task/Go2Terrain.yaml) — Look at the function `compute_reward_CaT` (beware that some reward terms have a weight of 0 and thus are deactivated, check weights in the config file); this implementation includes strong reward shaping, domain randomization, and training disturbances for robust sim‑to‑real, but it is written for legacy IsaacGym and the challenge is to re-implement it in Isaac Lab.
- **API References**:
    - [ArticulationData (`robot.data`)](https://isaac-sim.github.io/IsaacLab/main/source/api/lab/isaaclab.assets.html#isaaclab.assets.ArticulationData) — Contains `root_pos_w`, `joint_pos`, `projected_gravity_b`, etc.
    - [ContactSensorData (`_contact_sensor.data`)](https://isaac-sim.github.io/IsaacLab/main/source/api/lab/isaaclab.sensors.html#isaaclab.sensors.ContactSensorData) — Contains `net_forces_w` (contact forces).

---

# Final Project : Archit Sharma, Christian Hahn, Greta Perez-Haiek

The following changes were made keeping the requirements and the scope of the project in mind. The primary goal of the the changes was to enforce a stable gait following the desired direction of motion while rewarding feet clearance and a cyclic stepping pattern. Several of the changes were guided by the tutorial and project documents.

<video src="videos/rl-video-base.mp4" width="100%" controls autoplay loop muted> </video>
<p align="center"> Base Model </p>

## Additional Tutorial Rewards

Implementing the additional rewards as guided by tutorial step 5.2 in the `_get_rewards()` function. 

```python
# 1. Penalize non-vertical orientation (projected gravity on XY plane)
rew_orient = torch.sum(torch.square(self.robot.data.projected_gravity_b[:, :2]), dim=1)

# 2. Penalize vertical velocity (z-component of base linear velocity)
rew_lin_vel_z = torch.square(self.robot.data.root_lin_vel_b[:, 2])

# 3. Penalize high joint velocities
rew_dof_vel = torch.sum(torch.square(self.robot.data.joint_vel), dim=1)

# 4. Penalize angular velocity in XY plane (roll/pitch)
rew_ang_vel_xy = torch.sum(torch.square(self.robot.data.root_ang_vel_b[:, :2]), dim=1)

# 5. Action Regularization (L2 norm of actions)
rew_action_mag = torch.sum(torch.square(self.actions), dim = 1)
```
Adding the relevant rewards to the dict

```python
rewards = {
    ...
    "orient": rew_orient * self.cfg.orient_reward_scale,
    "lin_vel_z": rew_lin_vel_z * self.cfg.lin_vel_z_reward_scale,
    "dof_vel": rew_dof_vel * self.cfg.dof_vel_reward_scale,
    "ang_vel_xy": rew_ang_vel_xy * self.cfg.ang_vel_xy_reward_scale,
    "rew_action_mag": rew_action_mag * self.cfg.action_mag_reward_scale, 
    ...
}
```
<video src="videos/rl-video-tutorial.mp4" width="100%" controls autoplay loop muted> </video>
<p align="center"> After Implementing Additional Tutorial Rewards </p>

## Raibert Heuristic Control Strategy

Raibert heauristic calculates the ideal foot placement targets based on the command velocity and gait frequency. This reward penalizes the distance between the robot's actual foot positions and these calculated targets to maintain balance and achieve stable locomotion.

In `_get_rewards()`,
```python
self._step_contact_targets() # Update gait state
rew_raibert_heuristic = self._reward_raibert_heuristic()

rewards = {
        ...
        "raibert_heuristic": rew_raibert_heuristic * self.cfg.raibert_heuristic_reward_scale,
        ...
    }
```

In `_step_contact_targets()`,
```python
frequencies = 3.
phases = 0.5
offsets = 0.
bounds = 0.
durations = 0.5 * torch.ones((self.num_envs,), dtype=torch.float32, device=self.device)
self.gait_indices = torch.remainder(self.gait_indices + self.step_dt * frequencies, 1.0)

foot_indices = [self.gait_indices + phases + offsets + bounds,
                self.gait_indices + offsets,
                self.gait_indices + bounds,
                self.gait_indices + phases]

self.foot_indices = torch.remainder(torch.cat([foot_indices[i].unsqueeze(1) for i in range(4)], dim=1), 1.0)

for idxs in foot_indices:
    stance_idxs = torch.remainder(idxs, 1) < durations
    swing_idxs = torch.remainder(idxs, 1) > durations

    idxs[stance_idxs] = torch.remainder(idxs[stance_idxs], 1) * (0.5 / durations[stance_idxs])
    idxs[swing_idxs] = 0.5 + (torch.remainder(idxs[swing_idxs], 1) - durations[swing_idxs]) * (
                0.5 / (1 - durations[swing_idxs]))

self.clock_inputs[:, 0] = torch.sin(2 * np.pi * foot_indices[0])
self.clock_inputs[:, 1] = torch.sin(2 * np.pi * foot_indices[1])
self.clock_inputs[:, 2] = torch.sin(2 * np.pi * foot_indices[2])
self.clock_inputs[:, 3] = torch.sin(2 * np.pi * foot_indices[3])

# von mises distribution
kappa = 0.07
smoothing_cdf_start = torch.distributions.normal.Normal(0, kappa).cdf  # (x) + torch.distributions.normal.Normal(1, kappa).cdf(x)) / 2

smoothing_multiplier_FL = (smoothing_cdf_start(torch.remainder(foot_indices[0], 1.0)) * (
        1 - smoothing_cdf_start(torch.remainder(foot_indices[0], 1.0) - 0.5)) +
                            smoothing_cdf_start(torch.remainder(foot_indices[0], 1.0) - 1) * (
                                    1 - smoothing_cdf_start(
                                torch.remainder(foot_indices[0], 1.0) - 0.5 - 1)))
smoothing_multiplier_FR = (smoothing_cdf_start(torch.remainder(foot_indices[1], 1.0)) * (
        1 - smoothing_cdf_start(torch.remainder(foot_indices[1], 1.0) - 0.5)) +
                            smoothing_cdf_start(torch.remainder(foot_indices[1], 1.0) - 1) * (
                                    1 - smoothing_cdf_start(
                                torch.remainder(foot_indices[1], 1.0) - 0.5 - 1)))
smoothing_multiplier_RL = (smoothing_cdf_start(torch.remainder(foot_indices[2], 1.0)) * (
        1 - smoothing_cdf_start(torch.remainder(foot_indices[2], 1.0) - 0.5)) +
                            smoothing_cdf_start(torch.remainder(foot_indices[2], 1.0) - 1) * (
                                    1 - smoothing_cdf_start(
                                torch.remainder(foot_indices[2], 1.0) - 0.5 - 1)))
smoothing_multiplier_RR = (smoothing_cdf_start(torch.remainder(foot_indices[3], 1.0)) * (
        1 - smoothing_cdf_start(torch.remainder(foot_indices[3], 1.0) - 0.5)) +
                            smoothing_cdf_start(torch.remainder(foot_indices[3], 1.0) - 1) * (
                                    1 - smoothing_cdf_start(
                                torch.remainder(foot_indices[3], 1.0) - 0.5 - 1)))

self.desired_contact_states[:, 0] = smoothing_multiplier_FL
self.desired_contact_states[:, 1] = smoothing_multiplier_FR
self.desired_contact_states[:, 2] = smoothing_multiplier_RL
self.desired_contact_states[:, 3] = smoothing_multiplier_RR
```

In `_reward_raibert_heuristic()`,
```python
cur_footsteps_translated = self.foot_positions_w - self.robot.data.root_pos_w.unsqueeze(1)
footsteps_in_body_frame = torch.zeros(self.num_envs, 4, 3, device=self.device)
for i in range(4):
    footsteps_in_body_frame[:, i, :] = math_utils.quat_apply_yaw(math_utils.quat_conjugate(self.robot.data.root_quat_w),
                                                    cur_footsteps_translated[:, i, :])

# nominal positions: [FR, FL, RR, RL]
desired_stance_width = 0.25
desired_ys_nom = torch.tensor([desired_stance_width / 2, -desired_stance_width / 2, desired_stance_width / 2, -desired_stance_width / 2], device=self.device).unsqueeze(0)

desired_stance_length = 0.45
desired_xs_nom = torch.tensor([desired_stance_length / 2,  desired_stance_length / 2, -desired_stance_length / 2, -desired_stance_length / 2], device=self.device).unsqueeze(0)

# raibert offsets
phases = torch.abs(1.0 - (self.foot_indices * 2.0)) * 1.0 - 0.5
frequencies = torch.tensor([3.0], device=self.device)
x_vel_des = self._commands[:, 0:1]
yaw_vel_des = self._commands[:, 2:3]
y_vel_des = yaw_vel_des * desired_stance_length / 2
desired_ys_offset = phases * y_vel_des * (0.5 / frequencies.unsqueeze(1))
desired_ys_offset[:, 2:4] *= -1
desired_xs_offset = phases * x_vel_des * (0.5 / frequencies.unsqueeze(1))

desired_ys_nom = desired_ys_nom + desired_ys_offset
desired_xs_nom = desired_xs_nom + desired_xs_offset

desired_footsteps_body_frame = torch.cat((desired_xs_nom.unsqueeze(2), desired_ys_nom.unsqueeze(2)), dim=2)

err_raibert_heuristic = torch.abs(desired_footsteps_body_frame - footsteps_in_body_frame[:, :, 0:2])

reward = torch.sum(torch.square(err_raibert_heuristic), dim=(1, 2))

return reward
```

## Feet Clearance Reward

In `_get_rewards()`,
```python
feet_reward = self._reward_feet_clearance()

rewards = {
        ...
        "feet_clearance" : self.cfg.feet_clearance_reward_scale * feet_reward,
        ...
    }
```

In `_reward_feet_clearance()`
```python
phases = torch.abs(1.0 - (self.foot_indices * 2.0)) * 1.0 - 0.5
foot_height = self.foot_positions_w[:, :, 2]

target_height = 0.08 * phases + 0.02

rew_foot_clearance = torch.square(target_height - foot_height) * (1.0 - self.desired_contact_states)
rew_feet_clearance = torch.sum(rew_foot_clearance, dim=1)
return rew_feet_clearance
```

## Reward small contact forces on the feet

In `_get_rewards()`,
```python
contact_reward = self._reward_tracking_contacts_shaped_force()

rewards = {
        ...
        "contact_shaped_force" : self.cfg.tracking_contacts_shaped_force_reward_scale * contact_reward,
        ...
    }
```

In `_reward_tracking_contacts_shaped_force()`
```python
net_forces = self._contact_sensor.data.net_forces_w_history
latest_forces = net_forces[:, -1]
force_norm = torch.norm(latest_forces, dim=-1)
foot_forces = force_norm[:, self._feet_ids_sensor]

rew_tracking_contacts_shaped_force = 0.0
for i in range(4):
    rew_tracking_contacts_shaped_force += (1.0 - self.desired_contact_states[:, i]) * (1.0 - torch.exp(-foot_forces[:, i] ** 2 / 100.0))

rew_tracking_contacts_shaped_force = rew_tracking_contacts_shaped_force / 4.0
return rew_tracking_contacts_shaped_force
```

## Actuator Friction Model
For a more accurate model and to ease sim-to-real, damping terms are implemented within the joint control loop. A PD control loop calculates joint torques based on position and velocity errors.

In `_apply_action()`,
```python
# Compute PD torques
torques = (
    self.Kp * (
            self.desired_joint_pos 
            - self.robot.data.joint_pos 
        )
        - self.Kd * self.robot.data.joint_vel
)

qd = self.robot.data.joint_vel
tau_stiction = self.Fs * torch.tanh(qd / 0.1)
tau_viscous = self.mu_v * qd
tau_friction = tau_stiction + tau_viscous

torque = torque - tau_friction

# Apply torques to the robot
torque = torch.clip(torque, -self.torque_limits, self.torque_limits)
self.robot.set_joint_effort_target(torques)
```

<video src="videos/rl-video-gait.mp4" width="100%" controls autoplay loop muted> </video>
<p align="center"> After Implementing all rewards and the Actuator friction model </p>

## Extra Credits: Rough Terrain
Changes are made in the enviroment config to train the model on rough and uneven terrain. Foot clearance is rewarded heavily to incentivize walking over steps.

In the `Rob6323Go2EnvCfg` class,  
```python
...
terrain = TerrainImporterCfg(
    prim_path="/World/ground",
    terrain_type="generator",
    terrain_generator=ROUGH_TERRAINS_CFG,
    max_init_terrain_level=9,
    collision_group=-1,
    physics_material=sim_utils.RigidBodyMaterialCfg(
        friction_combine_mode="multiply",
        restitution_combine_mode="multiply",
        static_friction=1.0,
        dynamic_friction=1.0,
        restitution=0.0,
    ),
    debug_vis=False,
)
...
```
<video src="videos/rl-video-rough.mp4" width="100%" controls autoplay loop muted> </video>

Or for stepped terrain,
```python
terrain = TerrainImporterCfg(
    num_envs= 4096,
    env_spacing= 4.0,
    prim_path="/World/ground",
    terrain_type="generator",
    collision_group=-1,
    physics_material=sim_utils.RigidBodyMaterialCfg(
        friction_combine_mode="multiply",
        restitution_combine_mode="multiply",
        static_friction=1.0,
        dynamic_friction=1.0,
        restitution=0.0,
    ),
    debug_vis=False,

    terrain_generator= TerrainGeneratorCfg(
        num_rows=32,
        num_cols=32,
        size= (4.0, 4.0),
        horizontal_scale=0.05,
        vertical_scale=0.05,
        curriculum=False,
        sub_terrains={
            "pyramid_stairs": MeshPyramidStairsTerrainCfg(
                step_height_range=(0.05, 0.15),
                step_width=0.20,
                platform_width=1.25,
                border_width=0.0,
                holes=False,
            )
        }
    )
)
```
<video src="videos/rl-video-step.mp4" width="100%" controls autoplay loop muted> </video>