# %% [markdown]
# # Unitree G1 Humanoid: Place Apple in Bowl with PPO
#
# This notebook trains a Unitree G1 humanoid robot to place an apple in a bowl using Proximal Policy Optimization (PPO) reinforcement learning with ManiSkill3.
#
# ## Key Challenges
# - **High-dimensional action space**: 37 dimensions (whole-body control)
# - **Dual objectives**: Balance + manipulation simultaneously
# - **GPU parallelization**: 64 environments running in parallel on GPU
#
# ## Setup
# - Environment: `UnitreeG1PlaceAppleInBowl-v1`
# - Algorithm: PPO with GAE
# - Total timesteps: 5M
# - Hardware target: RTX 6000 Ada

# %%
# Part 1: Setup and Imports
import gymnasium as gym
import mani_skill.envs  # Register ManiSkill environments
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import mediapy as media
from tqdm import tqdm
import time
import os
from typing import Dict, Any, Tuple, Optional

# Hardware check
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA device: {torch.cuda.get_device_name(0)}")
    print(f"CUDA memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
else:
    print("WARNING: CUDA not available, training will be slow!")

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Create output directory
os.makedirs("./results_g1", exist_ok=True)

# %% [markdown]
# ## Part 2: Environment Exploration
#
# Let's first explore the environment to understand:
# - Observation space structure
# - Action space dimensions
# - Visual rendering
# - Random policy baseline

# %%
# Create single environment for exploration
from mani_skill.utils.wrappers.gymnasium import CPUGymWrapper

env = gym.make(
    'UnitreeG1PlaceAppleInBowl-v1',
    num_envs=1,
    obs_mode='state',
    render_mode='rgb_array'
)
env = CPUGymWrapper(env)

print("=" * 60)
print("Environment Information")
print("=" * 60)
print(f"Observation space: {env.observation_space}")
print(f"Action space: {env.action_space}")
print(f"Action shape: {env.action_space.shape}")
print(f"Action low: {env.action_space.low}")
print(f"Action high: {env.action_space.high}")

# Test reset
obs, info = env.reset()
print(f"\nInitial observation type: {type(obs)}")
if isinstance(obs, dict):
    print(f"Observation keys: {obs.keys()}")
    for k, v in obs.items():
        if isinstance(v, np.ndarray):
            print(f"  {k}: shape={v.shape}, dtype={v.dtype}")
else:
    print(f"Observation shape: {obs.shape}, dtype: {obs.dtype}")

# %%
# Render initial state
frame = env.render()
print(f"Render frame shape: {frame.shape}")
plt.figure(figsize=(10, 8))
plt.imshow(frame)
plt.title("Initial State: G1 Humanoid with Apple and Bowl")
plt.axis('off')
plt.tight_layout()
plt.savefig("./results_g1/initial_state.png", dpi=150)
plt.show()

# %%
# Run random policy for 1 episode
print("\n" + "=" * 60)
print("Running Random Policy Baseline")
print("=" * 60)

obs, info = env.reset()
frames = []
rewards = []
episode_reward = 0
max_steps = 250

for step in range(max_steps):
    # Random action
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    
    episode_reward += reward
    rewards.append(reward)
    
    # Render every 5 frames to save time
    if step % 5 == 0:
        frame = env.render()
        frames.append(frame)
    
    if terminated or truncated:
        print(f"Episode finished at step {step}")
        print(f"Total reward: {episode_reward:.4f}")
        break

print(f"Final episode reward: {episode_reward:.4f}")
print(f"Collected {len(frames)} frames")

# Show video
if len(frames) > 0:
    print("\nRandom policy video:")
    media.show_video(frames, fps=10)

# Plot reward curve
plt.figure(figsize=(10, 4))
plt.plot(rewards, label='Step Reward')
plt.axhline(y=0, color='r', linestyle='--', alpha=0.5)
plt.xlabel('Step')
plt.ylabel('Reward')
plt.title('Random Policy: Step Rewards (Expected near zero)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("./results_g1/random_policy_rewards.png", dpi=150)
plt.show()

env.close()
print("\nEnvironment exploration complete!")

# %% [markdown]
# ## Part 3: Actor-Critic Network
#
# Architecture:
# - Shared layers: obs_dim → 512 → 256 → 256 (Tanh activation)
# - Actor head: 256 → 128 → action_dim with learnable log_std initialized to -0.5
# - Critic head: 256 → 128 → 1
# - Orthogonal initialization: sqrt(2) for hidden, 0.01 for actor output, 1.0 for critic output

# %%
class ActorCritic(nn.Module):
    """Actor-Critic network for PPO with specified architecture."""
    
    def __init__(self, obs_dim: int, action_dim: int):
        super().__init__()
        
        # Shared feature extractor: obs_dim -> 512 -> 256 -> 256
        self.shared = nn.Sequential(
            nn.Linear(obs_dim, 512),
            nn.Tanh(),
            nn.Linear(512, 256),
            nn.Tanh(),
            nn.Linear(256, 256),
            nn.Tanh(),
        )
        
        # Actor head: 256 -> 128 -> action_dim
        self.actor = nn.Sequential(
            nn.Linear(256, 128),
            nn.Tanh(),
            nn.Linear(128, action_dim),
        )
        
        # Learnable log standard deviation (initialized to -0.5)
        self.log_std = nn.Parameter(torch.ones(action_dim) * -0.5)
        
        # Critic head: 256 -> 128 -> 1
        self.critic = nn.Sequential(
            nn.Linear(256, 128),
            nn.Tanh(),
            nn.Linear(128, 1),
        )
        
        # Orthogonal initialization
        for module in self.shared:
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
                nn.init.constant_(module.bias, 0)
        for module in self.actor:
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
                nn.init.constant_(module.bias, 0)
        # Actor output layer: small gain for initial near-zero actions
        nn.init.orthogonal_(self.actor[-1].weight, gain=0.01)
        nn.init.constant_(self.actor[-1].bias, 0)
        for module in self.critic:
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
                nn.init.constant_(module.bias, 0)
        # Critic output layer: gain 1.0
        nn.init.orthogonal_(self.critic[-1].weight, gain=1.0)
        nn.init.constant_(self.critic[-1].bias, 0)
    
    def forward(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass returning action mean, log_std, and value."""
        features = self.shared(obs)
        action_mean = self.actor(features)
        value = self.critic(features)
        return action_mean, self.log_std, value
    
    def get_action_and_value(self, obs: torch.Tensor, action: Optional[torch.Tensor] = None
                            ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Get action, log prob, entropy, and value for PPO update."""
        action_mean, log_std, value = self.forward(obs)
        std = torch.exp(log_std)
        
        # Create normal distribution
        dist = torch.distributions.Normal(action_mean, std)
        
        if action is None:
            action = dist.sample()
        
        log_prob = dist.log_prob(action).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)
        
        return action, log_prob, entropy, value.squeeze(-1)
    
    def get_value(self, obs: torch.Tensor) -> torch.Tensor:
        """Get value estimate for observations."""
        features = self.shared(obs)
        return self.critic(features).squeeze(-1)
    
    def get_action_mean(self, obs: torch.Tensor) -> torch.Tensor:
        """Get deterministic action (mean) for evaluation."""
        action_mean, _, _ = self.forward(obs)
        return action_mean

# Test network
test_obs_dim = 100  # Placeholder
test_action_dim = 37  # Unitree G1 action dim
test_net = ActorCritic(test_obs_dim, test_action_dim).to(device)
print("Actor-Critic network created successfully!")
print(f"Parameters: {sum(p.numel() for p in test_net.parameters()):,}")

# %% [markdown]
# ## Part 4: Running Mean Std (Welford Online Algorithm)
#
# Welford's online algorithm for computing running mean and variance.
# Normalizes observations to N(0,1).

# %%
class RunningMeanStd:
    """Tracks running mean and standard deviation using Welford's online algorithm."""
    
    def __init__(self, shape: Tuple[int, ...], epsilon: float = 1e-8):
        self.mean = np.zeros(shape, dtype=np.float64)
        self.var = np.ones(shape, dtype=np.float64)
        self.count = epsilon
        self.epsilon = epsilon
    
    def update(self, batch: np.ndarray):
        """Update running statistics with a batch of observations using Welford's algorithm."""
        batch_mean = batch.mean(axis=0)
        batch_var = batch.var(axis=0)
        batch_count = batch.shape[0]
        self._update_from_moments(batch_mean, batch_var, batch_count)
    
    def _update_from_moments(self, batch_mean: np.ndarray, batch_var: np.ndarray, batch_count: int):
        """Update statistics using Welford's online algorithm."""
        delta = batch_mean - self.mean
        tot_count = self.count + batch_count
        
        # Update mean
        new_mean = self.mean + delta * batch_count / tot_count
        
        # Update variance using Welford's method
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + delta**2 * self.count * batch_count / tot_count
        new_var = M2 / tot_count
        
        self.mean = new_mean
        self.var = new_var
        self.count = tot_count
    
    def normalize(self, obs: np.ndarray, clip: float = 10.0) -> np.ndarray:
        """Normalize observations to N(0,1) using running statistics."""
        return np.clip((obs - self.mean) / np.sqrt(self.var + self.epsilon), -clip, clip)
    
    def normalize_torch(self, obs: torch.Tensor, clip: float = 10.0) -> torch.Tensor:
        """Normalize torch tensor observations."""
        mean = torch.from_numpy(self.mean).to(obs.device).float()
        std = torch.from_numpy(np.sqrt(self.var + self.epsilon)).to(obs.device).float()
        return torch.clamp((obs - mean) / std, -clip, clip)

print("RunningMeanStd class (Welford algorithm) defined!")

# %% [markdown]
# ## Part 5: GPU Vectorized Rollout Buffer
#
# Stores rollouts from parallel environments directly on GPU.
# Shape: (rollout_steps, num_envs, ...)
# Computes GAE on full tensors, flattens for minibatches.

# %%
class RolloutBuffer:
    """Vectorized rollout buffer for GPU parallel environments."""
    
    def __init__(self, rollout_steps: int, num_envs: int, obs_dim: int, action_dim: int, device: torch.device):
        self.rollout_steps = rollout_steps
        self.num_envs = num_envs
        self.device = device
        
        # Pre-allocate tensors on GPU: shape (rollout_steps, num_envs, ...)
        self.observations = torch.zeros((rollout_steps, num_envs, obs_dim), dtype=torch.float32, device=device)
        self.actions = torch.zeros((rollout_steps, num_envs, action_dim), dtype=torch.float32, device=device)
        self.log_probs = torch.zeros((rollout_steps, num_envs), dtype=torch.float32, device=device)
        self.rewards = torch.zeros((rollout_steps, num_envs), dtype=torch.float32, device=device)
        self.values = torch.zeros((rollout_steps, num_envs), dtype=torch.float32, device=device)
        self.dones = torch.zeros((rollout_steps, num_envs), dtype=torch.float32, device=device)
        
        self.advantages = torch.zeros((rollout_steps, num_envs), dtype=torch.float32, device=device)
        self.returns = torch.zeros((rollout_steps, num_envs), dtype=torch.float32, device=device)
        
        self.step = 0
    
    def add(self, obs: torch.Tensor, action: torch.Tensor, log_prob: torch.Tensor, 
            reward: torch.Tensor, value: torch.Tensor, done: torch.Tensor):
        """Add a step of data to the buffer."""
        self.observations[self.step] = obs
        self.actions[self.step] = action
        self.log_probs[self.step] = log_prob
        self.rewards[self.step] = reward
        self.values[self.step] = value
        self.dones[self.step] = done
        self.step += 1
    
    def compute_returns_and_advantages(self, next_value: torch.Tensor, gamma: float, gae_lambda: float):
        """Compute GAE advantages and returns on GPU tensors."""
        last_gae = 0
        
        for t in reversed(range(self.rollout_steps)):
            if t == self.rollout_steps - 1:
                next_non_terminal = 1.0 - self.dones[t]
                next_value_t = next_value
            else:
                next_non_terminal = 1.0 - self.dones[t]
                next_value_t = self.values[t + 1]
            
            # TD error
            delta = self.rewards[t] + gamma * next_value_t * next_non_terminal - self.values[t]
            
            # GAE
            last_gae = delta + gamma * gae_lambda * next_non_terminal * last_gae
            self.advantages[t] = last_gae
        
        # Returns = advantages + values
        self.returns = self.advantages + self.values
    
    def get_batches(self, batch_size: int, num_epochs: int):
        """Generate minibatches for PPO training by flattening (rollout_steps, num_envs)."""
        # Flatten (rollout_steps, num_envs) -> (rollout_steps * num_envs)
        total_size = self.rollout_steps * self.num_envs
        
        # Create flattened tensors
        flat_obs = self.observations.reshape(-1, self.observations.shape[-1])
        flat_actions = self.actions.reshape(-1, self.actions.shape[-1])
        flat_log_probs = self.log_probs.reshape(-1)
        flat_advantages = self.advantages.reshape(-1)
        flat_returns = self.returns.reshape(-1)
        flat_values = self.values.reshape(-1)
        
        # Normalize advantages
        flat_advantages = (flat_advantages - flat_advantages.mean()) / (flat_advantages.std() + 1e-8)
        
        # Generate minibatches
        indices = torch.arange(total_size, device=self.device)
        
        for epoch in range(num_epochs):
            # Shuffle indices for each epoch
            shuffled_indices = indices[torch.randperm(total_size)]
            
            for start in range(0, total_size, batch_size):
                end = min(start + batch_size, total_size)
                mb_indices = shuffled_indices[start:end]
                
                yield (
                    flat_obs[mb_indices],
                    flat_actions[mb_indices],
                    flat_log_probs[mb_indices],
                    flat_advantages[mb_indices],
                    flat_returns[mb_indices],
                    flat_values[mb_indices],
                )
    
    def reset(self):
        """Reset the buffer for next rollout."""
        self.step = 0

print("RolloutBuffer class defined!")

# %% [markdown]
# ## Part 6: PPO Training Function
#
# Complete training loop with:
# - GPU vectorized environments (NO CPUGymWrapper)
# - Observation normalization with RunningMeanStd
# - Linear learning rate decay
# - Periodic evaluation on separate CPU env (10 episodes)
# - Save best checkpoint
# - Log metrics
#
# ManiSkill3 GPU vectorized envs auto-reset. Observations may be dict with key 'obs'.
# reward/terminated/truncated are tensors shape (num_envs,).

# %%
def flatten_obs(obs: Any) -> torch.Tensor:
    """Helper to flatten ManiSkill3 observations (dict or tensor) to tensor."""
    if isinstance(obs, dict):
        # If obs has 'obs' key, use it directly
        if 'obs' in obs:
            obs_tensor = obs['obs']
            if isinstance(obs_tensor, torch.Tensor):
                return obs_tensor.float()
            else:
                return torch.from_numpy(obs_tensor).float()
        
        # Otherwise, concatenate all tensor values in the dict
        obs_parts = []
        for key in sorted(obs.keys()):  # Sort for consistency
            val = obs[key]
            if isinstance(val, torch.Tensor):
                # Flatten if multi-dimensional
                if val.dim() > 1:
                    obs_parts.append(val.reshape(val.shape[0], -1))
                else:
                    obs_parts.append(val.unsqueeze(-1) if val.dim() == 0 else val)
            elif isinstance(val, np.ndarray):
                tensor_val = torch.from_numpy(val).float()
                if tensor_val.dim() > 1:
                    obs_parts.append(tensor_val.reshape(tensor_val.shape[0], -1))
                else:
                    obs_parts.append(tensor_val.unsqueeze(-1) if tensor_val.dim() == 0 else tensor_val)
        
        if len(obs_parts) == 0:
            raise ValueError("No valid observation tensors found in dict")
        
        return torch.cat(obs_parts, dim=-1)
    
    elif isinstance(obs, torch.Tensor):
        return obs.float()
    
    elif isinstance(obs, np.ndarray):
        return torch.from_numpy(obs.copy()).float()

    else:
        raise ValueError(f"Unsupported observation type: {type(obs)}")


def evaluate_policy(agent: ActorCritic, obs_rms: RunningMeanStd, num_episodes: int = 10, 
                    max_steps: int = 250, device: torch.device = device) -> Tuple[float, float]:
    """Evaluate policy on CPU environment. Returns mean reward and success rate."""
    eval_env = gym.make(
        'UnitreeG1PlaceAppleInBowl-v1',
        num_envs=1,
        obs_mode='state',
        render_mode='rgb_array'
    )
    eval_env = CPUGymWrapper(eval_env)
    
    episode_rewards = []
    successes = []
    
    for _ in range(num_episodes):
        obs, _ = eval_env.reset()
        episode_reward = 0
        done = False
        step_count = 0
        
        while not done and step_count < max_steps:
            # Convert obs to tensor
            obs_tensor = flatten_obs(obs).unsqueeze(0).to(device)
            
            # Normalize
            obs_np = obs_tensor.cpu().numpy()
            obs_normalized = obs_rms.normalize(obs_np)
            obs_tensor = torch.from_numpy(obs_normalized).to(device)
            
            # Get deterministic action (mean)
            with torch.no_grad():
                action = agent.get_action_mean(obs_tensor)
            
            # Step
            action_np = action.cpu().numpy()[0]
            obs, reward, terminated, truncated, info = eval_env.step(action_np)
            episode_reward += reward
            step_count += 1
            done = terminated or truncated
        
        episode_rewards.append(episode_reward)
        # Check for success in info
        if 'success' in info:
            successes.append(float(info['success']))
        elif 'episode' in info and 'success' in info['episode']:
            successes.append(float(info['episode']['success']))
    
    eval_env.close()
    
    mean_reward = np.mean(episode_rewards)
    success_rate = np.mean(successes) if len(successes) > 0 else 0.0
    
    return mean_reward, success_rate


def train_ppo(
    total_timesteps: int = 5_000_000,
    num_envs: int = 64,
    rollout_steps: int = 256,
    n_epochs: int = 10,
    batch_size: int = 512,
    lr: float = 3e-4,
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
    clip_eps: float = 0.2,
    ent_coef: float = 0.005,
    vf_coef: float = 0.5,
    max_grad_norm: float = 0.5,
    max_episode_steps: int = 250,
    eval_interval: int = 50,  # Updates between evals
    save_dir: str = "./results_g1",
):
    """Main PPO training loop for Unitree G1 humanoid with GPU vectorized envs."""
    
    print("=" * 60)
    print("PPO Training Configuration")
    print("=" * 60)
    print(f"Total timesteps: {total_timesteps:,}")
    print(f"Num envs: {num_envs}")
    print(f"Rollout steps: {rollout_steps}")
    print(f"Batch size: {batch_size}")
    print(f"Learning rate: {lr}")
    print(f"Device: {device}")
    print("=" * 60)
    
    # Create vectorized GPU environment (NO CPUGymWrapper)
    print("\nCreating GPU vectorized environment...")
    env = gym.make(
        'UnitreeG1PlaceAppleInBowl-v1',
        num_envs=num_envs,
        obs_mode='state',
    )
    
    # Get observation and action dimensions
    obs_sample, _ = env.reset()
    obs_tensor = flatten_obs(obs_sample)
    obs_dim = obs_tensor.shape[-1]
    action_dim = env.action_space.shape[-1]
    
    print(f"Observation dim: {obs_dim}")
    print(f"Action dim: {action_dim}")
    
    # Create agent
    agent = ActorCritic(obs_dim, action_dim).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=lr, eps=1e-5)
    
    # Observation normalization
    obs_rms = RunningMeanStd((obs_dim,))
    
    # Rollout buffer
    buffer = RolloutBuffer(rollout_steps, num_envs, obs_dim, action_dim, device)
    
    # Calculate training parameters
    num_updates = total_timesteps // (num_envs * rollout_steps)
    print(f"Total updates: {num_updates}")
    
    # Tracking
    episode_rewards = []
    episode_lengths = []
    policy_losses = []
    value_losses = []
    entropies = []
    kls = []
    eval_rewards = []
    eval_success_rates = []
    
    best_eval_reward = -float('inf')
    current_episode_reward = np.zeros(num_envs)
    current_episode_length = np.zeros(num_envs)
    
    # Training loop
    global_step = 0
    start_time = time.time()
    
    print("\nStarting training...")
    
    # Reset environment
    obs, info = env.reset()
    obs = flatten_obs(obs)
    
    # Progress bar
    pbar = tqdm(total=num_updates, desc="Training")
    
    for update in range(num_updates):
        # Linear learning rate decay
        current_lr = lr * (1 - update / num_updates)
        for param_group in optimizer.param_groups:
            param_group['lr'] = current_lr
        
        # Rollout collection
        agent.eval()
        
        for step in range(rollout_steps):
            global_step += num_envs
            
            # Convert to numpy for normalization
            obs_np = obs.cpu().numpy()
            obs_rms.update(obs_np)
            obs_normalized = obs_rms.normalize(obs_np)
            obs_normalized_t = torch.from_numpy(obs_normalized).to(device)
            
            # Get action and value
            with torch.no_grad():
                action, log_prob, entropy, value = agent.get_action_and_value(obs_normalized_t)
            
            # Step environment
            next_obs, reward, terminated, truncated, info = env.step(action)
            next_obs = flatten_obs(next_obs)
            
            # Track episode stats
            current_episode_reward += reward.cpu().numpy()
            current_episode_length += 1
            
            # Check for completed episodes (ManiSkill3 auto-resets)
            done = terminated | truncated
            if done.any():
                for env_id in range(num_envs):
                    if done[env_id]:
                        episode_rewards.append(current_episode_reward[env_id])
                        episode_lengths.append(current_episode_length[env_id])
                        current_episode_reward[env_id] = 0
                        current_episode_length[env_id] = 0
            
            # Store in buffer
            buffer.add(
                obs_normalized_t,
                action,
                log_prob,
                reward.float() if isinstance(reward, torch.Tensor) else torch.from_numpy(reward).float().to(device),
                value,
                done.float() if isinstance(done, torch.Tensor) else torch.from_numpy(done).float().to(device)
            )
            
            obs = next_obs
        
        # Compute returns and advantages
        with torch.no_grad():
            obs_np = obs.cpu().numpy()
            obs_normalized = obs_rms.normalize(obs_np)
            obs_normalized_t = torch.from_numpy(obs_normalized).to(device)
            next_value = agent.get_value(obs_normalized_t)
        
        buffer.compute_returns_and_advantages(next_value, gamma, gae_lambda)
        
        # PPO update
        agent.train()
        
        update_policy_losses = []
        update_value_losses = []
        update_entropies = []
        update_kls = []
        
        for batch_data in buffer.get_batches(batch_size, n_epochs):
            mb_obs, mb_actions, mb_old_log_probs, mb_advantages, mb_returns, mb_old_values = batch_data
            
            # Forward pass
            _, new_log_probs, entropy, new_values = agent.get_action_and_value(mb_obs, mb_actions)
            
            # Policy loss (PPO clip)
            ratio = torch.exp(new_log_probs - mb_old_log_probs)
            surr1 = ratio * mb_advantages
            surr2 = torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps) * mb_advantages
            policy_loss = -torch.min(surr1, surr2).mean()
            
            # Value loss (clipped)
            value_pred_clipped = mb_old_values + torch.clamp(
                new_values - mb_old_values, -clip_eps, clip_eps
            )
            value_loss1 = (new_values - mb_returns) ** 2
            value_loss2 = (value_pred_clipped - mb_returns) ** 2
            value_loss = 0.5 * torch.max(value_loss1, value_loss2).mean()
            
            # Entropy loss
            entropy_loss = -entropy.mean()
            
            # Total loss
            loss = policy_loss + vf_coef * value_loss + ent_coef * entropy_loss
            
            # KL divergence for logging
            with torch.no_grad():
                log_ratio = new_log_probs - mb_old_log_probs
                approx_kl = ((torch.exp(log_ratio) - 1) - log_ratio).mean()
            
            # Optimize
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(agent.parameters(), max_grad_norm)
            optimizer.step()
            
            update_policy_losses.append(policy_loss.item())
            update_value_losses.append(value_loss.item())
            update_entropies.append(entropy.mean().item())
            update_kls.append(approx_kl.item())
        
        policy_losses.append(np.mean(update_policy_losses))
        value_losses.append(np.mean(update_value_losses))
        entropies.append(np.mean(update_entropies))
        kls.append(np.mean(update_kls))
        
        buffer.reset()
        
        # Evaluation every 50 updates
        if (update + 1) % eval_interval == 0 or update == num_updates - 1:
            eval_reward, success_rate = evaluate_policy(agent, obs_rms, num_episodes=10, device=device)
            eval_rewards.append(eval_reward)
            eval_success_rates.append(success_rate)
            
            # Save best model
            if eval_reward > best_eval_reward:
                best_eval_reward = eval_reward
                torch.save({
                    'agent': agent.state_dict(),
                    'obs_rms_mean': obs_rms.mean,
                    'obs_rms_var': obs_rms.var,
                    'obs_rms_count': obs_rms.count,
                }, os.path.join(save_dir, 'best_model.pt'))
            
            # Save checkpoint
            torch.save({
                'agent': agent.state_dict(),
                'optimizer': optimizer.state_dict(),
                'obs_rms_mean': obs_rms.mean,
                'obs_rms_var': obs_rms.var,
                'obs_rms_count': obs_rms.count,
                'update': update,
                'global_step': global_step,
            }, os.path.join(save_dir, 'checkpoint.pt'))
        
        # Logging
        fps = int(global_step / (time.time() - start_time))
        recent_rewards = np.mean(episode_rewards[-100:]) if len(episode_rewards) > 0 else 0
        
        pbar.set_postfix({
            'fps': fps,
            'reward': f'{recent_rewards:.2f}',
            'eval_reward': f'{eval_rewards[-1]:.2f}' if len(eval_rewards) > 0 else 'N/A',
            'success': f'{eval_success_rates[-1]:.2%}' if len(eval_success_rates) > 0 else 'N/A',
        })
        pbar.update(1)
    
    pbar.close()
    env.close()
    
    print("\nTraining complete!")
    print(f"Total time: {time.time() - start_time:.2f}s")
    print(f"Best eval reward: {best_eval_reward:.2f}")
    
    # Return training history
    history = {
        'episode_rewards': episode_rewards,
        'episode_lengths': episode_lengths,
        'policy_losses': policy_losses,
        'value_losses': value_losses,
        'entropies': entropies,
        'kls': kls,
        'eval_rewards': eval_rewards,
        'eval_success_rates': eval_success_rates,
        'eval_interval': eval_interval,
        'num_updates': num_updates,
    }
    
    return agent, obs_rms, history

print("Training function defined!")

# %% [markdown]
# ## Part 7: Run Training
#
# Start the PPO training process with the specified hyperparameters.
# This will take 1-3 hours on RTX 6000 Ada.

# %%
# Training hyperparameters
TOTAL_TIMESTEPS = 5_000_000
NUM_ENVS = 64
ROLLOUT_STEPS = 256
N_EPOCHS = 10
BATCH_SIZE = 512
LR = 3e-4
GAMMA = 0.99
GAE_LAMBDA = 0.95
CLIP_EPS = 0.2
ENT_COEF = 0.005
VF_COEF = 0.5
MAX_GRAD_NORM = 0.5
MAX_EPISODE_STEPS = 250

# Run training
agent, obs_rms, history = train_ppo(
    total_timesteps=TOTAL_TIMESTEPS,
    num_envs=NUM_ENVS,
    rollout_steps=ROLLOUT_STEPS,
    n_epochs=N_EPOCHS,
    batch_size=BATCH_SIZE,
    lr=LR,
    gamma=GAMMA,
    gae_lambda=GAE_LAMBDA,
    clip_eps=CLIP_EPS,
    ent_coef=ENT_COEF,
    vf_coef=VF_COEF,
    max_grad_norm=MAX_GRAD_NORM,
    max_episode_steps=MAX_EPISODE_STEPS,
    eval_interval=50,
    save_dir="./results_g1",
)

# %% [markdown]
# ## Part 8: Training Curves
#
# Visualize the training progress with 2x3 subplots:
# rewards, lengths, policy loss, value loss, entropy, KL

# %%
fig, axes = plt.subplots(2, 3, figsize=(15, 8))

# Episode rewards with rolling average
ax = axes[0, 0]
if len(history['episode_rewards']) > 0:
    rewards = np.array(history['episode_rewards'])
    ax.plot(rewards, alpha=0.3, color='blue', label='Raw')
    # Rolling average
    window = min(100, len(rewards) // 10) if len(rewards) > 100 else 10
    if len(rewards) > window:
        rolling = np.convolve(rewards, np.ones(window)/window, mode='valid')
        ax.plot(range(window-1, len(rewards)), rolling, color='blue', linewidth=2, label=f'MA({window})')
    ax.axhline(y=0, color='r', linestyle='--', alpha=0.5)
    ax.set_xlabel('Episode')
    ax.set_ylabel('Reward')
    ax.set_title('Episode Rewards')
    ax.legend()
    ax.grid(True, alpha=0.3)

# Episode lengths
ax = axes[0, 1]
if len(history['episode_lengths']) > 0:
    lengths = np.array(history['episode_lengths'])
    ax.plot(lengths, alpha=0.3, color='green')
    window = min(100, len(lengths) // 10) if len(lengths) > 100 else 10
    if len(lengths) > window:
        rolling = np.convolve(lengths, np.ones(window)/window, mode='valid')
        ax.plot(range(window-1, len(lengths)), rolling, color='green', linewidth=2)
    ax.axhline(y=MAX_EPISODE_STEPS, color='r', linestyle='--', alpha=0.5, label='Max steps')
    ax.set_xlabel('Episode')
    ax.set_ylabel('Length')
    ax.set_title('Episode Lengths')
    ax.legend()
    ax.grid(True, alpha=0.3)

# Policy loss
ax = axes[0, 2]
ax.plot(history['policy_losses'], color='purple')
ax.set_xlabel('Update')
ax.set_ylabel('Policy Loss')
ax.set_title('Policy Loss')
ax.grid(True, alpha=0.3)

# Value loss
ax = axes[1, 0]
ax.plot(history['value_losses'], color='orange')
ax.set_xlabel('Update')
ax.set_ylabel('Value Loss')
ax.set_title('Value Loss')
ax.grid(True, alpha=0.3)

# Entropy
ax = axes[1, 1]
ax.plot(history['entropies'], color='red')
ax.set_xlabel('Update')
ax.set_ylabel('Entropy')
ax.set_title('Policy Entropy')
ax.grid(True, alpha=0.3)

# KL divergence
ax = axes[1, 2]
ax.plot(history['kls'], color='brown')
ax.set_xlabel('Update')
ax.set_ylabel('KL Divergence')
ax.set_title('Approximate KL')
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("./results_g1/training_curves.png", dpi=150)
plt.show()

# Plot evaluation rewards and success rate
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

eval_updates = [(i + 1) * history['eval_interval'] for i in range(len(history['eval_rewards']))]

ax1.plot(eval_updates, history['eval_rewards'], marker='o', color='blue', linewidth=2)
ax1.set_xlabel('Update')
ax1.set_ylabel('Mean Reward')
ax1.set_title('Evaluation Rewards')
ax1.grid(True, alpha=0.3)

ax2.plot(eval_updates, [s * 100 for s in history['eval_success_rates']], marker='o', color='green', linewidth=2)
ax2.set_xlabel('Update')
ax2.set_ylabel('Success Rate (%)')
ax2.set_title('Evaluation Success Rate')
ax2.set_ylim([0, 105])
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("./results_g1/evaluation_curves.png", dpi=150)
plt.show()

print("Training curves plotted!")

# %% [markdown]
# ## Part 9: Evaluation and Video
#
# Load the best model and evaluate on 20 episodes using deterministic policy (action mean).
# Report success rate and mean reward, record and show video of best episode.

# %%
def record_episode(agent: ActorCritic, obs_rms: RunningMeanStd, max_steps: int = 250,
                   deterministic: bool = True, device: torch.device = device) -> Tuple[float, bool, list]:
    """Record a single episode and return reward, success, and frames."""
    env = gym.make(
        'UnitreeG1PlaceAppleInBowl-v1',
        num_envs=1,
        obs_mode='state',
        render_mode='rgb_array'
    )
    env = CPUGymWrapper(env)
    
    obs, _ = env.reset()
    frames = []
    episode_reward = 0
    done = False
    step_count = 0
    success = False
    
    while not done and step_count < max_steps:
        # Render
        frame = env.render()
        frames.append(frame)
        
        # Prepare observation
        obs_tensor = flatten_obs(obs).unsqueeze(0).to(device)
        obs_np = obs_tensor.cpu().numpy()
        obs_normalized = obs_rms.normalize(obs_np)
        obs_tensor = torch.from_numpy(obs_normalized).to(device)
        
        # Get action (deterministic = action mean)
        with torch.no_grad():
            if deterministic:
                action = agent.get_action_mean(obs_tensor)
            else:
                action, _, _, _ = agent.get_action_and_value(obs_tensor)
        
        # Step
        action_np = action.cpu().numpy()[0]
        obs, reward, terminated, truncated, info = env.step(action_np)
        episode_reward += reward
        step_count += 1
        done = terminated or truncated
        
        if 'success' in info and info['success']:
            success = True
    
    # Capture final frame
    frames.append(env.render())
    
    env.close()
    return episode_reward, success, frames


# Load best model
print("Loading best model...")
checkpoint = torch.load("./results_g1/best_model.pt", map_location=device)

# Create new agent with same architecture
# We need to determine obs_dim and action_dim from the saved model
# For now, create a dummy env to get dimensions
temp_env = gym.make('UnitreeG1PlaceAppleInBowl-v1', num_envs=1, obs_mode='state')
temp_env = CPUGymWrapper(temp_env)
obs_sample, _ = temp_env.reset()
obs_tensor = flatten_obs(obs_sample)
obs_dim = obs_tensor.shape[-1]
action_dim = temp_env.action_space.shape[-1]
temp_env.close()

eval_agent = ActorCritic(obs_dim, action_dim).to(device)
eval_agent.load_state_dict(checkpoint['agent'])
eval_agent.eval()

# Restore obs_rms
eval_obs_rms = RunningMeanStd((obs_dim,))
eval_obs_rms.mean = checkpoint['obs_rms_mean']
eval_obs_rms.var = checkpoint['obs_rms_var']
eval_obs_rms.count = checkpoint['obs_rms_count']

# Run evaluation episodes
print("\n" + "=" * 60)
print("Running 20 Evaluation Episodes (Deterministic Policy)")
print("=" * 60)

eval_rewards = []
eval_successes = []
best_frames = None
best_reward = -float('inf')

for i in tqdm(range(20), desc="Evaluating"):
    reward, success, frames = record_episode(eval_agent, eval_obs_rms, deterministic=True, device=device)
    eval_rewards.append(reward)
    eval_successes.append(1.0 if success else 0.0)
    
    if reward > best_reward:
        best_reward = reward
        best_frames = frames

mean_reward = np.mean(eval_rewards)
success_rate = np.mean(eval_successes)
std_reward = np.std(eval_rewards)

print(f"\nEvaluation Results:")
print(f"  Mean reward: {mean_reward:.2f} ± {std_reward:.2f}")
print(f"  Success rate: {success_rate:.1%}")
print(f"  Min reward: {np.min(eval_rewards):.2f}")
print(f"  Max reward: {np.max(eval_rewards):.2f}")

# Show best episode video
if best_frames is not None and len(best_frames) > 0:
    print(f"\nBest episode (reward: {best_reward:.2f}) video:")
    media.show_video(best_frames, fps=20)

# %% [markdown]
# ## Part 10: Save Results
#
# Save the final model, statistics, and training history for future use.

# %%
# Save final model and results
print("=" * 60)
print("Saving Results")
print("=" * 60)

# Save final model (already saved best during training)
print(f"Best model saved at: ./results_g1/best_model.pt")

# Save training history
np.savez(
    "./results_g1/training_history.npz",
    episode_rewards=history['episode_rewards'],
    episode_lengths=history['episode_lengths'],
    policy_losses=history['policy_losses'],
    value_losses=history['value_losses'],
    entropies=history['entropies'],
    kls=history['kls'],
    eval_rewards=history['eval_rewards'],
    eval_success_rates=history['eval_success_rates'],
)
print("Training history saved: ./results_g1/training_history.npz")

# Save observation normalization stats
np.savez(
    "./results_g1/obs_rms_stats.npz",
    mean=obs_rms.mean,
    var=obs_rms.var,
    count=obs_rms.count,
)
print("Observation RMS stats saved: ./results_g1/obs_rms_stats.npz")

# Print summary
print("\n" + "=" * 60)
print("Training Summary")
print("=" * 60)
print(f"Environment: UnitreeG1PlaceAppleInBowl-v1")
print(f"Total timesteps: {TOTAL_TIMESTEPS:,}")
print(f"Number of parallel envs: {NUM_ENVS}")
print(f"Network: {obs_dim} -> 512 -> 256 -> 256 -> {action_dim}")
print(f"\nFinal Evaluation (20 episodes, deterministic policy):")
print(f"  Mean reward: {mean_reward:.2f} ± {std_reward:.2f}")
print(f"  Success rate: {success_rate:.1%}")
print(f"\nSaved files:")
print(f"  - ./results_g1/best_model.pt")
print(f"  - ./results_g1/training_history.npz")
print(f"  - ./results_g1/obs_rms_stats.npz")
print(f"  - ./results_g1/training_curves.png")
print(f"  - ./results_g1/evaluation_curves.png")
print(f"  - ./results_g1/initial_state.png")
print(f"  - ./results_g1/random_policy_rewards.png")
print("=" * 60)
print("Training complete!")
print("=" * 60)
