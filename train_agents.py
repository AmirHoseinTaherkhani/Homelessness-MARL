import ray
from ray import tune
from ray.rllib.algorithms.ppo import PPOConfig
from ai_negotiation_env import PolicyNegotiationEnv
import warnings
warnings.filterwarnings('ignore')

# Initialize Ray
ray.init(ignore_reinit_error=True)

# Create an instance of the environment to access spaces and agents
env = PolicyNegotiationEnv()

# Configure PPO for multi-agent training, optimized for Studio Lab
config = (
    PPOConfig()
    .environment("policy_negotiation_env")
    .env_runners(
        num_env_runners=1,          # Single runner for free tier
        rollout_fragment_length=50, # Smaller fragment for faster rollouts
        sample_timeout_s=200,        # Adjusted timeout
    )
    .training(
        lr=0.0001,
        train_batch_size=1000,     # Reduced for memory constraints
        gamma=0.99,
    )
    .multi_agent(
        policies={
            agent: (None, env.observation_space[agent], env.action_space[agent], {})
            for agent in env.agents
        },
        policy_mapping_fn=lambda agent_id, *args, **kwargs: agent_id,
    )
    .api_stack(
        enable_rl_module_and_learner=False,
        enable_env_runner_and_connector_v2=False,
    )
)

# Run training
tuner = tune.Tuner(
    "PPO",
    param_space=config.to_dict(),
    run_config=tune.RunConfig(
        stop={"training_iteration": 100},
        verbose=0,
    ),
)
results = tuner.fit()

# Get and print the best checkpoint
best_result = results.get_best_result()
checkpoint_path = best_result.checkpoint
print(f"Best model saved at: {checkpoint_path}")

# Shutdown Ray
ray.shutdown()