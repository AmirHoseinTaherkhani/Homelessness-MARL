import numpy as np
import json
from gymnasium import spaces
from ray.rllib.env import MultiAgentEnv
from transformers import pipeline
from sentence_transformers import SentenceTransformer

class PolicyNegotiationEnv(MultiAgentEnv):
    def __init__(self, config=None):
        super().__init__()
        # Define agents participating in the negotiation
        self.agents = ["law_enforcement", "shelter_services", "city_government", "residents"]
        self._agent_ids = set(self.agents)
        self.embedding_size = 768  # Size of sentence embeddings
        self.obs_size = 3 + self.embedding_size * len(self.agents)  # State + embeddings
        self.max_steps = 100  # Maximum steps per episode
        self.episode_count = 0  # Track number of episodes

        # Lazy-loaded models for proposal generation and embeddings
        self.llm = None
        self.sentence_model = None

        # Action space: Each agent proposes a budget allocation (0 to 1)
        self.action_space = spaces.Dict({
            agent: spaces.Box(low=np.float32(0.0), high=np.float32(1.0), shape=(1,), dtype=np.float32)
            for agent in self.agents
        })

        # Observation space: [budget_law, budget_shelter, homelessness_rate] + embeddings
        self.observation_space = spaces.Dict({
            agent: spaces.Box(low=-np.inf, high=np.inf, shape=(self.obs_size,), dtype=np.float32)
            for agent in self.agents
        })

        # Initialize state by resetting the environment
        self.reset()

    def _initialize_models(self):
        """Load language and embedding models only when needed."""
        if self.llm is None:
            self.llm = pipeline("text-generation", model="distilgpt2", device=-1)
        if self.sentence_model is None:
            self.sentence_model = SentenceTransformer("all-distilroberta-v1")

    def reset(self, seed=None, options=None):
        """Reset the environment to an initial state and generate initial proposals."""
        self._initialize_models()
        self.episode_count += 1  # Increment episode counter
        self.step_count = 0
        self.budget_allocation = {"law_enforcement": 0.5, "shelter_services": 0.5}
        self.homelessness_rate = 0.2
        self.proposals = {}
        self.embeddings = {}

        # Generate and log initial proposals
        self._generate_proposals()

        # Construct initial observations
        state_obs = np.array([0.5, 0.5, 0.2], dtype=np.float32)
        all_embeddings = np.concatenate([self.embeddings[agent] for agent in self.agents], axis=0)
        observation = np.concatenate([state_obs, all_embeddings]).astype(np.float32)
        observations = {agent: observation for agent in self.agents}
        infos = {agent: {"proposal": self.proposals[agent]} for agent in self.agents}

        return observations, infos

    def step(self, actions):
        """Advance the environment by one step based on agent actions."""
        self.step_count += 1

        # Update budget allocations based on actions
        for agent in self.agents:
            if agent in actions:
                if agent == "law_enforcement":
                    self.budget_allocation["law_enforcement"] = np.clip(actions[agent][0], 0.0, 1.0)
                elif agent == "shelter_services":
                    self.budget_allocation["shelter_services"] = np.clip(actions[agent][0], 0.0, 1.0)

        # Normalize budget to sum to 1.0
        total_budget = sum(self.budget_allocation.values())
        if total_budget > 1.0:
            for sector in self.budget_allocation:
                self.budget_allocation[sector] /= total_budget

        # Update homelessness rate based on shelter budget (example logic)
        reduction = 0.005 * self.budget_allocation["shelter_services"]
        self.homelessness_rate = max(0.0, self.homelessness_rate - reduction)

        # Generate and log new proposals
        self._generate_proposals()

        # Construct observations
        state_obs = np.array([
            self.budget_allocation["law_enforcement"],
            self.budget_allocation["shelter_services"],
            self.homelessness_rate
        ], dtype=np.float32)
        all_embeddings = np.concatenate([self.embeddings[agent] for agent in self.agents], axis=0)
        observation = np.concatenate([state_obs, all_embeddings]).astype(np.float32)
        observations = {agent: observation for agent in self.agents}

        # Define rewards (example: minimize homelessness rate)
        reward = -self.homelessness_rate
        rewards = {agent: reward for agent in self.agents}

        # Check termination conditions
        done = self.step_count >= self.max_steps or self.homelessness_rate < 0.1
        dones = {agent: done for agent in self.agents}
        dones["__all__"] = done
        truncated = {agent: self.step_count >= self.max_steps for agent in self.agents}
        truncated["__all__"] = self.step_count >= self.max_steps

        infos = {agent: {"proposal": self.proposals[agent]} for agent in self.agents}

        return observations, rewards, dones, truncated, infos

    def _generate_proposals(self):
        """Generate proposals for each agent and log them in JSON Lines format."""
        for agent in self.agents:
            prompt = (
                f"As {agent}, propose a specific budget split (e.g., 60% for law enforcement, 40% for shelter services) "
                f"and one action to reduce homelessness (current rate: {self.homelessness_rate:.2f})."
            )
            proposal = self.llm(prompt, max_length=100, num_return_sequences=1, pad_token_id=50256)[0]["generated_text"]
            proposal = proposal.replace(prompt, "").strip()
            proposal = ' '.join(proposal.split())

            self.proposals[agent] = proposal
            embedding = self.sentence_model.encode(proposal, convert_to_tensor=False).astype(np.float32)
            min_val = np.min(embedding)
            max_val = np.max(embedding)
            embedding = 2 * ((embedding - min_val) / (max_val - min_val + 1e-8) - 0.5)
            embedding = np.clip(embedding, -1.0, 1.0).astype(np.float32)
            self.embeddings[agent] = embedding

            # Log proposal details
            log_entry = {
                "episode": self.episode_count,
                "agent": agent,
                "step": self.step_count,
                "proposal": proposal,
                "homelessness_rate": float(self.homelessness_rate),
                "budget_law_enforcement": float(self.budget_allocation["law_enforcement"]),
                "budget_shelter_services": float(self.budget_allocation["shelter_services"])
            }
            with open("negotiation_logs.json", "a") as f:
                json.dump(log_entry, f)
                f.write("\n")

# Environment creator and registration for RLlib
def env_creator(config=None):
    return PolicyNegotiationEnv(config or {})

from ray.tune.registry import register_env
register_env("policy_negotiation_env", env_creator)