from pettingzoo import ParallelEnv
import numpy as np

class PolicyNegotiationEnv(ParallelEnv):
    """Multi-agent environment for policy funding negotiations."""
    
    def __init__(self):
        self.agents = ["law_enforcement", "shelter_services", "city_government", "residents"]
        self.num_agents = len(self.agents)

        # Initial funding allocations (starting at 50/50 split)
        self.budget_allocation = {
            "law_enforcement": 0.5,  # 50% of the budget
            "shelter_services": 0.5   # 50% of the budget
        }

        # Define action space: Each agent controls its own budget category
        self.action_spaces = {
            "law_enforcement": np.array([0.0, 1.0]),  # Adjust police funding
            "shelter_services": np.array([0.0, 1.0]),  # Adjust shelter funding
            "city_government": np.array([0.0, 1.0]),  # Adjust final budget balance
            "residents": np.array([-1.0, 1.0])  # React to funding decisions (-1 unhappy, 1 happy)
        }

        self.observation_spaces = {agent: np.array([0.0, 1.0]) for agent in self.agents}

    def reset(self):
        """Reset the environment for a new negotiation session."""
        self.budget_allocation = {"law_enforcement": 0.5, "shelter_services": 0.5}
        return {
            agent: np.array([self.budget_allocation["law_enforcement"], self.budget_allocation["shelter_services"]])
            for agent in self.agents
        }

    def step(self, actions):
        """Agents propose new budget allocations, and we update accordingly."""

        # Assign new budget allocations based on agent actions
        self.budget_allocation["law_enforcement"] = np.clip(actions["law_enforcement"], 0, 1)
        self.budget_allocation["shelter_services"] = np.clip(actions["shelter_services"], 0, 1)

        # City Government Agent ensures the total remains balanced (50/50)
        total_budget = self.budget_allocation["law_enforcement"] + self.budget_allocation["shelter_services"]
        if total_budget > 1.0:
            excess = total_budget - 1.0
            self.budget_allocation["law_enforcement"] -= excess / 2
            self.budget_allocation["shelter_services"] -= excess / 2

        # Compute rewards:
        rewards = {
            "law_enforcement": self.budget_allocation["law_enforcement"],  # Prefers more police funding
            "shelter_services": self.budget_allocation["shelter_services"],  # Prefers more shelter funding
            "city_government": -abs(self.budget_allocation["law_enforcement"] - 0.5),  # Penalizes extreme changes
            "residents": -(abs(self.budget_allocation["law_enforcement"] - self.budget_allocation["shelter_services"]))  # Prefers balance
        }

        # Check if the negotiation ends (placeholder condition)
        done = {agent: False for agent in self.agents}

        return self.budget_allocation, rewards, done, {}

# Test environment
if __name__ == "__main__":
    env = PolicyNegotiationEnv()
    observations = env.reset()
    print("Initial Observations:", observations)
