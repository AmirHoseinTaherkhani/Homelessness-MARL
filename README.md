# AI-Powered Urban Policy Negotiation

**Welcome to a smarter way to tackle urban challenges!**

This repository houses an innovative AI-powered negotiation model designed to simulate budget allocation decisions for urban policy, focusing on the trade-offs between law enforcement and homelessness services in Los Angeles. By blending multi-agent reinforcement learning (MARL), large language model (LLM) negotiations, and natural language processing (NLP), this project offers data-driven insights into optimizing limited resources for maximum impact.

## What’s This Project About?

Homelessness in Los Angeles is a tough nut to crack—with over 75,000 unhoused individuals (2023 LAHSA report) and finite budgets, policymakers face hard choices. This project uses AI to simulate negotiations between stakeholders like city officials, law enforcement, shelter services, and residents. The result? A tool that reveals effective funding strategies, visualized beautifully, and grounded in real-world data. Whether you're a developer, urban planner, or AI enthusiast, this project invites you to explore how technology can shape better cities.

## Key Features

- **Multi-Agent Reinforcement Learning (MARL)**: Powered by Ray RLlib, agents negotiate budget splits to minimize homelessness.
- **LLM-Driven Negotiations**: DistilGPT2 generates realistic stakeholder arguments, adding depth to the simulation.
- **NLP Insights**: spaCy analyzes negotiation logs to uncover priorities (e.g., “Shelter Services focuses 80% on housing”).
- **Real-World Data**: Built on LA’s 2024 budget and LAHSA statistics for authenticity.
- **Visualization**: TensorBoard tracks training progress and results.

## Getting Started

### Prerequisites

Before diving in, ensure you have the following installed:
- Python 3.8+
- Git
- [Ray RLlib](https://docs.ray.io/en/latest/rllib/index.html) (`pip install "ray[rllib]"`)
- [Transformers](https://huggingface.co/docs/transformers/installation) (`pip install transformers`)
- [spaCy](https://spacy.io/usage) (`pip install spacy`) and its English model (`python -m spacy download en_core_web_sm`)
- [TensorBoard](https://www.tensorflow.org/tensorboard/get_started) (`pip install tensorboard`)

A decent CPU or GPU is recommended for training—Apple Silicon works great on CPU!

### Installation

1. **Clone the Repository**  
   ```bash
   git clone https://github.com/yourusername/ai-urban-policy-negotiation.git
   cd ai-urban-policy-negotiation

2. Set Up a Virtual Environment (optional but recommended)
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate

3. Install Dependencies
    ```bash
    pip install -r requirements.txt

4. Download spaCy Model
    ```bash
    python -m spacy download en_core_web_sm

5. Running the Project
    ```bash
    python train.py

6. Visualize Results
    ```bash
    tensorboard --logdir ./ray_results
    python word_cloud.py
    python sentiment_analysis.py
    python budget_allocation.py