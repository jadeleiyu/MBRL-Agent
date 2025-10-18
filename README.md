# DevWeb - Virtual Web Environment for LLM Training

A reinforcement learning framework for training LLM agents in simulated web environments without real internet access.

## Overview

This system provides a complete framework for training LLM agents to interact with web environments:
- **Virtual Web Environment**: One LLM simulates web pages, forms, and interactions
- **Web Interaction Agent**: Another LLM learns to navigate and interact with the virtual web
- **LLM-as-Judge**: Evaluates agent performance and provides rewards for successful interactions
- **GRPO Training**: Optimizes agent behavior using reinforcement learning
- **Safe Training**: All interactions happen in simulation, no real internet access required

## Core Components

### 1. LLM-as-Judge Reward System (`llm_judge_reward.py`)

Uses LLM as a judge to evaluate agent performance in virtual web interactions.

```python
from devweb import create_llm_judge_reward

# Create judge
judge = create_llm_judge_reward(
    model_path="/path/to/judge/model",
    temperature=0.2,
    top_p=0.9
)

# Evaluate response
task_info = {
    "instruction": "Evaluate factual question answering ability",
    "question": "What is artificial intelligence?"
}
action = "Artificial intelligence is computer systems that simulate human intelligence."

reward_output = judge(task_info=task_info, action=action)
print(f"Reward: {reward_output.reward}")
print(f"Detailed scores: {reward_output.metadata['detailed_scores']}")
```

### 2. Rollout Data Collector (`rollout_collector.py`)

Collects agent interaction trajectory data in virtual web environments.

```python
from devweb import create_rollout_collector

# Create collector
collector = create_rollout_collector(storage_path="./trajectories.jsonl")

# Start trajectory collection
collector.start_trajectory(
    task_id="task_1",
    task_description="Test task",
    agent_model="agent_model",
    env_model="env_model"
)

# Add interaction steps
for turn in range(3):
    collector.add_step(
        turn=turn,
        observation={"question": f"Question{turn}"},
        action=f"Answer{turn}",
        reward=0.7 + turn * 0.1,
        done=(turn == 2),
        info={"step_info": f"Step{turn}"}
    )

# End trajectory
 trajectory = collector.end_trajectory()
print(f"Total reward: {trajectory.total_reward}")
```

### 3. GRPO Trainer (`grpo_trainer.py`)

GRPO algorithm implementation for optimizing web interaction policies based on rollout data.

```python
from devweb import create_grpo_trainer, GRPOConfig

# Create trainer
config = GRPOConfig(
    learning_rate=1e-5,
    batch_size=32,
    num_epochs=3,
    group_size=4
)
trainer = create_grpo_trainer(config)

# Prepare training data
training_data = trainer.prepare_training_data(trajectories)

# Execute training step
current_logprobs = [0.0] * len(training_data["actions"])
training_stats = trainer.train_step(training_data, current_logprobs)

print(f"Policy loss: {training_stats['policy_loss']}")
```

### 4. Integrated System (`llm_grpo_system.py`)

Complete system for training web interaction agents in virtual environments.

```python
from devweb import create_llm_grpo_system

# Create integrated system
system = create_llm_grpo_system(
    agent_model_path="/path/to/agent/model",
    judge_model_path="/path/to/judge/model",
    max_turns=3,
    storage_path="./training_data.jsonl"
)

# Run training
class SimpleAgent:
    def act(self, observation):
        return "This is a simulated response"

agent = SimpleAgent()
result = system.train_with_grpo(
    num_rollouts=10,
    agent=agent
)

print(f"Training round: {result['training_round']}")
print(f"Average reward: {result['average_reward']}")
```

## Quick Start

### 1. Install Dependencies

Ensure necessary dependencies are installed:
```bash
pip install vllm numpy
```

### 2. Run Examples

View basic functionality demonstration:
```bash
cd devweb
python example_llm_grpo.py
```

### 3. Run Tests

Run unit tests to ensure functionality:
```bash
python -m pytest test_llm_grpo.py -v
```

## Architecture Design

### Data Flow

```
Web Agent → Virtual Web Environment → LLM Judge → Reward → Rollout Collector → GRPO Trainer
```

1. **Web Agent**: Interacts with virtual web pages (clicks, forms, navigation)
2. **Virtual Web Environment**: Simulates web pages, forms, and user interactions
3. **LLM Judge**: Evaluates web interaction success and provides rewards
4. **Rollout Collector**: Collects web interaction trajectory data
5. **GRPO Trainer**: Optimizes web interaction policies using reinforcement learning

### Key Features

- **Virtual Web Simulation**: Realistic web interaction training without internet access
- **Safe Training Environment**: No risk of real-world consequences or data exposure
- **Modular Design**: Each component can be used independently
- **Extensibility**: Easy to add new web interaction scenarios or evaluation criteria
- **Data Persistence**: Supports saving and loading web interaction trajectories
- **Monitoring and Evaluation**: Provides training progress and web interaction performance metrics

## Configuration

### LLM Judge Configuration

- `model_path`: Judge model path
- `temperature`: Sampling temperature (0.0-1.0)
- `top_p`: Top-p sampling parameter (0.0-1.0)
- `max_tokens`: Maximum tokens to generate

### GRPO Training Configuration

- `learning_rate`: Learning rate
- `batch_size`: Batch size
- `num_epochs`: Training epochs
- `group_size`: Group size
- `advantage_clip`: Advantage function clipping
- `entropy_coef`: Entropy regularization coefficient

## Practical Usage Recommendations

### 1. Model Selection

- **Virtual Environment Model**: Choose models with strong world modeling and simulation capabilities
- **Web Agent Model**: Select models with good reasoning and interaction capabilities
- **Judge Model**: Choose models with strong evaluation and feedback generation

### 2. Web Interaction Scenario Design

Design realistic web interaction scenarios:
- E-commerce browsing and purchasing
- Form filling and submission
- Navigation and search tasks
- Multi-step workflows
- Error handling and recovery

### 3. Evaluation Criteria

Design evaluation criteria for web interactions:
- Task completion success
- Efficiency and number of steps
- User experience quality
- Error avoidance
- Goal achievement

### 4. Training Strategy

- Start with simple web interactions and gradually increase complexity
- Monitor reward changes and interaction success rates
- Regularly evaluate agent performance on held-out scenarios
- Save best model checkpoints for deployment

## Extension Development

### Adding New Evaluation Criteria

Inherit `LLMJudgeRewardFunction` class and override evaluation logic:

```python
class CustomJudgeReward(LLMJudgeRewardFunction):
    def __call__(self, reward_input):
        # Custom evaluation logic
        pass
```

### Adding New Training Algorithms

Inherit `GRPOTrainer` class and implement training logic:

```python
class CustomTrainer(GRPOTrainer):
    def train_step(self, training_data, current_logprobs):
        # Custom training logic
        pass
```

## Troubleshooting

### Common Issues

1. **LLM Output Parsing Failure**
   - Check if prompt requires JSON format output
   - Verify model supports required output format

2. **Training Not Converging**
   - Adjust learning rate and batch size
   - Check reward function design
   - Validate data quality

3. **Insufficient Memory**
   - Reduce batch size
   - Use smaller models
   - Enable gradient accumulation

## License

This project is open source under MIT License.