#!/usr/bin/env python3
"""
Quick test to verify all imports and basic functionality
"""

import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

print("Testing imports...")

try:
    from devweb.llm_judge_reward import create_llm_judge_reward
    print("✓ llm_judge_reward import successful")
except Exception as e:
    print(f"✗ llm_judge_reward import failed: {e}")

try:
    from devweb.rollout_collector import create_rollout_collector
    print("✓ rollout_collector import successful")
except Exception as e:
    print(f"✗ rollout_collector import failed: {e}")

try:
    from devweb.grpo_trainer import create_grpo_trainer, GRPOConfig
    print("✓ grpo_trainer import successful")
except Exception as e:
    print(f"✗ grpo_trainer import failed: {e}")

try:
    from devweb.llm_grpo_system import create_llm_grpo_system
    print("✓ llm_grpo_system import successful")
except Exception as e:
    print(f"✗ llm_grpo_system import failed: {e}")

try:
    import numpy as np
    print("✓ numpy import successful")
except Exception as e:
    print(f"✗ numpy import failed: {e}")

print("\nTesting basic functionality...")

# Test basic numpy functionality
try:
    test_array = np.array([1, 2, 3])
    mean_val = np.mean(test_array)
    print(f"✓ Numpy basic functionality: mean([1,2,3]) = {mean_val}")
except Exception as e:
    print(f"✗ Numpy basic functionality failed: {e}")

# Test rollout collector
try:
    collector = create_rollout_collector()
    collector.start_trajectory(
        task_id="quick_test",
        task_description="Quick test",
        agent_model="test",
        env_model="test"
    )
    collector.add_step(
        turn=0,
        observation={"test": "data"},
        action="test_action",
        reward=0.5,
        done=False,
        info={}
    )
    trajectory = collector.end_trajectory()
    print(f"✓ Rollout collector: created trajectory with {len(trajectory.steps)} steps")
except Exception as e:
    print(f"✗ Rollout collector failed: {e}")

# Test GRPO trainer
try:
    trainer = create_grpo_trainer()
    print("✓ GRPO trainer created successfully")
except Exception as e:
    print(f"✗ GRPO trainer creation failed: {e}")

print("\n✅ All basic tests completed!")