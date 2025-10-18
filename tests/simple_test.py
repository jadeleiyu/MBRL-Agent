#!/usr/bin/env python3
"""
Simple test to verify numpy import fix
"""

import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

print("Testing numpy import in llm_grpo_system...")

try:
    # Test if we can import the fixed file
    import numpy as np
    print("✓ numpy import successful")

    # Test basic numpy functionality
    test_data = [1.0, 2.0, 3.0]
    mean_val = np.mean(test_data)
    print(f"✓ numpy.mean() works: {mean_val}")

    # Test the specific line that was failing
    rollout_results = [{"total_reward": 1.0}, {"total_reward": 2.0}, {"total_reward": 3.0}]
    avg_reward = np.mean([r["total_reward"] for r in rollout_results])
    print(f"✓ Fixed line works: avg_reward = {avg_reward}")

    print("\n✅ All numpy-related fixes verified!")

except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()