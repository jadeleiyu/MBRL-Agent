#!/usr/bin/env python3
"""
Test script for Model-driven Virtual World System
"""

import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from devweb import create_model_world_system


def test_model_world_system():
    """Test the model-driven virtual world system"""
    print("="*60)
    print("TESTING MODEL-DRIVEN VIRTUAL WORLD SYSTEM")
    print("="*60)

    # Model path - all components use the same model for simplicity
    model_path = "/mnt/shared-storage-user/medeval-share/model/Qwen2.5-7B-Instruct"

    try:
        # Create model world system
        system = create_model_world_system(
            world_model_path=model_path,
            agent_model_path=model_path,
            judge_model_path=model_path,
            max_turns=2,
            storage_path="./data/model_world_trajectories.jsonl"
        )

        print("✓ Model World System created successfully")

        # Run training
        print("\n--- Running Model World Training ---")
        result = system.train_with_grpo(
            num_rollouts=2  # Small number for testing
        )

        print("✓ Model World Training completed:")
        print(f"  - Success: {result['success']}")
        print(f"  - Training round: {result['training_round']}")
        print(f"  - Number of trajectories: {result['num_trajectories']}")
        if 'average_reward' in result:
            print(f"  - Average reward: {result['average_reward']:.3f}")

        # Get system status
        status = system.get_system_status()
        print(f"\nSystem Status:")
        print(f"  - Training rounds: {status['training_rounds']}")
        print(f"  - Best reward: {status['best_reward']:.3f}")
        print(f"  - Number of trajectories: {status['num_trajectories']}")
        print(f"  - World model: {status['world_model']}")
        print(f"  - Agent model: {status['agent_model']}")
        print(f"  - Judge model: {status['judge_model']}")

        return True

    except Exception as e:
        print(f"Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Main test function"""
    print("Model-driven Virtual World Test")
    print("Model Path:", "/mnt/shared-storage-user/medeval-share/model/Qwen2.5-7B-Instruct")

    success = test_model_world_system()
    if success:
        print("OK")
    return 0 if success else 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)