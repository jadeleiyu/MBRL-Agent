#!/usr/bin/env python3
"""
Startup script for Model-driven Virtual World System
Uses dotenv configuration for easy setup
"""

import os
import sys
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from devweb import create_model_world_system


def main():
    """Main startup function"""
    print("="*60)
    print("MODEL-DRIVEN VIRTUAL WORLD SYSTEM")
    print("="*60)

    # Get configuration from environment variables
    world_model_path = os.getenv("WORLD_MODEL_PATH", "/mnt/shared-storage-user/medeval-share/model/Qwen2.5-7B-Instruct")
    agent_model_path = os.getenv("AGENT_MODEL_PATH", "/mnt/shared-storage-user/medeval-share/model/Qwen2.5-7B-Instruct")
    judge_model_path = os.getenv("JUDGE_MODEL_PATH", "/mnt/shared-storage-user/medeval-share/model/Qwen2.5-7B-Instruct")
    max_turns = int(os.getenv("MAX_TURNS", "3"))
    storage_path = os.getenv("STORAGE_PATH", "./data/model_world_trajectories.jsonl")

    print(f"Configuration:")
    print(f"  - World Model: {world_model_path}")
    print(f"  - Agent Model: {agent_model_path}")
    print(f"  - Judge Model: {judge_model_path}")
    print(f"  - Max Turns: {max_turns}")
    print(f"  - Storage Path: {storage_path}")
    print()

    try:
        # Create model world system
        system = create_model_world_system(
            world_model_path=world_model_path,
            agent_model_path=agent_model_path,
            judge_model_path=judge_model_path,
            max_turns=max_turns,
            storage_path=storage_path
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

        print("\n" + "="*60)
        print("✅ STARTUP COMPLETED SUCCESSFULLY")
        print("="*60)

    except Exception as e:
        print(f"❌ Startup failed with error: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)