import gymnasium as gym
import json
import numpy as np
from wrappers.recording_wrapper import StructuredDataGymWrapper

# Example with nested dictionary mappings for a more complex environment
def nested_mappings_example():
    """
    Demonstrates using nested dictionary mappings for a MuJoCo environment
    with complex observation space.
    """
    # Create HalfCheetah environment
    env = gym.make('HalfCheetah-v4')
    
    # Define complex nested mappings
    # HalfCheetah observation space has 17 dimensions:
    # - 0: z-coordinate of the torso
    # - 1-2: sin/cos of the angle between torso and world
    # - 3-8: joint angles
    # - 9: velocity of the torso along x-axis
    # - 10: velocity of the torso along y-axis
    # - 11-17: joint velocities
    cheetah_mappings = {
        "state": {
            "torso": {
                "position": {
                    "z": 0
                },
                "orientation": {
                    "sin": 1,
                    "cos": 2
                },
                "velocity": {
                    "x": 9,
                    "y": 10
                }
            },
            "joints": {
                "angles": [3, 4, 5, 6, 7, 8],
                "velocities": [11, 12, 13, 14, 15, 16]
            },
            # Computed values
            "speed": "np.sqrt(obs[9]**2 + obs[10]**2)"
        },
        "action": {
            "joint_efforts": "obs"  # Pass through all actions
        }
    }
    
    # Create the wrapped environment
    wrapped_env = StructuredDataGymWrapper(env, save_path='LOOKHERE', mappings_config=cheetah_mappings,use_pkl=False)
    
    # Run a short episode
    observation, info = wrapped_env.reset()
    done = False
    step_count = 0
    max_steps = 100
    
    while not done and step_count < max_steps:
        action = env.action_space.sample()  # random action
        observation, reward, terminated, truncated, info = wrapped_env.step(action)
        done = terminated or truncated
        step_count += 1
    wrapped_env.save_episode_data()
    # Get the collected data
    episode_data = wrapped_env.get_episode_data()
    
    # Print some of the nested structure for verification
    print("\nEpisode with Nested Mappings:")
    # print(f"Episode length: {len(episode_data['states'])}")
    print("\nExample of nested state structure:")
    # print(episode_data['states'])
    # print(f"Torso z-position: {episode_data['states'][0]['torso']['position']['z']}")
    # print(f"Torso orientation (sin): {episode_data['states'][0]['torso']['orientation']['sin']}")
    # print(f"Joint angles: {episode_data['states'][0]['joints']['angles']}")
    # print(f"Computed speed: {episode_data['states'][0]['speed']}")
    
    return episode_data

# Example analyzing the nested data
def analyze_nested_data(episode_data):
    """
    Example of how to analyze data with nested structure.
    """
    # Extract specific nested values for analysis
    torso_z = [state['torso']['position']['z'] for state in episode_data['states']]
    speeds = [state['speed'] for state in episode_data['states']]
    
    # Print statistics
    print("\nData Analysis:")
    print(f"Average torso height: {np.mean(torso_z):.4f}")
    print(f"Maximum speed: {np.max(speeds):.4f}")
    print(f"Average speed: {np.mean(speeds):.4f}")
    
    # Example of accessing joint data
    joint_angles = np.array([state['joints']['angles'] for state in episode_data['states']])
    joint_velocities = np.array([state['joints']['velocities'] for state in episode_data['states']])
    
    # Calculate joint statistics
    print("\nJoint Analysis:")
    print(f"Joint angle ranges:")
    for i in range(joint_angles.shape[1]):
        min_angle = np.min(joint_angles[:, i])
        max_angle = np.max(joint_angles[:, i])
        print(f"  Joint {i}: {min_angle:.4f} to {max_angle:.4f}")
    
    print(f"\nJoint velocity statistics:")
    for i in range(joint_velocities.shape[1]):
        mean_vel = np.mean(joint_velocities[:, i])
        max_vel = np.max(np.abs(joint_velocities[:, i]))
        print(f"  Joint {i}: avg={mean_vel:.4f}, max abs={max_vel:.4f}")

# Saving nested mapping configuration to JSON
def save_mapping_config():
    """
    Example of saving a complex nested mapping configuration to a JSON file.
    """
    cheetah_mappings = {
        "state": {
            "torso": {
                "position": {
                    "z": 0
                },
                "orientation": {
                    "sin": 1,
                    "cos": 2
                },
                "velocity": {
                    "x": 9,
                    "y": 10
                }
            },
            "joints": {
                "angles": [3, 4, 5, 6, 7, 8],
                "velocities": [11, 12, 13, 14, 15, 16]
            },
            "speed": "np.sqrt(obs[9]**2 + obs[10]**2)"
        },
        "action": {
            "joint_efforts": "obs"
        }
    }
    
    # Save to JSON file
    with open("cheetah_nested_mappings.json", "w") as f:
        json.dump(cheetah_mappings, f, indent=2)
    
    print("\nSaved nested mapping configuration to 'cheetah_nested_mappings.json'")

# Run all examples
if __name__ == "__main__":
    # Run nested mappings example
    episode_data = nested_mappings_example()
    
    # Analyze the data
    # analyze_nested_data(episode_data)
    
    # Save mapping configuration
    save_mapping_config()