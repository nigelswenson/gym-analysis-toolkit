import gymnasium as gym
import numpy as np
import json
import os
from typing import Dict, Any, List, Union, Optional, Callable
import pickle as pkl

class StructuredDataGymWrapper(gym.Wrapper):
    """
    A wrapper for Gym environments that structures the raw observations, actions, and rewards
    according to user-defined mappings for easier visualization and analysis.
    
    The mappings allow users to specify meaningful names for dimensions in state and action spaces.
    """
    
    def __init__(self, 
                env: gym.Env, 
                mappings_config: Optional[Union[Dict, str]] = None,
                save_path: Optional[str] = None,
                use_pkl: Optional[bool]=False):
        """
        Initialize the wrapper with an environment and optional mappings.
        
        Args:
            env: The Gym environment to wrap
            mappings_config: Either a dictionary containing mappings or a path to a JSON file
                             with mappings. If None, will use generic mappings.
            save_path: Directory to save episode data. If None, data will only be kept in memory.
        """
        super().__init__(env)
        self.episode_count = 0
        self.save_path = save_path
        
        if save_path and not os.path.exists(save_path):
            os.makedirs(save_path)
        
        # Initialize episode data
        self.episode_data = self._init_episode_data()
        
        # Load mappings
        self.mappings = self._load_mappings(mappings_config)

        self.recording = True
        self.use_pkl = use_pkl
    def _load_mappings(self, mappings_config: Optional[Union[Dict, str]]) -> Dict:
        """Load and validate mappings from config dict or JSON file."""
        default_mappings = {
            "state": {"default": lambda x: x},
            "action": {"default": lambda x: x},
            "reward": {"value": lambda x: x}
        }
        
        if mappings_config is None:
            return default_mappings
        
        # Load from JSON file if provided as string
        if isinstance(mappings_config, str):
            try:
                with open(mappings_config, 'r') as f:
                    mappings_config = json.load(f)
            except Exception as e:
                print(f"Error loading mappings from {mappings_config}: {e}")
                return default_mappings
        
        # Convert JSON-style mappings to callable functions
        processed_mappings = {}
        
        # Process state mappings
        if "state" in mappings_config:
            processed_mappings["state"] = {}
            for key, mapping in mappings_config["state"].items():
                processed_mappings["state"][key] = self._create_mapping_function(mapping)
        else:
            processed_mappings["state"] = default_mappings["state"]
            
        # Process action mappings
        if "action" in mappings_config:
            processed_mappings["action"] = {}
            for key, mapping in mappings_config["action"].items():
                processed_mappings["action"][key] = self._create_mapping_function(mapping)
        else:
            processed_mappings["action"] = default_mappings["action"]
            
        # Process reward mappings
        if "reward" in mappings_config:
            processed_mappings["reward"] = {}
            for key, mapping in mappings_config["reward"].items():
                processed_mappings["reward"][key] = self._create_mapping_function(mapping)
        else:
            processed_mappings["reward"] = default_mappings["reward"]
            
        return processed_mappings
    
    def _create_mapping_function(self, mapping_spec) -> Callable:
        """
        Convert a JSON mapping specification to a callable function.
        
        The mapping_spec can be a list of indices, a single index, or a dictionary
        of lists and indexes
        """
        if isinstance(mapping_spec, list):
            # List of indices to extract (e.g., [0, 1] for first two elements)
            indices = mapping_spec
            return lambda obs: [obs[i] for i in indices]
        elif isinstance(mapping_spec, int):
            # Single index to extract (e.g., 0 for first element)
            index = mapping_spec
            return lambda obs: obs[index]
        elif isinstance(mapping_spec, dict):
            # Handle dictionaries that contain more lists
            mapping_dict = {}
            for key in mapping_spec.keys():
                mapping_dict[key] = self._create_mapping_function(mapping_spec=mapping_spec[key])
            return lambda obs: {k: fn(obs) for k,fn in mapping_dict.items()}
            # return mapping_dict
        elif isinstance(mapping_spec, str):
            # Handle string expressions like "obs[0] + obs[1]"
            # This uses eval which can be dangerous if mappings come from untrusted sources
            try:
                # For safety, we limit the locals that can be used in the eval
                return lambda obs: eval(mapping_spec, {"__builtins__": {}}, {"obs": obs, "np": np})
            except Exception as e:
                print(f"Error creating mapping function from '{mapping_spec}': {e}")
                return lambda obs: None
        else:
            # Default passthrough
            return lambda obs: mapping_spec
    
    def _init_episode_data(self) -> Dict[str, Any]:
        """Initialize empty data structure for a new episode."""
        return {
            "timestep_data": [],
            "metadata": {
                "env_id": self.env.spec.id if hasattr(self.env, "spec") and self.env.spec is not None else "unknown",
                "episode_id": self.episode_count,
                "observation_space": str(self.observation_space),
                "action_space": str(self.action_space)
            }
        }
    
    def _apply_mappings(self, data_type: str, raw_data) -> Dict[str, Any]:
        """Apply the appropriate mappings to raw data."""
        mapped_data = {}
        
        # Get mappings for this data type
        mappings = self.mappings.get(data_type, {})
        
        # If no specific mappings, create generic ones based on data shape
        if not mappings or (len(mappings) == 1 and "default" in mappings):
            if isinstance(raw_data, (np.ndarray, list)):
                if hasattr(raw_data, "shape") and len(raw_data.shape) > 0:
                    # Array with shape
                    for i in range(len(raw_data)):
                        mapped_data[f"dim_{i}"] = raw_data[i]
                else:
                    # Single value or scalar
                    mapped_data["value"] = raw_data
            else:
                # Single value
                mapped_data["value"] = raw_data
        else:
            # Apply user-defined mappings
            for key, mapping_func in mappings.items():
                try:
                    mapped_data[key] = mapping_func(raw_data)
                except Exception as e:
                    print(f"Error applying mapping '{key}' to {data_type}: {e}")
                    mapped_data[key] = None
        
        return mapped_data
    
    def reset(self, **kwargs):
        """Reset the environment and initialize a new episode data collection."""
        observation, info = self.env.reset(**kwargs)
        
        # Increment episode counter
        self.episode_count += 1
        
        # Initialize new episode data
        self.episode_data = self._init_episode_data()
        
        # Store structured observation
        structured_obs = self._apply_mappings("state", observation)
        self.episode_data["start_state"] = structured_obs
        self.episode_data["start_info"] = info
    
        return observation, info
    
    def step(self, action):
        """Take a step in the environment and record structured data."""
        observation, reward, terminated, truncated, info = self.env.step(action)
        # Store structured data
        structured_obs = self._apply_mappings("state", observation)
        structured_action = self._apply_mappings("action", action)
        structured_reward = self._apply_mappings("reward", reward)
        timestep_dict = {}
        timestep_dict['state'] = structured_obs
        timestep_dict['action'] = structured_action
        timestep_dict['reward'] = structured_reward
        timestep_dict['terminated'] = terminated
        timestep_dict['truncated'] = truncated
        timestep_dict['info'] = info
        self.episode_data["timestep_data"].append(timestep_dict)
        # self.episode_data["actions"].append(structured_action)
        # self.episode_data["rewards"].append(structured_reward)
        # self.episode_data["dones"].append(terminated)
        # self.episode_data["truncated"].append(truncated)
        # self.episode_data["infos"].append(info)
        
        # Save episode data if the episode is done and save_path is set
        if (terminated or truncated) and self.save_path:
            self.save_episode_data()
        
        return observation, reward, terminated, truncated, info
    
    def save_episode_data(self, custom_path: Optional[str] = None):
        """Save the current episode data to a JSON file."""
        if not self.save_path and not custom_path:
            print("Warning: No save path specified. Episode data not saved.")
            return
        
        save_dir = custom_path or self.save_path
        os.makedirs(save_dir, exist_ok=True)
        
        # Create filename based on environment and episode id
        env_name = self.episode_data["metadata"]["env_id"].replace("/", "_")

        if self.use_pkl:
            filename = f"{env_name}_episode_{self.episode_data['metadata']['episode_id']}.pkl"
            filepath = os.path.join(save_dir, filename)
            with open(filepath, 'wb') as f:
                pkl.dump(self.episode_data,f)
        else:
            filename = f"{env_name}_episode_{self.episode_data['metadata']['episode_id']}.json"
            filepath = os.path.join(save_dir, filename)
            # Convert numpy arrays and other non-serializable objects to lists
            serializable_data = self._make_serializable(self.episode_data)
            
            # Save to file
            with open(filepath, 'w') as f:
                json.dump(serializable_data, f, indent=2)
        
        return filepath
    
    def _make_serializable(self, obj):
        """Convert object to JSON serializable format."""
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, dict):
            return {k: self._make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._make_serializable(item) for item in obj]
        elif isinstance(obj, tuple):
            return [self._make_serializable(item) for item in obj]
        else:
            # Try to convert to a basic type, or use string representation as fallback
            try:
                json.dumps(obj)
                return obj
            except (TypeError, OverflowError):
                return str(obj)
    
    def get_episode_data(self):
        """Return the collected episode data in the structured format."""
        return self.episode_data
    
    def eval(self):
        self.recording = True
    
    def train(self):
        self.recording = False
