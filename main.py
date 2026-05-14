import ogbench

dataset_name = 'pointmaze-medium-navigate-v0'
env, train_dataset, val_dataset = ogbench.make_env_and_datasets(dataset_name)

# The dataset is a dict, with keys: 'observations', 'actions', 'goals', etc.
print("Dataset keys:", train_dataset.keys())

# Access the first trajectory / first element properly
# Usually each key is a numpy array of shape [num_samples, ...]
print("Observations shape:", train_dataset['observations'].shape)
print("Actions shape:", train_dataset['actions'].shape)
print("Goals shape:", train_dataset['goals'].shape)

# Example: get the first transition
first_obs = train_dataset['observations'][0]
first_action = train_dataset['actions'][0]
first_goal = train_dataset['goals'][0]

print("First observation:", first_obs)
print("First action:", first_action)