import numpy as np

# Initialize Q-table with zeros
q_table = np.zeros([state_space, action_space])

for i in range(num_episodes):
    state = env.reset()
    done = False
    total_reward = 0
    
    while not done:
        # Choose action with highest Q-value for current state
        action = np.argmax(q_table[state])
        
        # Perform the action and get the reward
        next_state, reward, done, _ = env.step(action)
        next_state = np.reshape(next_state, [1, state_space])
        
        # Update Q-value for the current state-action pair
        target = reward + discount_factor * np.max(q_table[next_state])
        q_table[state][action] += learning_rate * (target - q_table[state][action])
        
        # Update the current state
        state = next_state
        total_reward += reward
