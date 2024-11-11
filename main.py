from environment.flappy_bird_env import make_env
from agent.dqn_agent import DQNAgent
import time

def train_dqn(episodes=50):
    env = make_env()
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    agent = DQNAgent(state_size, action_size)

    for e in range(episodes):
        episode_start_time = time.time()

    # Training code for each episode here...

        state,_ = env.reset()
        done = False
        truncated = False
        total_reward = 0
        while not (done or truncated):
            action = agent.act(state)
            next_state, reward, done, truncated, _ = env.step(action)
            
            # Consider the episode finished if either done or truncated
            agent.remember(state, action, reward, next_state, done or truncated)
            # print(f"state type : {type(state)}")
            agent.replay()
            
            state = next_state
            total_reward += reward
        print("oui")
        if e % 10 == 0:
            # print(f"state type : {type(state)}")

            agent.update_target_model()
            # print(f"state type : {type(state)}")

        print(f"Episode {e}/{episodes} - Score: {total_reward}, Epsilon: {agent.epsilon}")
        episode_duration = time.time() - episode_start_time
        print(f"Episode duration: {episode_duration}")


    env.close()
    agent.save("trained_dqn_agent.pth")


# def train_dqn(episodes=1000):
#     env = make_env()
#     state_size = env.observation_space.shape[0]
#     action_size = env.action_space.n
#     agent = DQNAgent(state_size, action_size)

#     for e in range(episodes):
#         # Ensure that `state` is just the observation by unpacking correctly
#         state, _ = env.reset()  # Unpack `obs` from `env.reset()`
        
#         done = False
#         truncated = False
#         total_reward = 0
#         while not (done or truncated):
#             action = agent.act(state)

#             # Properly unpack all values from `env.step()`
#             next_state, reward, done, truncated, _ = env.step(action)

#             # Store experience with `next_state` only being the array
#             agent.remember(state, action, reward, next_state, done or truncated)
#             agent.replay()

#             state = next_state
#             total_reward += reward

#         if e % 10 == 0:
#             agent.update_target_model()

#         print(f"Episode {e}/{episodes} - Score: {total_reward}, Epsilon: {agent.epsilon}")

#     env.close()


if __name__ == "__main__":
    start_time = time.time()
    train_dqn(10)
    end_time = time.time()
    total_duration = end_time - start_time
    print(f"Training completed in {total_duration:.2f} seconds.")
