import unittest
import torch
import numpy as np
from agent.dqn_agent import DQNAgent

class TestDQNAgent(unittest.TestCase):
    def setUp(self):
        self.agent = DQNAgent(state_size=180, action_size=2)
        self.sample_state = np.random.rand(180)  # Random sample state for testing
        self.sample_action = 1
        self.sample_reward = 1.0
        self.sample_next_state = np.random.rand(180)
        self.done = False
    def test_act_exploration(self):
        # Set epsilon to 1 (full exploration)
        self.agent.epsilon = 1.0
        action = self.agent.act(self.sample_state)
        # Since epsilon=1, the action should be random (0 or 1)
        self.assertIn(action, [0, 1])

    def test_act_exploitation(self):
        # Set epsilon to 0 (no exploration, pure exploitation)
        self.agent.epsilon = 0.0
        action = self.agent.act(self.sample_state)
        # Ensure a valid action (either 0 or 1) is returned
        self.assertIn(action, [0, 1])

    def test_remember(self):
        # Add an experience to the memory
        self.agent.remember(self.sample_state, self.sample_action, self.sample_reward, self.sample_next_state, self.done)
        # Ensure memory has 1 entry
        self.assertEqual(len(self.agent.memory), 1)

    def test_memory_limit(self):
        # Fill memory beyond max length
        for _ in range(2100):  # agent memory has a limit of 2000
            self.agent.remember(self.sample_state, self.sample_action, self.sample_reward, self.sample_next_state, self.done)
        # Ensure memory is capped at 2000 experiences
        self.assertEqual(len(self.agent.memory), 2000)

    def test_update_target_model(self):
        # Manually modify target model weights
        old_weights = [param.clone() for param in self.agent.target_model.parameters()]
        self.agent.update_target_model()
        # Check that target model weights are now equal to main model weights
        for old_param, new_param in zip(old_weights, self.agent.target_model.parameters()):
            self.assertTrue(torch.equal(old_param, new_param))

    def test_replay_empty_memory(self):
        # Run replay with an empty memory (should not crash)
        try:
            self.agent.replay()
        except Exception as e:
            self.fail(f"Replay raised an exception with empty memory: {e}")

    def test_replay_with_enough_memory(self):
        # Fill memory to enable replay (with batch size of 64)
        for _ in range(64):
            self.agent.remember(self.sample_state, self.sample_action, self.sample_reward, self.sample_next_state, self.done)
        # Run replay and ensure it processes without errors
        try:
            self.agent.replay()
        except Exception as e:
            self.fail(f"Replay raised an exception with filled memory: {e}")

    def test_epsilon_decay(self):
        # Set initial epsilon
        initial_epsilon = self.agent.epsilon
        # Perform a replay to trigger epsilon decay
        for _ in range(64):
            self.agent.remember(self.sample_state, self.sample_action, self.sample_reward, self.sample_next_state, self.done)
        self.agent.replay()
        # Ensure epsilon has decayed
        self.assertLess(self.agent.epsilon, initial_epsilon)


if __name__ == '__main__':
    unittest.main()