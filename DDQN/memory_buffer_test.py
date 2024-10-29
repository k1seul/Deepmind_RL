import unittest
import numpy as np
from Env import ExperienceMemoryBuffer  # Replace with your actual module name

class TestExperienceMemoryBuffer(unittest.TestCase):

    def setUp(self):
        self.buffer = ExperienceMemoryBuffer(maxlen=10, state_shape=(84, 84))

    def test_initialization(self):
        # Verify initial values
        self.assertEqual(self.buffer.capacity, 10)
        self.assertEqual(self.buffer.size, 0)
        self.assertEqual(self.buffer.position, 0)
        self.assertEqual(self.buffer.stack_num, 4)
        self.assertEqual(self.buffer.states.shape, (10, 84, 84))
        self.assertEqual(self.buffer.actions.shape, (10,))
        self.assertEqual(self.buffer.rewards.shape, (10,))
        self.assertEqual(self.buffer.dones.shape, (10,))

    def test_add_experience(self):
        # Add a sample experience
        state = np.zeros((84, 84), dtype=np.uint8)
        action = 1
        reward = 1.0
        done = False

        self.buffer.add_experience(state, action, reward, done)

        # Check if experience is correctly added
        self.assertTrue((self.buffer.states[0] == state).all())
        self.assertEqual(self.buffer.actions[0], action)
        self.assertEqual(self.buffer.rewards[0], reward)
        self.assertEqual(self.buffer.dones[0], done)
        self.assertEqual(self.buffer.size, 1)

    def test_buffer_overflow(self):
        # Fill the buffer beyond its capacity
        for i in range(12):
            state = np.ones((84, 84), dtype=np.uint8) * i
            self.buffer.add_experience(state, i, i * 0.1, i % 2 == 0)
        
        # Verify buffer overflow handling
        self.assertEqual(self.buffer.size, 10)  # Should not exceed maxlen
        self.assertTrue((self.buffer.states[0] == np.ones((84, 84), dtype=np.uint8) * 10).all())
        self.assertTrue((self.buffer.states[-1] == np.ones((84, 84), dtype=np.uint8) * 9).all())
        
    def test_sample(self):
        # Populate the buffer
        for i in range(10):
            state = np.ones((84, 84), dtype=np.uint8) * i
            self.buffer.add_experience(state, i, i * 0.1, i % 2 == 0)
        
        batch_size = 4
        states, actions, rewards, next_states, dones = self.buffer.sample(batch_size)
        
        # Check sample sizes
        self.assertEqual(states.shape[0], batch_size - 1)
        self.assertEqual(actions.shape[0], batch_size - 1)
        self.assertEqual(rewards.shape[0], batch_size - 1)
        self.assertEqual(next_states.shape[0], batch_size - 1)
        self.assertEqual(dones.shape[0], batch_size - 1)

    def test_len(self):
        # Test the __len__ method
        self.assertEqual(len(self.buffer), 0)
        self.buffer.add_experience(np.zeros((84, 84), dtype=np.uint8), 1, 1.0, False)
        self.assertEqual(len(self.buffer), 1)

    def test_select_stacked_states(self):
        # Populate buffer
        for i in range(10):
            state = np.ones((84, 84), dtype=np.uint8) * i
            self.buffer.add_experience(state, i, i * 0.1, i % 2 == 0)
        
        # Test stacked states
        stacked_state = self.buffer.select_stacked_states(5)
        self.assertEqual(stacked_state.shape, (4, 84, 84))
        self.assertTrue((stacked_state[0] == np.ones((84, 84), dtype=np.uint8) * 2).all())
        self.assertTrue((stacked_state[-1] == np.ones((84, 84), dtype=np.uint8) * 5).all())

if __name__ == '__main__':
    unittest.main()
