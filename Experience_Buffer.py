import numpy as np
from PriorityHeap import PriorityHeap, SumTree


class ExperienceMemoryBuffer:
    ### Numpy implementation
    def __init__(self, maxlen=100_000, state_shape=(84, 84), n_step=1, FRAME_STACK=4):
        self.capacity = maxlen
        self.states = np.zeros((self.capacity, *state_shape), dtype=np.uint8)
        # self.states = np.zeros((self.capacity, *state_shape), dtype=np.float32)
        self.actions = np.zeros(self.capacity, dtype=np.uint8)
        self.rewards = np.zeros(self.capacity, dtype=np.float32)
        self.dones = np.zeros(self.capacity, dtype=np.float32)
        self.position = 0
        self.size = 0
        self.stack_num = FRAME_STACK
        self.n_step = n_step

    def add_experience(self, state, action, reward, done):
        index = self.position 
        self.states[index] = state
        self.actions[index] = action
        self.rewards[index] = reward
        self.dones[index] = done

        self.position = (self.position + 1 ) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size=32):
        indices = np.random.choice(self.size - self.n_step, batch_size, replace=False)
        for i in range(len(indices)):
            # better edge case handling
            if indices[i] < self.position and indices[i] + self.n_step > self.position:
                indices[i] = int(min(self.position - self.n_step, indices[i]))
            if indices[i] + self.n_step > self.capacity - 1:
                indices[i] = int(min(self.capacity - self.n_step - 1, indices[i]))
        states = np.stack(
            [
                np.stack([self.select_stacked_states(i + j) for i in indices])
                for j in range(self.n_step)
            ]
        )
        actions = np.stack([self.actions[indices + j] for j in range(self.n_step)])
        rewards = np.stack([self.rewards[indices + j] for j in range(self.n_step)])
        dones = np.stack([self.dones[indices + j] for j in range(self.n_step)])
        next_states = np.stack(
            [
                np.stack([self.select_stacked_states(i + j + 1) for i in indices])
                for j in range(self.n_step)
            ]
        )

        return states, actions, rewards, next_states, dones

    def __len__(self):
        return self.size

    def select_stacked_states(self, index):
        indices = list(range(index - self.stack_num + 1, index + 1))
        indices = [max(i, 0) for i in indices]

        return np.stack([self.states[i] for i in indices])


class ExperienceMemoryBufferPER:
    ### Numpy implementation
    def __init__(
        self,
        maxlen=10_000,
        state_shape=(84, 84),
        n_step=1,
        FRAME_STACK=4,
        beta_start=0.4,
    ):
        self.capacity = maxlen
        self.states = np.zeros((self.capacity, *state_shape), dtype=np.uint8)
        self.actions = np.zeros(self.capacity, dtype=np.float32)
        self.rewards = np.zeros(self.capacity, dtype=np.float32)
        self.dones = np.zeros(self.capacity, dtype=np.float32)
        self.position = 0
        self.size = 0
        self.stack_num = FRAME_STACK
        self.n_step = n_step
        self.beta = beta_start
        self.beta_end = 1.0
        self.beta_anneal_episode = 500_000
        self.per_heap = SumTree(capacity=maxlen)

    def beta_anneal(self):
        self.beta = min(
            self.beta + (self.beta_end - self.beta) / self.beta_anneal_episode,
            self.beta_end,
        )

    def add_experience(self, state, action, reward, done):
        index = self.position % self.capacity
        # import pdb; pdb.set_trace()
        self.states[index] = state
        self.actions[index] = action
        self.rewards[index] = reward
        self.dones[index] = done
        self.per_heap.add(self.per_heap.max, index)

        self.position += 1
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size=32):
        random_indecies, priorities, indices = self.per_heap.sample(
            batch_size=batch_size
        )
        # indices = np.array(indices, dtype=np.uint64)
        # import pdb; pdb.set_trace()
        self.sampled_indices = random_indecies
        # print(indices)

        importance_weights = (1 / (priorities * batch_size)) ** self.beta
        importance_weights /= np.max(importance_weights)
        # print(importance_weights)
        # import pdb; pdb.set_trace()

        for i in range(len(indices)):
            # if indices[i] + self.n_step > min(self.position, self.capacity - 1):
            #     indices[i] = int(min(self.position, self.capacity - 1) - self.n_step)
            # better edge case handling
            if indices[i] < self.position and indices[i] + self.n_step > self.position:
                indices[i] = int(min(self.position - self.n_step, indices[i]))
            if indices[i] + self.n_step > self.capacity - 1:
                indices[i] = int(min(self.capacity - self.n_step - 1, indices[i]))

        states = np.stack(
            [
                np.stack([self.select_stacked_states(i + j) for i in indices])
                for j in range(self.n_step)
            ]
        )
        # import pdb; pdb.set_trace()
        actions = np.stack([self.actions[indices + j] for j in range(self.n_step)])
        rewards = np.stack([self.rewards[indices + j] for j in range(self.n_step)])
        dones = np.stack([self.dones[indices + j] for j in range(self.n_step)])
        next_states = np.stack(
            [
                np.stack([self.select_stacked_states(i + j + 1) for i in indices])
                for j in range(self.n_step)
            ]
        )

        return states, actions, rewards, next_states, dones, importance_weights

    def update_td_error(self, td_errors):
        for i in range(32):
            self.per_heap.update(
                self.sampled_indices[i], td_errors[i] ** self.per_heap.alpha
            )
        # print(sum(td_errors))
        # import pdb; pdb.set_trace()

    def __len__(self):
        return self.size

    def select_stacked_states(self, index):
        # import pdb; pdb.set_trace()
        if index > min(self.capacity, self.position):
            index = self.position
        # import pdb; pdb.set_trace()
        indices = list(range(index - self.stack_num + 1, index + 1))
        indices = [max(i, 0) for i in indices]

        try:
            [self.states[i] for i in indices]
        except:
            import pdb

            pdb.set_trace()

        return np.stack([self.states[i] for i in indices])


import gymnasium as gym
import random


def check_mem(env: gym.Env, mem_size=100_000):
    ### env float32 -> uint8: 2.6gb -> 0.7gb
    ### changed action, reward to uint8 -> 0.7gb almost same
    ### numpy replay memory buffer implementation -> 0.65Gb
    state, _ = env.reset()
    mem = ExperienceMemoryBuffer(mem_size)
    for _ in range(mem_size):
        action = action = random.randint(0, env.action_space.n - 1)
        state = np.array(state)
        next_state, reward, done, truncated, _ = env.step(action)
        action = np.array(action, dtype=np.uint8)
        reward = np.array(reward, dtype=np.uint8)
        done = np.array(done, dtype=np.bool8)
        # import pdb; pdb.set_trace()
        mem.add_experience(state, action, reward, done)
        state = next_state
        if done or truncated:
            state, _ = env.reset()


import unittest
import numpy as np


class TestExperienceMemoryBuffer(unittest.TestCase):
    def setUp(self):
        self.buffer = ExperienceMemoryBuffer(
            maxlen=10, state_shape=(84, 84), n_step=2, FRAME_STACK=4
        )
        self.state = np.ones((84, 84), dtype=np.uint8)
        self.action = 1
        self.reward = 1.0
        self.done = False

    def test_initialization(self):
        self.assertEqual(len(self.buffer), 0)
        self.assertEqual(self.buffer.states.shape, (10, 84, 84))
        self.assertEqual(self.buffer.actions.shape, (10,))
        self.assertEqual(self.buffer.rewards.shape, (10,))
        self.assertEqual(self.buffer.dones.shape, (10,))

    def test_add_experience(self):
        self.buffer.add_experience(self.state, self.action, self.reward, self.done)
        self.assertEqual(len(self.buffer), 1)
        np.testing.assert_array_equal(self.buffer.states[0], self.state)
        self.assertEqual(self.buffer.actions[0], self.action)
        self.assertEqual(self.buffer.rewards[0], self.reward)
        self.assertEqual(self.buffer.dones[0], self.done)

    def test_sampling(self):
        for _ in range(10):
            self.buffer.add_experience(self.state, self.action, self.reward, self.done)
        batch_size = 2
        states, actions, rewards, next_states, dones = self.buffer.sample(batch_size)
        self.assertEqual(states.shape, (2, 2, 4, 84, 84))
        self.assertEqual(actions.shape, (2, 2))
        self.assertEqual(rewards.shape, (2, 2))
        self.assertEqual(next_states.shape, (2, 2, 4, 84, 84))
        self.assertEqual(dones.shape, (2, 2))


class TestExperienceMemoryBufferPER(unittest.TestCase):
    def setUp(self):
        self.buffer = ExperienceMemoryBufferPER(
            maxlen=10, state_shape=(84, 84), n_step=2, FRAME_STACK=4
        )
        self.state = np.ones((84, 84), dtype=np.uint8)
        self.action = 1
        self.reward = 1.0
        self.done = False

    def test_add_experience(self):
        self.buffer.add_experience(self.state, self.action, self.reward, self.done)
        self.assertEqual(len(self.buffer), 1)
        np.testing.assert_array_equal(self.buffer.states[0], self.state)
        self.assertEqual(self.buffer.actions[0], self.action)
        self.assertEqual(self.buffer.rewards[0], self.reward)
        self.assertEqual(self.buffer.dones[0], self.done)


if __name__ == "__main__":
    unittest.main()
