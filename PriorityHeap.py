import numpy as np
import random
import unittest
import pickle


class PriorityHeap:
    def __init__(self, maxlen=1_000_000, n_step=1):
        self.capacity = maxlen
        ## heap_node = [td_error, index]
        self.perheap = np.zeros(shape=(maxlen, 2), dtype=np.float64)
        self.position = 0
        self.element_num = 0
        self.alpha = 0.4
        self.beta_start = 0.5  ## -> anneal to 1
        self.beta_end = 1.0
        self.k = 32
        self.n_step = n_step
        if maxlen == 1_000_000:
            with open("heap_segments.pkl", "rb") as f:
                self.heap_segment_index = pickle.load(f)
        else:
            with open(f"heap_segments_10000.pkl", "rb") as f:
                self.heap_segment_index = pickle.load(f)

    def add_element(self, index, td_error=1):
        # index = self.position % self.capacity
        position = self.element_num % self.capacity
        if self.element_num == 0:
            self.perheap[position, 0] = td_error
        else:
            self.perheap[position, 0] = self.__max__()

        self.perheap[position, 1] = index
        self.position = min(self.position + 1, self.capacity - 1)
        self.element_num += 1

    def __sum__(self):
        return np.sum(self.perheap[:, 0] ** self.alpha)

    def __repr__(self):
        return str(self.perheap[: self.position])

    def __max__(self):
        return np.max(self.perheap[:, 0])

    def sort(self):
        self.perheap[: self.position] = self.perheap[: self.position][
            np.argsort(self.perheap[: self.position, 0])[::-1]
        ]

    def random_select_one(self, start, end):
        return random.choice(range(start, end + 1))

    def random_batch_indicies(self):
        random_batch_start = self.heap_segment_index[self.position]
        random_indecies = [random.randint(0, random_batch_start[0])]
        random_indecies += [
            random.randint(random_batch_start[i - 1], random_batch_start[i])
            for i in range(1, self.k - 1)
        ]
        # random_indecies += [random.randint(random_batch_start[self.k -2], random_batch_start[self.k-2])]
        indices = self.perheap[random_indecies, 1].astype(int)
        return random_indecies, indices

    def update_elements(self, indicies, priorities):
        # import pdb; pdb.set_trace()
        self.perheap[indicies, 0] = priorities
        # pdb.set_trace()


class SumTree:
    def __init__(self, capacity):
        self.capacity = capacity
        self.tree = np.zeros(shape=(2 * capacity - 1), dtype=np.float64)
        self.data = np.zeros(shape=(capacity), dtype=np.uint32)
        self.n = 0
        self.alpha = 0.6
        self.batch_size = 32
        self.p_ranges = [
            (i / self.batch_size, (i + 1) / self.batch_size)
            for i in range(self.batch_size)
        ]
        self.max = 1

    def add(self, value, buffer_index):
        idx = self.capacity + self.n - 1
        self.data[self.n] = buffer_index
        self.update(idx, value**self.alpha)
        self.max = max(value, self.max)

        self.n += 1
        if self.n >= self.capacity:
            self.n = 0

    def update(self, leaf, value):
        self.max = max(value, self.max)
        old_val = self.tree[leaf]
        change = value - old_val
        self.tree[leaf] = value
        parenet_idx = (leaf - 1) // 2
        self._propagate(parenet_idx, change)

    def _propagate(self, idx, change):
        self.tree[idx] += change

        parent_idx = (idx - 1) // 2
        if parent_idx >= 0:
            self._propagate(parent_idx, change)

    def get_leaf(self, value):
        leaf_idx = 0

        while leaf_idx < self.capacity - 1:
            left = 2 * leaf_idx + 1
            right = 2 * leaf_idx + 2
            if value <= self.tree[left]:
                leaf_idx = left
            else:
                value -= self.tree[left]
                leaf_idx = right

        priority = self.tree[leaf_idx]
        data = self.data[leaf_idx - self.capacity + 1]

        return leaf_idx, priority, data

    def sample(self, batch_size=32):
        p_values = [random.uniform(*i) * self.total_sum for i in self.p_ranges]
        # print(p_values)
        leaf_index = np.zeros(shape=(batch_size,), dtype=np.uint32)
        priorities = np.zeros(shape=(batch_size,), dtype=np.float64)
        data_index = np.zeros(shape=(batch_size,), dtype=np.uint32)

        for n, p in enumerate(p_values):
            leaf, priority, data = self.get_leaf(p)
            leaf_index[n] = leaf
            data_index[n] = data
            priorities[n] = priority

        # print(self.max)

        return leaf_index, priorities, data_index

    @property
    def total_sum(self):
        return self.tree[0]


class TestSumTree(unittest.TestCase):
    def setUp(self):
        self.capacity = 100_000
        self.sum_tree = SumTree(self.capacity)
        random.seed(0)

    def test_initialization(self):
        """Test that SumTree initializes correctly."""
        self.assertEqual(self.sum_tree.capacity, self.capacity)
        self.assertTrue(
            np.array_equal(self.sum_tree.tree, np.zeros(2 * self.capacity - 1))
        )
        self.assertTrue(
            np.array_equal(self.sum_tree.data, np.zeros(self.capacity, dtype=np.uint32))
        )
        self.assertEqual(self.sum_tree.n, 0)

    def test_add(self):
        """Test the add method to ensure it adds values and updates indices correctly."""
        priority_values = np.arange(100_000) / 100_000
        batch_index = range(len(priority_values))

        for i in range(len(batch_index)):
            self.sum_tree.add(priority_values[i], batch_index[i])
        self.assertEqual(self.sum_tree.data[0], 0)
        # import pdb; pdb.set_trace()
        self.assertAlmostEqual(
            self.sum_tree.tree[self.capacity], priority_values[1] ** self.sum_tree.alpha
        )

        self.assertEqual(
            self.sum_tree.total_sum,
            sum(priority_values**self.sum_tree.alpha),
            "total sum is equal",
        )
        self.assertEqual(
            self.sum_tree.n, 0, "Index should wrap around to 0 after filling capacity"
        )
        print(self.sum_tree.total_sum)

    def test_update(self):
        """Test that update propagates changes correctly in the tree."""
        self.sum_tree.add(1.0, 0)
        leaf_index = self.capacity - 1
        old_value = self.sum_tree.tree[leaf_index]
        old_sum = self.sum_tree.total_sum

        # Update the leaf and check propagation
        self.sum_tree.update(leaf_index, 2.0**self.sum_tree.alpha)
        self.assertNotEqual(self.sum_tree.tree[leaf_index], old_value)
        self.assertEqual(
            self.sum_tree.total_sum,
            self.sum_tree.tree[0],
            "Total sum should be updated",
        )
        self.assertAlmostEqual(
            self.sum_tree.total_sum, old_sum + 2**self.sum_tree.alpha - 1
        )

    def test_get_leaf(self):
        """Test get_leaf method with known cumulative sums."""
        values = np.arange(100_000) / 100_000
        for i, v in enumerate(values):
            self.sum_tree.add(v, i)

        total_sum = self.sum_tree.total_sum
        print(total_sum)
        result = []
        for i in range(100_000):
            leaf_idx, priority, data = self.sum_tree.get_leaf(i * total_sum / 100_000)
            result.append((leaf_idx, priority, data))

        # Verify that each call retrieves the correct indices and priorities
        print(result[0])
        self.assertEqual(len(result), 100_000)
        self.assertEqual(result[0][2], 0)
        self.assertEqual(result[1][2], 1)
        self.assertEqual(result[2][2], 2)
        self.assertEqual(result[3][2], 3)


if __name__ == "__main__":
    # unittest.main()
    # tree = SumTree(100_000)
    # import time
    # start_time = time.time()
    # for i in range(1_000_000):
    #     tree.add(i, i)

    # end_time = time.time()

    # elapsed_time = end_time - start_time
    # print(f"Elapsed time: {elapsed_time} seconds")
    # start_time = time.time()
    # for i in range(1_000_000):
    #     tree.max

    # end_time = time.time()

    # elapsed_time = end_time - start_time
    # print(f"Elapsed time: {elapsed_time} seconds")
    tree = SumTree(100_000)
    ary = np.arange(100_000) / 100_000

    for n, i in enumerate(ary):
        tree.add(i, n)

    print(tree.data)

    sample1, sample2, sample3 = tree.sample(100)

    import seaborn as sns
    import matplotlib.pyplot as plt

    plt.figure()
    # sample2 = np.array(sample1)
    sample2 = sample3
    print(sample3)

    # Define the number of bins
    num_bins = 10
    bins = np.linspace(sample2.min(), sample2.max(), num_bins)

    # Digitize the data into bins
    print(f"max: {sample2.max()} min: {sample2.min()}")
    data_binned = np.digitize(sample2, bins)
    sns.countplot(x=data_binned)
    # print(sample2)
    plt.savefig("dummy.png")
