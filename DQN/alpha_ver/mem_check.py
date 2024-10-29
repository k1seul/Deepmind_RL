import sys
import numpy as np
import copy 

from pympler import asizeof
def check_memory_usage(obj):
    """Check the memory usage of an object."""
    size = sys.getsizeof(obj)
    print(f'The memory usage of the object is: {size/(1024 ** 3)} GB')

# Example usage
if __name__ == "__main__":
    # Checking memory usage for different types of objects
    array_size = 3_000_000
    my_list = [np.ones((84,84,1), dtype=np.float32) for _ in range(array_size)]
    # my_array = np.ones((84,84, array_size))

    print(id(my_list[0]) == id(my_list[1])) 
    # my_list = [1, 2, 3, 4, 5]
    check_memory_usage(my_list)
    # check_memory_usage(my_array)
    from pympler import asizeof
    total_size = asizeof.asizeof(my_list)
    print(f'Total memory usage of the list (including contents): {total_size/(1024 ** 3)} GB')
    




    
