import os
import subprocess

def run_tensorboard(log_dir='runs', port=6006):
    """
    Function to run TensorBoard on the specified log directory.
    
    Args:
    log_dir (str): The directory where TensorBoard logs are stored (default is 'runs').
    port (int): The port to run TensorBoard on (default is 6006).
    """
    try:
        # Ensure the log directory exists
        if not os.path.exists(log_dir):
            print(f"Log directory '{log_dir}' does not exist.")
            return

        # Run TensorBoard using subprocess
        print(f"Running TensorBoard on log directory: {log_dir}")
        subprocess.run(['tensorboard', '--logdir', log_dir, '--port', str(port)], check=True)

    except subprocess.CalledProcessError as e:
        print(f"Error running TensorBoard: {e}")

if __name__ == "__main__":
    run_tensorboard()
