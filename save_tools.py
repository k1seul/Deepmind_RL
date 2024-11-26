from tensorboardX import SummaryWriter
import numpy as np
import torch


def save_gif_to_tensorboard(
    frames: np.ndarray, writer: SummaryWriter, tag: str, global_step: int = 0
):
    """
    Saves a sequence of frames as a GIF in TensorBoard.

    Args:
        frames (np.ndarray): A NumPy array of shape (n_frames, height, width).
        writer (SummaryWriter): TensorBoard SummaryWriter object.
        tag (str): Tag under which to store the GIF in TensorBoard.
        global_step (int): Global step value to record with the GIF.
    """
    # Write the video to TensorBoard
    frames = np.repeat(frames[:, np.newaxis, :, :], repeats=3, axis=1)
    frames_tensor = torch.from_numpy(frames)  # Shape: (1000, 1, 84, 84)
    writer.add_image(
        f"{tag} Image", frames_tensor[-1], global_step=global_step
    )  # Log the last frame as a sample, for checking the score
    writer.add_video(
        f"{tag} Video", frames_tensor.unsqueeze(0), fps=10, global_step=global_step
    )
