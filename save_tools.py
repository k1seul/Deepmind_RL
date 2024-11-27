from tensorboardX import SummaryWriter
import numpy as np
import torch
from PIL import Image
import os


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


def save_numpy_to_gif(
    frames: np.ndarray,
    dir: str = 'results/pong.gif',
    final_frame_dir: str = 'results/final_frame.png',
    fps: int = 10
):
    """
    Saves a sequence of frames as a GIF file and the final frame as a PNG image.

    Args:
        frames (np.ndarray): A NumPy array of shape (n_frames, height, width). If grayscale, values should be in [0, 255].
        dir (str): File path to save the GIF.
        final_frame_dir (str): File path to save the final frame as a PNG.
        fps (int): Frames per second for the GIF animation.
    """
    # Ensure directory exists
    os.makedirs(os.path.dirname(dir), exist_ok=True)
    os.makedirs(os.path.dirname(final_frame_dir), exist_ok=True)
    # frames = np.repeat(frames[:, np.newaxis, :, :], repeats=3, axis=1)

    # Normalize frames if they are not in 0-255
    if frames.dtype != np.uint8:
        frames = (255 * (frames - np.min(frames)) / (np.max(frames) - np.min(frames))).astype(np.uint8)

    # Create a list of PIL Images from the frames
    images = [Image.fromarray(frame) for frame in frames]

    # Save the GIF
    images[0].save(
        dir,
        save_all=True,
        append_images=images[1:],
        duration=int(1000 / fps),
        loop=0
    )

    # Save the final frame as a PNG
    final_frame = images[-1]
    final_frame.save(final_frame_dir)
