import torch
from tensorboardX import SummaryWriter
from datetime import datetime
import os, sys
import itertools, multiprocessing
import time
from Agent import Agent


N_STEP = 3
FRAME_STACK = 4
DOUBLE_NET = True
DUAL_NET = False
PER = True
TOTAL_FRAMES = 50_000_000

NETWORK_NAME = "DQN"

if DUAL_NET:
    NETWORK_NAME += "_DUAL_net"

if DOUBLE_NET:
    NETWORK_NAME += "_Double_net"

if PER:
    NETWORK_NAME += "_PER"

if N_STEP > 1:
    NETWORK_NAME += f"_{N_STEP}step"


def run_combinations():
    # Define all possible values for the variables
    n_step_values = [1, 3]  # Example values, modify as needed
    frame_stack_values = [1, 4]  # Example values
    double_net_values = [False, True]
    dual_net_values = [False, True]
    per_values = [False]
    total_frames = 1_000_00

    # Generate all combinations
    combinations = list(
        itertools.product(
            n_step_values,
            frame_stack_values,
            double_net_values,
            dual_net_values,
            per_values,
        )
    )

    # Prepare arguments for multiprocessing
    tasks = []
    for n_step, frame_stack, double_net, dual_net, per in combinations:
        network_name = "DQN"

        if dual_net:
            network_name += "_DUAL_net"

        if double_net:
            network_name += "_Double_net"

        if per:
            network_name += "_PER"

        if n_step > 1:
            network_name += f"_{n_step}step"

        tasks.append(
            (n_step, frame_stack, double_net, dual_net, per, network_name, total_frames)
        )

    # Use multiprocessing to run tasks in parallel
    num_processes = min(2, len(tasks))  # Use all available cores or task count
    with multiprocessing.Pool(processes=num_processes) as pool:
        pool.starmap(run_main_logic, tasks)


def run_main_logic(args):
    """
    Placeholder function for the main logic.
    Replace with actual code to train/evaluate the model.
    """
    print(
        f"Executing main logic for {args.network_name} with TOTAL_FRAMES={args.total_frames}"
    )
    # Your main code goes here
    env_name = args.env_name
    now = datetime.now()
    pid = multiprocessing.current_process().pid
    print(
        f"Running main logic for game {args.env_name} (PID: {pid}) with episodic_life: {args.episodic_life}"
    )

    # Simulate work
    time.sleep(2)

    # Log PID to the file
    with open(args.pid_file, "a") as f:
        f.write(
            f"Game: {args.env_name}, Episodic Life: {args.episodic_life}, PID: {pid}\n"
        )

    log_dir = f"runs/{env_name}_{args.network_name}_{args.seed}"
    writer = SummaryWriter(log_dir=log_dir)
    agent = Agent(args, writer=writer)

    writer.add_hparams(vars(args), metric_dict={})
    agent.train(writer)
    os.makedirs(f"data/{args.env_name}")
    torch.save(
        agent.network.state_dict(), f"data/{args.env_name}/{args.network_name}.pt"
    )
    writer.close()
    time.sleep(1)
    print(f"Completed: {args.network_name}")
