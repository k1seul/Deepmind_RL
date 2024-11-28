import argparse
import os, datetime
from Agent import Agent
from run_logic import run_main_logic, run_combinations
import multiprocessing
import copy
import itertools
import ray

## General
parser = argparse.ArgumentParser(description="DQN")
parser.add_argument("--id", type=str, default="default", help="Experiment ID")
parser.add_argument("--seed", type=int, default=10, help="Random seed")
parser.add_argument("--device", type=str, default="cuda", help="network run device")
parser.add_argument("--ray", action='store_true', help="run multiple envs")
### Don't use, gym is faster
parser.add_argument("--env_pool", action="store_true", help="use env_pool to make env instead of gymnasium (envs are not in ALE name space)")
parser.add_argument("--torch_buffer", action="store_true", help="use gpu experience buffer implemented with pytorch")
parser.add_argument(
    "--performance-eval-freq",
    type=str,
    default=1e5,
    help="performace evaluation frequency number",
)
parser.add_argument(
    "--performace-eval-max-frame",
    type=int,
    default=10_000,
    help="performance evaluation max frame",
)
parser.add_argument(
    "--network-name", type=str, default="DQN", help="network architecture"
)
parser.add_argument(
    "--pid-file", default="running_pids.txt", help="name of pids recording text file"
)


## Network architectures
parser.add_argument(
    "--double-net", action="store_true", help="double network with target network"
)
parser.add_argument("--per", action="store_true", help="prioritized experience replay")
parser.add_argument("--dual-net", action="store_true", help="dualing network")
parser.add_argument(
    "--noisy-net", action="store_true", help="noisy net of dqn neural network"
)


## Environment Setting
parser.add_argument(
    "--env_name", type=str, default="ALE/Breakout-v5", help="name of a gym environment"
)
parser.add_argument(
    "--episodic_life", action="store_true", help="turns on episodic life"
)

## Experience replay setting
parser.add_argument(
    "--buffer-size",
    type=int,
    default=1_000_000,
    help="size of experience replay buffer",
)


## Learning setting
parser.add_argument(
    "--frame-stack", type=int, default=4, help="frame stacking for state inputs"
)
parser.add_argument(
    "--target-net-update-freq",
    type=int,
    default=1e4,
    help="update frequency of target network",
)
parser.add_argument(
    "--epsilon-end", type=float, default=0.1, help="epsilon greedy end value"
)
parser.add_argument(
    "--epsilon_decay_frames", type=int, default=1e6, help="epsilon linear decay steps"
)
parser.add_argument(
    "--gamma", type=float, default=0.99, help="gamma time decay of cummulative reward"
)
parser.add_argument(
    "--total-frames", type=int, default=2e7, help="total learning step of the agent"
)
parser.add_argument("--n-steps", type=int, default=1, help="n step learning number")
parser.add_argument(
    "--update-freq", type=int, default=4, help="update frequency of main network"
)
# parser.add_argument('--lr', type=float, default= 1e-4, help='learning rate of adam optimizer')
parser.add_argument("--rmsprop", action='store_true', help='uses rmsprop optimizer')
parser.add_argument(
    "--lr", type=float, default=6.25 * 1e-5, help="learning rate of adam optimizer"
)
parser.add_argument(
    "--adam-eps", type=float, default=1.5e-4, help="epsilon of adam optimizer"
)
parser.add_argument(
    "--weight_decay", type=float, default=1e-4, help='weight decay of the optimizer'
)
parser.add_argument(
    "--momentum", type=float, default=0.0, help='momentum value of the optimizer'
)
parser.add_argument(
    "--train-start-frame", type=int, default=5e3, help="training start frame number"
)

parser.add_argument(
    "--game_set", type=int, default=0, help="training game set" 
)

@ray.remote(num_gpus=0.05)
def run_task(args, game, rng):
    args.env_name = game
    args.episodic_life = True
    args.seed = rng
    run_main_logic(args)

args = parser.parse_args()
# Iterate over and print all arguments
for arg, value in vars(args).items():
    print(f"{arg}: {value}")

if args.game_set == 0: 
    games = [
    "ALE/Breakout-v5",
    "ALE/SpaceInvaders-v5",
    "ALE/Alien-v5",
    "ALE/TimePilot-v5",
    "ALE/Asterix-v5"]
    epsodic_lifes = [True] * len(games)
elif args.game_set == 1:
    games = [
    ] 
    epsodic_lifes = [True, True, True]


if args.double_net:
    args.network_name += "_double"
if args.per:
    args.network_name += "_per"
if args.dual_net:
    args.network_name += "_dual"
if args.noisy_net:
    args.network_name += "_noisy"


args.network_name += args.id
if not(os.path.exists(args.pid_file)):
    with open(args.pid_file, "w") as f:
        f.write("Game PID Log\n")


results_dir = os.path.join("results", args.id)
if not os.path.exists(results_dir):
    os.makedirs(results_dir)
print(results_dir)

# run_main_logic(args)
rngs = [0,1,2]
processes = []



if args.ray: 
    ray.init(num_gpus=1)

    # Submit tasks
    tasks = [run_task.remote(args, game, rng) 
            for game, rng in itertools.product(games, rngs)]
    ray.get(tasks)
else:
    run_main_logic(args)



# # Start all processes
# for process in processes:
#     process.start()

# # Wait for all processes to finish
# for process in processes:
#     process.join()
