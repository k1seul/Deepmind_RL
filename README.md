# Deepmind_RL
Reimplementation of Deepmind Reinforcement learning models

## Reimplementation Schedule
1) [] [DQN(2015)](http://arxiv.org/abs/1312.5602))
2) [] [DDQN(2015)](http://arxiv.org/abs/1509.06461)
3) [] [PER(2015)](http://arxiv.org/abs/1511.05952)
4) [] [Dueling(2015)](http://arxiv.org/abs/1511.06581)
5) [] [Noisy Networks(2017)](https://arxiv.org/abs/1706.10295)
6) [] [C51(2017)](https://arxiv.org/abs/1707.06887)

Tasks chosen in each models will be tasks with significant different pointed out by the paper.
I will also describe each parameters and preprocessing of atari envrionment needed for well training.

### Hardwhere Spec
* CPU: RYZEN 5 5600X 6-Core Processor
* RAM: ddr4 64GB
* GPU: RTX4070(12GB)

Environment used is the Arcade Learning Environment([ALE(https://ale.farama.org/)) which was ran on the cpu. Although computation times might vary, running 1 million frames(4 skipped frames, so 4 million frames to be exact) took about 3 hours with about 10GB of memory used(1 million episode replay buffer). With multiprocessing, I could run about 5 environment with 80~90% CPU(4.36/3.7GHz), 82%(51GB) Memory used and 1.7 GB GPU memory use. Because of the main network being small (3 conv layers and 2 fc layers) and environment running on cpu, the main bottleneck for the computation time and memory used were heavly dependant on cpu.

## DQN(2015) 
### Tasks
* Breakout ([ALE/Breakout-v5](https://ale.farama.org/environments/breakout/), with repeat_action_prob = 0) 
* Pong ([ALE/Pong-v5](https://ale.farama.org/environments/pong/), with repeat_action_prob = 0)
* Seaquest ([ALE/Seaquest-v5](https://ale.farama.org/environments/seaquest/), with action_repeat_prob = 0)
* SpaceInvaders ([ALE/SpaceInvaders-v5](https://ale.farama.org/environments/space_invaders/), with action_repeat_prob = 0)
* Zaxxon ([ALE/Zaxxon-v5](https://ale.farama.org/environments/zaxxon/), with action_repeat_prob = 0)

Note:
1) Some of the enviroments(such as breakout) trained better by using episodic life(termination of environment after single failure) but some enviroment such as pong did not. I will specifiy which enviroment was used having episodic life.
2) Also metric used in this reimplementation is maximum rewards of 10 agents with non-episodic life. This can be misleading estimation of point stastics showing only agents with best performance. But since the objective of this project is to reimplement and show the exact performance metric shown on each of the paper, I will follow the maximum performance metric.

### Reimplementation Targets 
![image](https://github.com/user-attachments/assets/9229a82b-11b0-4d5b-99be-47f8d55f0280)

![image](https://github.com/user-attachments/assets/ab78f449-8692-470e-a9fc-54acf55fa7de)












References
----------

[1] [Rainbow: Combining Improvements in Deep Reinforcement Learning](https://arxiv.org/abs/1710.02298)  
[2] [Playing Atari with Deep Reinforcement Learning](http://arxiv.org/abs/1312.5602)  
[3] [Deep Reinforcement Learning with Double Q-learning](http://arxiv.org/abs/1509.06461)  
[4] [Prioritized Experience Replay](http://arxiv.org/abs/1511.05952)  
[5] [Dueling Network Architectures for Deep Reinforcement Learning](http://arxiv.org/abs/1511.06581)  
[6] [Reinforcement Learning: An Introduction](http://www.incompleteideas.net/sutton/book/ebook/the-book.html)  
[7] [A Distributional Perspective on Reinforcement Learning](https://arxiv.org/abs/1707.06887)  
[8] [Noisy Networks for Exploration](https://arxiv.org/abs/1706.10295)  
[9] [When to Use Parametric Models in Reinforcement Learning?](https://arxiv.org/abs/1906.05243)  
