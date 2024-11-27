# DQN Wrappers Used

* Clip Reward
Clips rewards to three discrete value of -1, 0, 1 based on their signs.

* No-op Actions
30 No-op action(0) on reset. Helps to randomize starting state for diverse exploration

* Fire Reset
Presses the fire button on reset, helps to get out of POMDP state and better directed exploration. For example in the breakout environment, fire button must be pressed so that the ball is shown, however
in after some training the agent might prefer to move and never press the fire button.

* Episodic Life(depends on the environment)
In some of the environments agent have lifes so that it can continue to play after single failure. However termination after only a single failure might be benifitial for the agent to learn faster in earlier stages.
(Experience replay buffer will be filled with more early stage data) This might help the agent to learn faster if earler stage data can be used to generalize to latter stages. However This is environment dependant.

* Max and Skip Environment
Frame skipping for faster training. Repeats action for each frames. The environment returns maximum value of each pixels as state. This is because of atari's implementation of flikering projectiles (such as in space invaders) which might fliker with similar
frequncy to frame skip number which results not showing up in states.

* Warp Frames
Simple wrapper to convert 210 * 110 * 3 RGB images to 84 * 84 grayscale images.  

