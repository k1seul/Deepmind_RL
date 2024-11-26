import torch
from torch.nn import MSELoss
import numpy as np
from Experience_Buffer import ExperienceMemoryBuffer, ExperienceMemoryBufferPER
from networks import DQN, DQNSimple
from networks import DuelDQN, DualDQNSimple
from collections import deque
import random
from tensorboardX import SummaryWriter
from Env import make_env, wrap_deepmind, make_atari
from save_tools import save_gif_to_tensorboard
import os


class Agent:
    def __init__(self, args, writer: SummaryWriter) -> None:
        # Learning setting
        self.frame_stack = args.frame_stack
        self.epsilon = 1.0
        self.epsilon_end = args.epsilon_end
        self.epsilon_decay_frames = args.epsilon_decay_frames
        self.epsilon_decay_rate = (1.0 - self.epsilon_end) / self.epsilon_decay_frames
        self.gamma = args.gamma
        self.total_frames = args.total_frames
        self.n_steps = args.n_steps
        self.update_freq = args.update_freq
        self.train_start_frame = args.train_start_frame

        # Env init
        self.env_name = args.env_name
        self.network_name = args.network_name
        # self.env = make_env(self.env_name, clip_reward=True)
        env = make_atari(self.env_name)
        self.env = wrap_deepmind(
            env, episode_life=args.episodic_life, clip_rewards=True, frame_stack=False
        )
        self.n_actions = self.env.action_space.n

        # Network init
        self.dual_net = args.dual_net
        self.double_net = args.double_net
        self.per = args.per
        self.noisy_net = args.noisy_net
        if args.env_name == "CartPole-v1":
            self._init_simple_network(args)
        else:
            self._init_network(args)

        # Experience Buffer init
        if args.env_name == "CartPole-v1":
            self._init_experience_replay_simple(args)
        else:
            self._init_experience_replay(args)

        # Optimizer setting
        # self.optim = torch.optim.Adam(self.network.parameters(), lr= args.lr, eps=args.adam_eps)
        self.optim = torch.optim.RMSprop(
            self.network.parameters(), lr=0.00025, alpha=0.95
        )

        # Other setting
        self.device = args.device
        self.writer = writer
        self.performance_eval_freq = args.performance_eval_freq
        self.performance_eval_max_frame = args.performace_eval_max_frame

    def _init_network(self, args):
        if self.double_net:
            if self.dual_net:
                self.network = DuelDQN(n_actions=self.n_actions, args=args)
                self.target_net = DuelDQN(n_actions=self.n_actions, args=args)
            else:
                self.network = DQN(n_actions=self.n_actions, args=args)
                self.target_net = DQN(n_actions=self.n_actions, args=args)

            self._remove_target_grad()
            self.target_net_update_freq = args.target_net_update_freq
            return
        else:
            if self.dual_net:
                self.network = DuelDQN(n_actions=self.n_actions, args=args)
            else:
                self.network = DQN(n_actions=self.n_actions, args=args)
            return

    def _init_simple_network(self, args):
        if self.double_net:
            if self.dual_net:
                self.network = DualDQNSimple(
                    n_actions=self.n_actions, FRAME_STACK=self.frame_stack
                )
                self.target_net = DualDQNSimple(
                    n_actions=self.n_actions, FRAME_STACK=self.frame_stack
                )
            else:
                self.network = DQNSimple(
                    n_actions=self.n_actions, FRAME_STACK=self.frame_stack
                )
                self.target_net = DQNSimple(
                    n_actions=self.n_actions, FRAME_STACK=self.frame_stack
                )

            self._remove_target_grad()
            self.target_net_update_freq = args.target_net_update_freq
            return
        else:
            if self.dual_net:
                self.network = DualDQNSimple(
                    n_actions=self.n_actions, FRAME_STACK=self.frame_stack
                )
            else:
                self.network = DQNSimple(
                    n_actions=self.n_actions, FRAME_STACK=self.frame_stack
                )
            return

    def _remove_target_grad(self):
        for param in self.target_net.parameters():
            param.requires_grad = False

    def _init_experience_replay(self, args):
        if self.per:
            self.mem = ExperienceMemoryBufferPER(
                maxlen=args.buffer_size,
                state_shape=[84, 84],
                n_step=self.n_steps,
                FRAME_STACK=self.frame_stack,
            )
            self.loss = MSELoss(reduce="none")
        else:
            self.mem = ExperienceMemoryBuffer(
                maxlen=args.buffer_size,
                state_shape=[84, 84],
                n_step=self.n_steps,
                FRAME_STACK=self.frame_stack,
            )
            self.loss = MSELoss()

    def _init_experience_replay_simple(self, args):
        if self.per:
            self.mem = ExperienceMemoryBufferPER(
                maxlen=args.buffer_size,
                state_shape=[1, 4],
                n_step=self.n_steps,
                FRAME_STACK=self.frame_stack,
            )
            self.loss = MSELoss(reduce="none")
        else:
            self.mem = ExperienceMemoryBuffer(
                maxlen=args.buffer_size,
                state_shape=[1, 4],
                n_step=self.n_steps,
                FRAME_STACK=self.frame_stack,
            )
            self.loss = MSELoss()

    def _decay_epsilon(self):
        self.epsilon = max(self.epsilon_end, self.epsilon - self.epsilon_decay_rate)

    def reset(self, env=None):
        if env is None:
            state, _ = self.env.reset()
        else:
            state, _ = env.reset()
        self.states_mem = deque(maxlen=self.frame_stack)
        state = np.array(state, dtype=np.uint8)
        for _ in range(self.frame_stack):
            self.states_mem.append(state)
        return state

    def train(self, writer):
        t = 0
        episode_num = 0

        while t < self.total_frames:
            done = False
            truncated = False
            self.state = self.reset()
            total_rewards = 0
            episode_length = 0
            avg_q_values_total = 0
            td_loss_total = 0

            while not (done or truncated):
                state = np.array(self.state, dtype=np.float32)
                action = self.action(np.array(self.states_mem, dtype=np.float32))
                next_state, reward, done, truncated, _ = self.env.step(action)
                total_rewards += reward

                next_state = np.array(next_state, dtype=np.float32)
                action = np.array(action)
                reward = np.array(reward)
                done = np.array(done, dtype=np.bool8)
                self.mem.add_experience(state, action, reward, done)

                if 10 * t % (self.total_frames) == 0:
                    self.save_network(t)

                t += 1
                episode_length += 1
                self.state = next_state
                self.states_mem.append(next_state)

                if t >= self.train_start_frame:
                    if t % self.update_freq == 0:
                        update_samples = self.mem.sample()
                        # import pdb;pdb.set_trace()
                        if update_samples is None:
                            raise BufferError("update sample is empty!")
                        avg_q_values, td_loss = self.update(update_samples)
                        avg_q_values_total += avg_q_values
                        td_loss_total += td_loss
                    self._decay_epsilon()
                    if self.double_net and t % self.target_net_update_freq == 0:
                        self.target_net.load_state_dict(self.network.state_dict())

                if self.per and t > 500_000:
                    self.mem.beta_anneal()

            if episode_num % 100 == 0 and episode_num > 0:
                performance, rendered_imgs = self.evaluate()
                writer.add_scalar("data/performace", performance, episode_num)
                save_gif_to_tensorboard(
                    np.stack(rendered_imgs),
                    writer=writer,
                    global_step=episode_num,
                    tag="gif",
                )

            if episode_num % 10 == 0:
                writer.add_scalar("data/rewards", total_rewards, episode_num)
                writer.add_scalar("data/epsilons", self.epsilon, episode_num)
                writer.add_scalar(
                    "data/steps", round(t / self.total_frames * 100, 1), episode_num
                )
                writer.add_scalar("data/episode_length", episode_length, episode_num)
                writer.add_scalar(
                    "data/td_error", td_loss_total / episode_length, episode_num
                )
                writer.add_scalar(
                    "data/avg_q", avg_q_values_total / episode_length, episode_num
                )

            print(
                f"Episode Number: {episode_num}, rewards: {total_rewards}, epsilon: {self.epsilon}"
            )
            episode_num += 1

    @torch.no_grad()
    def evaluate(self):
        ## evaluate the agent with fixed parameter and zero epsilon for 10 episodes
        eval_episode = 10

        # sum_rewards = 0
        rewards = []
        max_frame = self.performance_eval_max_frame
        # env = make_env(self.env_name, clip_reward=False)
        env = make_atari(self.env_name)
        rendering = []
        env = wrap_deepmind(
            env, episode_life=False, clip_rewards=False, frame_stack=False
        )
        for i in range(eval_episode):
            done = False
            truncated = False
            self.reset(env=env)
            rendered_imgs = []
            total_rewards = 0
            frame = 0
            while not (done or truncated) and frame < max_frame:
                frame += 1
                action = self.action(
                    np.array(self.states_mem, dtype=np.float32), exploit=True
                )
                next_state, reward, done, truncated, _ = env.step(action)
                total_rewards += reward
                self.states_mem.append(next_state)
                rendered_imgs.append(next_state)
            # sum_rewards += total_rewards
            rewards.append(total_rewards)
            rendering.append(rendered_imgs)

        return max(rewards), rendering[rewards.index(max(rewards))]

    def action(self, state, exploit=False):
        if self.noisy_net:
            # if exploit:
            #     self.network.set_zero_noise()
            # else:
            #     self.network.reset()
            obs = torch.tensor(state, dtype=torch.float32).to(self.device)
            q_values = self.network(obs.reshape(1, *obs.shape))
            action = np.argmax(q_values.detach().cpu().numpy())
            return action

        if random.random() < self.epsilon and not exploit:
            action = random.randint(0, self.n_actions - 1)
        else:
            if random.random() < 0.05:
                action = random.randint(0, self.n_actions - 1)
            else:
                obs = torch.tensor(state, dtype=torch.float32).to(self.device)
                q_values = self.network(obs.reshape(1, *obs.shape))
                action = np.argmax(q_values.detach().cpu().numpy())

        return action

    def update(self, samples):
        if self.per:
            states, actions, rewards, next_states, dones, importance_weight = samples
            importance_weight = torch.tensor(
                importance_weight, device=self.device, dtype=torch.float32
            )
        else:
            states, actions, rewards, next_states, dones = samples

        # Convert to tensors and move to device once
        states = torch.tensor(states, device=self.device, dtype=torch.float32)
        actions = torch.tensor(actions, device=self.device, dtype=torch.int64)
        rewards = torch.tensor(rewards, device=self.device, dtype=torch.float32)
        next_states = torch.tensor(next_states, device=self.device, dtype=torch.float32)
        dones = torch.tensor(dones, device=self.device, dtype=torch.float32)

        # if self.noisy_net:
        #     self.network.reset()
        #     self.target_net.reset()

        # Select Q-values for next states with double network if applicable
        Q_target_n_next_states = (self.target_net if self.double_net else self.network)(
            next_states[-1]
        ).detach()

        # Calculate Q_current_states and Q_max_next_actions just once
        Q_current_states = self.network(states[0])[
            torch.arange(actions[0].size(0)), actions[0]
        ]
        Q_next_states = self.network(next_states[-1])
        Q_max_next_actions = torch.argmax(Q_next_states, axis=1).detach()

        # Calculate cumulative discounted reward with broadcasting and torch.cumprod
        discounts = torch.cumprod(self.gamma * (1 - dones[: self.n_steps]), dim=0)
        y_i = rewards[0] + torch.sum(rewards[1 : self.n_steps] * discounts[:-1], axis=0)

        # Add last reward term from the n-step target
        reward_to_add = Q_target_n_next_states[
            range(Q_max_next_actions.shape[0]), Q_max_next_actions
        ]
        y_i += reward_to_add * discounts[-1]

        # Calculate loss
        # import pdb; pdb.set_trace()
        loss = self.loss(Q_current_states, y_i.detach())

        # Apply importance weights if PER is enabled
        loss = (loss * importance_weight).sum() if self.per else loss

        # Optimization step with gradient clipping
        self.optim.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.network.parameters(), max_norm=10.0)
        self.optim.step()

        # Log stats
        avg_q_value = Q_current_states.mean().detach().cpu().numpy()
        td_loss = loss.detach().cpu().numpy() / 32
        if self.per:
            losses = torch.abs(Q_current_states - y_i).detach().cpu().numpy()
            self.mem.update_td_error(losses)

        # if avg_q_value < 0:
        #     import pdb; pdb.set_trace()
        return avg_q_value, td_loss

    def save_network(self, step):
        if not (os.path.exists(f"data/{self.env_name}")):
            os.makedirs(f"data/{self.env_name}")
        torch.save(
            self.network.state_dict(),
            f"data/{self.env_name}/{self.network_name}_{step}.pt",
        )
