
# Deep Mind DQN(2015) 리뷰 (+ code) 
[Human-level control through deep reinforcement learning](https://www.nature.com/articles/nature14236)
[Deep Mind](https://deepmind.google/discover/blog/deep-reinforcement-learning/)
#### Nature 
### Selection Reason:
* 표면적: 딥러닝을 이용한 강화학습의 시작. 24년인 지금은 이 때보다 강화학습이 비주류이지만, 언제나 강화학습의 시간이 올 것같아 근본부터 리뷰.

* 내면적: 2024년 Oct 당시 인용 32,257회, Deep Mind 여러 유명인들이 참여

### Author Information
#### First: [Volodymyr Mnih](https://scholar.google.com/citations?user=rLdfJ1gAAAAJ&hl=en)

Deep Mind 연구원. 최근은 논문이 많이 없으심. 계속 강화학습 또는 다른 방법론으로 [Control](https://en.wikipedia.org/wiki/Control_theory)에 대한 연구 진행중
##### Most cited papers
* [Human-level control through deep reinforcement learning](https://www.nature.com/articles/nature14236)]
* [[Playing Atari with Deep Reinforcement Learning](https://scholar.google.com/citations?view_op=view_citation&hl=en&user=rLdfJ1gAAAAJ&citation_for_view=rLdfJ1gAAAAJ:WF5omc3nYNoC)]
* [[Asynchronous methods for deep reinforcement learning](https://scholar.google.com/scholar?cluster=14460380466928185185&hl=en&oi=scholarr)]

#### Last: [Martin Riedmiller](https://sites.google.com/view/riedmiller/home)
이전 Albert-Ludwigs-University Freiburg 교수이자 현 Deep mind 연구자.
현재 강화학습을 실생활에서 활용에 대한 연구중(list organized with chat GPT)
-   **Probabilistic Inference and Preference Optimization**: Applying probabilistic methods to optimize agent preferences, which could enhance performance in complex environments by improving decision-making processes.
    
-   **Language Model-Guided Sub-Goal Generation**: Leveraging large language models to assist in generating intermediate goals or "sub-goals" for controlling RL agents, which might improve task performance and adaptability.
    
-   **Control in Physical Systems (e.g., Magnetic Confinement)**: Using neural networks to control magnetic fields in devices like tokamaks (used in fusion research) and robots, highlighting the practical applications of RL in controlling complex systems.
    
-   **Auto-curriculum and Sim-to-Real Transfer**: Techniques such as demonstration-led curricula allow RL models to learn tasks efficiently by simulating real-world environments, particularly in robotics.
    
-   **Inverse Reinforcement Learning for Language**: Focusing on scalable methods to teach models how to imitate language, which has implications for human-like interactions in RL-based systems.
    
-   **Model Efficiency and Scalability in RL**: Addressing challenges in data efficiency and scalability, which are essential for deploying RL models in larger and more complex tasks.
    
-   **Application to Robotics and Physical Systems**: Many of these papers are oriented toward the practical, real-world deployment of RL, from controlling robots to manipulating fluid dynamics, emphasizing how RL can transform fields like robotics and control engineering

## Abstract  

지금까지 강화학습 모델의 문제점은 실제 complex한 환경에서 작용을 못한다는 점에 있다. 이는,
* Optimal한 value and Q function을 찾기위해서는 정해진 policy 별 가능한 모든 states에 [Monte Carlo search](https://en.wikipedia.org/wiki/Monte_Carlo_tree_search) 또는 [TD학습](https://en.wikipedia.org/wiki/Temporal_difference_learning)을 진행하여하는데, 가능한 state가 무수히 많을 시(Atari에서는 84 * 110 * 3 pixel에서 각 0~255 수치 가능 따라서 $(3*84*110)^{256} = 27720^{256}$ 불가능한 탐색숫자임.) exact 한 함수를 못 구할 수 있다. 또한 policy iteration 및 value iteration을 동시에 시행하는 [Generalized Policy Iteration](http://incompleteideas.net/book/ebook/node46.html) 경우도 이러한 large scale에 수렴 가능한 함수를 만드는데 어려움을 겪었다. 

저자는 Artificial Neural Network 더나아가 Deep Neural Network가 이러한 large scale한 환경에서 [functional approximation](https://arxiv.org/pdf/1610.04161) 이 뛰어난 점에 착안을 하여 연구를 시작하고 성공했다고 추정한다. 


## Background 
강화학습 관점으로 보면, 이 논문은 매우 기본적인 지식만 사용하고 있다. 각 요소의 정의는 다음과 같다 

* Environment: $\mathcal{E} : (s \times a) \rightarrow s \times r$
* Action: $a_t \in \mathcal{A} = \{1, \dots, K\}$
* Observation (image): $x_t \in \mathbb{R^d}$
* State: a stack of 4  continued states to induce MDP state $s_t = [x_{t-3}, x_{t-2}, x_{t-1}, x_{t}]$
* Rewards: $r_t \in \mathbb{R}$ (1 if positive else 0 for simplification)
* Return: $R_t = \sum_{t'=t}^T\gamma^{t'-t}r_{t'}$ ($T$ is the termination time step)
* Optimal $Q$: $Q^*(s,a) = \max_{\pi}\mathbb{E}[R_t|s_t = s, a_t = a, \pi]$
* Optimal $Q$ obeys the Bellman Equations 
$$Q^*(s,a) = \mathbb{E}_{s' \sim \mathcal{E}}\bigl[ r + \gamma \max_{a'} Q^*(s', a')| s, a\bigr]$$

So, the Bellman equation can be used to iteratively update the $Q$ function
$$Q_{i+1}(s,a) = \mathbb{E} \bigl[ r + \gamma \max_{a'}Q_i(s', a')|s,a\bigr]$$
The true optimal $Q$ function can be calculated using functional approximation
$$Q(s,a;\theta) \approx Q^*(s,a)$$

So the loss becomes 
$$L_i(\theta_i) = \mathbb{E}_{s, a \sim p(\cdot)}\bigl[ (y_i - Q(s,a;\theta_i))^2\bigr]$$

Taking the gradient,
$$\nabla_{\theta_i}L_i(\theta_i) = \mathbb{E}_{s, a \sim p(\cdot); s' \sim \mathcal{E}} \bigl[ (r + \gamma\max_{a'}Q(s',a';\theta_{i-1}) - Q(s, a;\theta_i))\nabla_{\theta_i}Q(s,a;\theta_i)\bigr]$$

Which does not need to model the world hence \textbf{model-free}.

## Deep Reinforcement Learning

* Using Deep Neural Network for optimal $Q^*$ functional approximation 
* Experience Replay of 1,000,000 $\mathcal{D} = e_1, \dots, e_N, \text{ where }e_t = (s_{t},a_t,r_t,s_{t+1})$ memory.
* Batch of 32 experiences, $e \sim \mathcal{D}$ which are randomly pooled, are used for Q learning
* Uses $\epsilon$-greedy policy, hence off-policy learning (learning-policy $neq$ behavior-policy)

### Algorithm 1: Deep Q-learning with Experience Replay

1. Initialize replay memory $D$ to capacity $N$.
2. Initialize action-value function $Q$ with random weights.
3. For episode $= 1$ to $M$:
   1. Initialize sequence $s_1 = \{x_1\}$ and preprocessed sequence $\phi_1 = \phi(s_1)$.
   2. For $t = 1$ to $T$:
      1. With probability $\epsilon$, select a random action $a_t$; otherwise select $a_t = \max_a Q^*(\phi(s_t), a; \theta)$.
      2. Execute action $a_t$ in the emulator and observe reward $r_t$ and image $x_{t+1}$.
      3. Set $s_{t+1} = (s_t, a_t, x_{t+1})$ and preprocess $\phi_{t+1} = \phi(s_{t+1})$.
      4. Store transition $(\phi_t, a_t, r_t, \phi_{t+1})$ in $D$.
      5. Sample random minibatch of transitions $(\phi_j, a_j, r_j, \phi_{j+1})$ from $D$.
      6. Set 
         $$
         y_j =
         \begin{cases}
         r_j & \text{for terminal } \phi_{j+1} \\
         r_j + \gamma \max_{a'} Q(\phi_{j+1}, a'; \theta) & \text{for non-terminal } \phi_{j+1}
         \end{cases}
         $$
      7. Perform a gradient descent step on $(y_j - Q(\phi_j, a_j; \theta))^2$ according to equation (3).
4. End for.

### Model Architecture

## Summary of Deep Q-Networks for Atari Games

Working with raw Atari frames can be computationally intensive. To address this, the following preprocessing steps are applied:

| Step                          | Description                                                                                     |
|-------------------------------|-------------------------------------------------------------------------------------------------|
| **Input Size**                | Raw frames are 210 × 160 pixels with a 128-color palette.                                     |
| **Grayscale Conversion**      | Raw frames are converted to grayscale.                                                         |
| **Down-sampling**             | Images are down-sampled to 110 × 84 pixels.                                                   |
| **Cropping**                  | An 84 × 84 region capturing the playing area is cropped, as square inputs are required.       |
| **Stacking Frames**           | The function \( \phi \) preprocesses the last four frames of history and stacks them.         |

### Neural Network Architecture

To parameterize the Q-value, the architecture uses the following structure:

| Layer                        | Description                                                                                |
|------------------------------|--------------------------------------------------------------------------------------------|
| **Input Layer**              | Input size of 84 × 84 × 4 (processed image).                                             |
| **First Hidden Layer**       | 16 convolutional filters of size 8 × 8 with a stride of 4, followed by a rectifier nonlinearity. |
| **Second Hidden Layer**      | 32 convolutional filters of size 4 × 4 with a stride of 2, followed by a rectifier nonlinearity. |
| **Final Hidden Layer**       | Fully connected layer with 256 rectifier units.                                           |
| **Output Layer**             | Fully connected linear layer with one output for each valid action (4 to 18 actions depending on the game). |

### Advantages

- **Single Forward Pass:** The architecture allows computation of Q-values for all actions in a state with only one forward pass, improving efficiency.

This approach is referred to as Deep Q-Networks (DQN).


## Implementation and Result

Look at the [DQN code base](https://github.com/k1seul/Deepmind_RL/tree/main/DQN) I made!
