## Deep reinforcement learning: Navigation

### Goal

The goal of this project is to train an agent to navigate and collect yellow bananas while avoiding blue bananas in a square world with no obstacles. 

The **state space** has 37 dimensions and contains the agent's velocity, along with ray-based perception of objects around agent's forward direction. 

The **action space** is discrete, with four actions corresponding to:
- **`0`** - move forward.
- **`1`** - move backward.
- **`2`** - turn left.
- **`3`** - turn right.

A **reward** of +1 is provided for collecting a yellow banana, and a reward of -1 is provided for collecting a blue banana. 

The problem is considered solved when the agent gets the average score of +13 over 100 consecutive episodes.

### Methods
This solution implements a vanilla [Deep Q-Network](https://deepmind.com/research/publications/human-level-control-through-deep-reinforcement-learning), a value-based method in deep reinforcement learning. 

#### Network architecture
As the state vector in this version of the task describes agent's and objects' positions directly, there are no convolutional layers. The input into the network is a state vector of length 37.

The Q-Network consists of three fully connected layers:

| Layer | Dimensions | Activation |
|:-----:|:----------:|:----------:|
|   1   |\[37, 128\]        | ReLU       |
|   2   |\[128, 128\]      | ReLU       |
|   3   | \[128, 4\]        |    None        |


#### Parameters

The solution, as implemented and saved in the accompanying `checkpoint.pth` file, was obtained with the following parameters:

| Parameter | Value |Explanation |
|:-----|:----------|:----------|
|BUFFER_SIZE | int(1e5) |replay buffer size|
|BATCH_SIZE | 64        | minibatch size|
|GAMMA | 0.99           | discount factor|
|TAU | 3e-3              |for soft update of target parameters|
|LR | 5e-4               |learning rate |
|UPDATE_EVERY | 10       | how often to update the network|
|eps_start | 1.0         |starting value of $\epsilon$, the likelihood of random action|
|eps_end|0.01            | the multiplicative rate of decay in $\epsilon$ with each period|
|eps_decay|0.99          | the minimum value of $\epsilon$ |
|max_t | 50000| maximum number of steps per episode

### Results

The figure below and the training callback output below illustrate how the algorithm converges to maintaining an average over 13 points for 100 consecutive episodes  over the course of the 401 episodes. 

    Episode 1	Average Score: 0.002
    Episode 100	Average Score: 2.21
    Episode 200	Average Score: 8.48
    Episode 300	Average Score: 11.04
    Episode 400	Average Score: 12.96
    Episode 401	Average Score: 13.06
    Environment solved in 301 episodes!	Average Score: 13.06


### Potential areas for further work

The performance on this task could be further improved by applying more advanced algorithms, that improve on the common pitfalls of the vanilla DQN.  Papers that propose potential alternatives include:
1. [Deep Reinforcement Learning with Double Q-learning](https://arxiv.org/abs/1509.06461)
2. [Prioritized Experience Replay](https://arxiv.org/abs/1511.05952)
3. [Rainbow: Combining Improvements in Deep Reinforcement Learning](https://arxiv.org/abs/1710.02298)

Instead of being provided a 37-dimensional vector that encodes the state, the actor could also be taught to learn directly from pixels of the task by passing those through CNN layers.

