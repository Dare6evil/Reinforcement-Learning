from matplotlib import pyplot
import os
import sys

sys.path.append(os.pardir)
import data
import gymnasium
import modules
import numpy
import torch

batch_size = 32
buffer_size = 100000
device = torch.device("cpu" if not torch.cuda.is_available() else "cuda:0")
env = gymnasium.make("CartPole-v1", render_mode="human")
episodes = 1000
annealing_num_steps = episodes // 2
epsilon_end = 0.1
epsilon_init = 1.0
gamma = 0.98
lr = 0.0005
# max_total_reward = 0
reward_history = [0] * episodes
runs = 5
sync_interval = 20
for run in range(1, 1 + runs):
    epsilon = epsilon_init
    q = modules.DuelingQ(env.action_space.n, *env.observation_space.shape)
    optimizer = torch.optim.Adam(q.parameters(), lr)
    q.to(device)
    q_target = modules.DuelingQ(env.action_space.n, *env.observation_space.shape)
    q_target.to(device)
    replay_buffer = data.ReplayBuffer(buffer_size)
    state, _ = env.reset()
    for episode in range(episodes):
        total_reward = 0
        while True:
            if epsilon < numpy.random.rand():
                action = q(torch.Tensor(state).to(device)).argmax().item()
            else:
                action = numpy.random.choice(env.action_space.n)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            replay_buffer.add(state, action, reward, next_state, int(done))
            if batch_size < len(replay_buffer):
                (
                    state_batch,
                    action_batch,
                    reward_batch,
                    next_state_batch,
                    done_batch,
                ) = replay_buffer.sample(batch_size)
                loss = torch.nn.functional.mse_loss(
                    (1 - torch.Tensor(numpy.array(done_batch)).to(device))
                    * gamma
                    * q_target(torch.Tensor(numpy.array(next_state_batch)).to(device))
                    .detach()
                    .max(1)
                    .values
                    + torch.Tensor(reward_batch).to(device),
                    q(torch.Tensor(numpy.array(state_batch)).to(device))[
                        numpy.arange(batch_size), action_batch
                    ],
                )
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            total_reward += reward
            if done:
                state, _ = env.reset()
                break
            state = next_state
        epsilon = max(
            epsilon - (epsilon_init - epsilon_end) / annealing_num_steps, epsilon_end
        )
        # if max_total_reward < total_reward:
        #     max_total_reward = total_reward
        #     torch.save(q.state_dict(), "Dueling DQN.pth")
        # elif max_total_reward == total_reward:
        #     torch.save(q.state_dict(), "Dueling DQN.pth")
        if not episode % sync_interval:
            q_target.load_state_dict(q.state_dict())
        reward_history[episode] += (total_reward - reward_history[episode]) / run
env.close()
pyplot.plot(reward_history)
pyplot.xlabel("Episode")
pyplot.ylabel("Total Reward")
pyplot.show()
