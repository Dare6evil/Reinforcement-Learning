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
env = gymnasium.make("Pendulum-v1", render_mode="human")
episodes = 1000
gamma = 0.98
lr_pi = 0.0002
lr_q = 0.0005
mu = 0
# max_total_reward = 0
reward_history = [0] * episodes
runs = 5
sigma = 0.2
sync_interval = 1
theta = 0.15
for run in range(1, 1 + runs):
    pi = modules.DeterministicPolicy(
        *env.action_space.shape, *env.observation_space.shape
    )
    optimizer_pi = torch.optim.Adam(pi.parameters(), lr_pi)
    pi.to(device)
    pi_target = modules.DeterministicPolicy(
        *env.action_space.shape, *env.observation_space.shape
    )
    pi_target.to(device)
    q = modules.Q(1, env.action_space.shape[0] + env.observation_space.shape[0])
    optimizer_q = torch.optim.Adam(q.parameters(), lr_q)
    q.to(device)
    q_target = modules.Q(1, env.action_space.shape[0] + env.observation_space.shape[0])
    q_target.to(device)
    replay_buffer = data.ReplayBuffer(buffer_size)
    state, _ = env.reset()
    for episode in range(episodes):
        total_reward = 0
        x = mu * numpy.ones(env.action_space.shape[0])
        while True:
            x += (mu - x) * theta + numpy.random.normal(
                size=env.action_space.shape[0]
            ) * sigma
            action = (
                numpy.array([pi(torch.Tensor(state).to(device)).detach().item()]) + x
            )
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
                loss_pi = -q(
                    torch.cat(
                        (
                            pi(torch.Tensor(numpy.array(state_batch)).to(device)),
                            torch.Tensor(numpy.array(state_batch)).to(device),
                        ),
                        1,
                    )
                ).mean()
                optimizer_pi.zero_grad()
                loss_pi.backward()
                optimizer_pi.step()
                loss_q = torch.nn.functional.mse_loss(
                    (1 - torch.Tensor(done_batch).to(device).unsqueeze(1))
                    * gamma
                    * q_target(
                        torch.cat(
                            (
                                pi_target(
                                    torch.Tensor(numpy.array(next_state_batch)).to(
                                        device
                                    )
                                ),
                                torch.Tensor(numpy.array(next_state_batch)).to(device),
                            ),
                            1,
                        )
                    ).detach()
                    + torch.Tensor(reward_batch).to(device).unsqueeze(1),
                    q(
                        torch.cat(
                            (
                                torch.Tensor(numpy.array(action_batch)).to(device),
                                torch.Tensor(numpy.array(state_batch)).to(device),
                            ),
                            1,
                        )
                    ),
                )
                optimizer_q.zero_grad()
                loss_q.backward()
                optimizer_q.step()
            total_reward += reward
            if done:
                state, _ = env.reset()
                break
            state = next_state
        # if max_total_reward < total_reward:
        #     max_total_reward = total_reward
        #     torch.save(q.state_dict(), "DDPG.pth")
        # elif max_total_reward == total_reward:
        #     torch.save(q.state_dict(), "DDPG.pth")
        if not episode % sync_interval:
            pi_target.load_state_dict(pi.state_dict())
            q_target.load_state_dict(q.state_dict())
        reward_history[episode] += (total_reward - reward_history[episode]) / run
env.close()
pyplot.plot(reward_history)
pyplot.xlabel("Episode")
pyplot.ylabel("Total Reward")
pyplot.show()
