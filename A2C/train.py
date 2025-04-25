import os
import sys

sys.path.append(os.pardir)
import gymnasium
import modules
import numpy
import torch

device = torch.device("cpu" if not torch.cuda.is_available() else "cuda:0")
env_batch = []
gamma = 0.98
lr_pi = 0.0002
lr_v = 0.0005
n = 2
for i in range(n):
    env = gymnasium.make("CartPole-v1", render_mode="human")
    env_batch.append(env)
runs = 1
for run in range(1, 1 + runs):
    env = env_batch[0]
    pi = modules.Policy(env.action_space.n, *env.observation_space.shape)
    optimizer_pi = torch.optim.Adam(pi.parameters(), lr_pi)
    pi.to(device)
    state_batch = []
    for i in range(n):
        env = env_batch[i]
        state, _ = env.reset()
        state_batch.append(state)
    v = modules.V(*env.observation_space.shape)
    optimizer_v = torch.optim.Adam(v.parameters(), lr_v)
    v.to(device)
    for _ in range(1):  #
        action_batch = []
        done_batch = []
        next_state_batch = []
        reward_batch = []
        for i in range(n):
            env = env_batch[i]
            state = state_batch[i]
            probs = pi(torch.Tensor(state).to(device))
            action = numpy.random.choice(len(probs), p=probs.detach().cpu().numpy())
            action_batch.append(action)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            done_batch.append(done)
            next_state_batch.append(next_state)
            reward_batch.append(reward)
        b = v(torch.Tensor(numpy.array(state_batch)).to(device))
        probs_batch = pi(torch.Tensor(numpy.array(state_batch)).to(device))
        target = (1 - torch.Tensor(numpy.array(done_batch)).to(device)) * gamma * v(
            torch.Tensor(numpy.array(next_state_batch)).to(device)) + torch.Tensor(reward_batch).to(device)
        loss_pi = -(target - b) * torch.log(probs_batch[action_batch])
        optimizer_pi.zero_grad()
        loss_pi.mean().backward(retain_graph=True)
        loss_v = torch.nn.functional.mse_loss(target, b.expand(2, 2))
        optimizer_v.zero_grad()
        loss_v.backward()
        optimizer_pi.step()
        optimizer_v.step()
        for i in range(n):
            done = done_batch[i]
            if done:
                env = env_batch[i]
                state, _ = env.reset()
                state_batch[i] = state
            else:
                next_state = next_state_batch[i]
                state = next_state
                state_batch[i] = state
env.close()
