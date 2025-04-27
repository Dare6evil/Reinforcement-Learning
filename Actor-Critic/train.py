from matplotlib import pyplot
import os
import sys

sys.path.append(os.pardir)
import gymnasium
import modules
import numpy
import torch

device = torch.device("cpu" if not torch.cuda.is_available() else "cuda:0")
env = gymnasium.make("CartPole-v1", render_mode="human")
episodes = 1000
gamma = 0.98
lr_pi = 0.0002
lr_v = 0.0005
# max_total_reward = 0
reward_history = [0] * episodes
runs = 5
for run in range(1, 1 + runs):
    pi = modules.Policy(env.action_space.n, *env.observation_space.shape)
    optimizer_pi = torch.optim.Adam(pi.parameters(), lr_pi)
    pi.to(device)
    state, _ = env.reset()
    v = modules.V(*env.observation_space.shape)
    optimizer_v = torch.optim.Adam(v.parameters(), lr_v)
    v.to(device)
    for episode in range(episodes):
        total_reward = 0
        while True:
            b = v(torch.Tensor(state).to(device))
            probs = pi(torch.Tensor(state).to(device))
            action = numpy.random.choice(len(probs), p=probs.detach().cpu().numpy())
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            target = (1 - done) * gamma * v(
                torch.Tensor(next_state).to(device)
            ).detach() + reward
            loss_pi = -(target - b) * torch.log(probs[action])
            optimizer_pi.zero_grad()
            loss_pi.backward(retain_graph=True)
            loss_v = torch.nn.functional.mse_loss(target, b)
            optimizer_v.zero_grad()
            loss_v.backward()
            optimizer_pi.step()
            optimizer_v.step()
            total_reward += reward
            if done:
                state, _ = env.reset()
                break
            state = next_state
        # if max_total_reward < total_reward:
        #     max_total_reward = total_reward
        #     torch.save(pi.state_dict(), f"Actor-Critic.pth")
        # elif max_total_reward == total_reward:
        #     torch.save(pi.state_dict(), f"Actor-Critic.pth")
        reward_history[episode] += (total_reward - reward_history[episode]) / run
env.close()
pyplot.plot(reward_history)
pyplot.xlabel("Episode")
pyplot.ylabel("Total Reward")
pyplot.show()
