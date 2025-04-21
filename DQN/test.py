import gymnasium
import os
import sys

sys.path.append(os.pardir)
import modules
import torch

device = torch.device("cpu" if not torch.cuda.is_available() else "cuda:0")
env = gymnasium.make("CartPole-v0", render_mode="human")
q = modules.Q(env.action_space.n, *env.observation_space.shape)
q.to(device)
q.eval()
q.load_state_dict(torch.load("DQN.pth", weights_only=True))
state, _ = env.reset()
total_reward = 0
while True:
    action = q(torch.Tensor(state).to(device).detach()).argmax().item()
    next_state, reward, terminated, _, _ = env.step(action)
    total_reward += reward
    if terminated:
        break
    state = next_state
env.close()
print(total_reward)
