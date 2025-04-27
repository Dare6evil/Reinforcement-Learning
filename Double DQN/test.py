import os
import sys

sys.path.append(os.pardir)
import gymnasium
import modules
import torch

device = torch.device("cpu" if not torch.cuda.is_available() else "cuda:0")
env = gymnasium.make("CartPole-v1", render_mode="human")
q = modules.Q(env.action_space.n, *env.observation_space.shape)
q.eval()
q.load_state_dict(torch.load("Double DQN.pth", weights_only=True))
q.to(device)
state, _ = env.reset()
total_reward = 0
while True:
    action = q(torch.Tensor(state).to(device)).detach().argmax().item()
    next_state, reward, terminated, _, _ = env.step(action)
    total_reward += reward
    if terminated:
        break
    state = next_state
env.close()
print(total_reward)
