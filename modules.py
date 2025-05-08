import torch


class DeterministicPolicy(torch.nn.Module):
    def __init__(self, action_size, state_size):
        super().__init__()
        self.pi = torch.nn.Sequential(
            torch.nn.Linear(state_size, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, action_size),
        )

    def forward(self, x):
        pi = self.pi(x)
        return pi


class DuelingQ(torch.nn.Module):
    def __init__(self, action_size, state_size):
        super().__init__()
        self.a = torch.nn.Sequential(
            torch.nn.Linear(state_size, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, action_size),
        )
        self.v = torch.nn.Sequential(
            torch.nn.Linear(state_size, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 1),
        )

    def forward(self, x):
        a = self.a(x)
        v = self.v(x)
        if 1 == a.ndim:
            return a + v - a.mean()
        return a + v - a.mean(1, True)


class Policy(torch.nn.Module):
    def __init__(self, action_size, state_size):
        super().__init__()
        self.pi = torch.nn.Sequential(
            torch.nn.Linear(state_size, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, action_size),
            torch.nn.Softmax(0),
        )

    def forward(self, x):
        pi = self.pi(x)
        return pi


class Q(torch.nn.Module):
    def __init__(self, action_size, state_size):
        super().__init__()
        self.q = torch.nn.Sequential(
            torch.nn.Linear(state_size, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, action_size),
        )

    def forward(self, x):
        q = self.q(x)
        return q


class V(torch.nn.Module):
    def __init__(self, state_size):
        super().__init__()
        self.v = torch.nn.Sequential(
            torch.nn.Linear(state_size, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 1),
        )

    def forward(self, x):
        v = self.v(x)
        return v
