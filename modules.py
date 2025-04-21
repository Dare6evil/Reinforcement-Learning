import torch


class DuelingQ(torch.nn.Module):
    def __init__(self, action_size, state_size):
        super().__init__()
        self.a = torch.nn.Sequential(
            torch.nn.Linear(state_size, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, action_size)
        )
        self.v = torch.nn.Sequential(
            torch.nn.Linear(state_size, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 1)
        )

    def forward(self, x):
        a = self.a(x)
        v = self.v(x)
        if 1 != a.ndim:
            return a + v - a.mean(1, True)
        return a + v - a.mean()


class Policy(torch.nn.Module):
    def __init__(self, action_size, state_size):
        super().__init__()
        self.layer1 = torch.nn.Linear(state_size, 128)
        self.layer2 = torch.nn.Linear(128, action_size)

    def forward(self, x):
        x = torch.nn.functional.relu(self.layer1(x))
        return torch.nn.functional.softmax(self.layer2(x), dim=0)


class Q(torch.nn.Module):
    def __init__(self, action_size, state_size):
        super().__init__()
        self.layer1 = torch.nn.Linear(state_size, 128)
        self.layer2 = torch.nn.Linear(128, 128)
        self.layer3 = torch.nn.Linear(128, action_size)

    def forward(self, x):
        x = torch.nn.functional.relu(self.layer1(x))
        x = torch.nn.functional.relu(self.layer2(x))
        return self.layer3(x)


class V(torch.nn.Module):
    def __init__(self, state_size):
        super().__init__()
        self.layer1 = torch.nn.Linear(state_size, 128)
        self.layer2 = torch.nn.Linear(128, 1)

    def forward(self, x):
        x = torch.nn.functional.relu(self.layer1(x))
        return self.layer2(x)
