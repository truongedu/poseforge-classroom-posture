"""Minimal MLP architectures for the PoseForge 52-feature posture classifier."""

from torch import nn

MODEL_REGISTRY = {
    "mlp_small": (64, 32),
    "mlp_base": (128, 64),
    "mlp_wide": (256, 128),
}


class PostureMLP(nn.Module):
    def __init__(self, model_name="mlp_base", dropout=0.15, input_dim=52, num_classes=6):
        super().__init__()

        if model_name not in MODEL_REGISTRY:
            raise ValueError(f"Unknown model: {model_name}; choose {list(MODEL_REGISTRY)}")

        h1, h2 = MODEL_REGISTRY[model_name]
        self.net = nn.Sequential(
            nn.Linear(input_dim, h1), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(h1, h2), nn.ReLU(),
            nn.Linear(h2, num_classes),
        )

    def forward(self, x):
        return self.net(x)
