# model.py — simple & stable (your old Jetson-style net)

import torch
import torch.nn as nn
import numpy as np

class MyModel(nn.Module):
    def __init__(self, input_channels=3, input_size=32):
        super().__init__()
        self.input_channels = input_channels  # keep for clarity

        self.feature_extractor = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

        # dynamic flatten size (works for 32x32 CIFAR / 28x28 MNIST paths)
        with torch.no_grad():
            dummy = torch.zeros(1, input_channels, input_size, input_size)
            out = self.feature_extractor(dummy)
            flattened_size = out.view(1, -1).size(1)

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flattened_size, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 10),
        )

    def forward(self, x):
        x = self.feature_extractor(x)
        return self.classifier(x)

    # --- keep your convert_weights projection as-is (useful in your DFL setup) ---
    def convert_weights(self, incoming_deltas_dict):
        adapted_deltas = {}
        current_state = self.state_dict()
        device = next(self.parameters()).device
        for name, current_param in current_state.items():
            if name not in incoming_deltas_dict:
                adapted_deltas[name] = torch.zeros_like(current_param, device=device)
                continue
            incoming = incoming_deltas_dict[name].detach().to(device)
            target_shape = current_param.shape
            if incoming.shape == target_shape:
                adapted_deltas[name] = incoming
            elif len(incoming.shape) == len(target_shape) and incoming.dtype.is_floating_point and current_param.dtype.is_floating_point:
                projected = self._project_tensor(incoming, target_shape)
                if projected is None:
                    adapted_deltas[name] = torch.zeros_like(current_param, device=device)
                else:
                    adapted_deltas[name] = projected.to(device)
            else:
                adapted_deltas[name] = torch.zeros_like(current_param, device=device)
        final_adapted = {}
        for name, param in current_state.items():
            final_adapted[name] = adapted_deltas.get(name, torch.zeros_like(param, device=device)).to(device)
        return final_adapted

    @staticmethod
    def _project_tensor(tensor, target_shape):
        try:
            flat = tensor.view(-1)
            target_size = int(np.prod(target_shape))
            if flat.numel() >= target_size:
                proj = torch.randn(target_size, flat.numel(), device=flat.device) * (1.0 / np.sqrt(target_size))
                out = proj @ flat
                out = torch.nan_to_num(out, nan=0.0, posinf=1e5, neginf=-1e5)
            else:
                pad = torch.zeros(target_size - flat.numel(), device=tensor.device, dtype=tensor.dtype)
                out = torch.cat([flat, pad])
            if torch.isnan(out).any() or torch.isinf(out).any():
                return None
            return out.view(target_shape)
        except Exception:
            return None