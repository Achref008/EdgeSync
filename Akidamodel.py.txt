# model.py (for Akida with NODE_ID specific logic)

import torch
import torch.nn as nn
import numpy as np
from config import NODE_ID # Import NODE_ID

class MyModel(nn.Module):
    def __init__(self, input_channels=1):
        super(MyModel, self).__init__()
        self.input_channels = input_channels
        
        # Determine image size based on NODE_ID
        if NODE_ID == 7:
            image_size = 28  # MNIST images are 28x28
            self.adapter = nn.Sequential(
                nn.Conv2d(1, 4, kernel_size=1),  # Adapt grayscale to richer features
                nn.ReLU()
            )
            feature_extractor_input_channels = 4
        else:
            image_size = 32  # CIFAR10 images are 32x32
            self.adapter = nn.Identity()
            feature_extractor_input_channels = input_channels  # This will be 3 for CIFAR10
            
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(feature_extractor_input_channels, 16, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        
        # Correct calculation of flattened_size based on the full architecture flow
        # Use a dummy input on CPU first, then transfer to device if needed later
        with torch.no_grad():
            dummy_input_for_adapter = torch.zeros(1, self.input_channels, image_size, image_size)
            dummy_output_from_adapter = self.adapter(dummy_input_for_adapter)
            dummy_output_from_feature_extractor = self.feature_extractor(dummy_output_from_adapter)
            flattened_size = dummy_output_from_feature_extractor.view(1, -1).size(1)
             
        if NODE_ID == 7:
            self.classifier = nn.Sequential(
                nn.Linear(flattened_size, 32),
                nn.ReLU(),
                nn.Linear(32, 10)
            )
        else:
            self.classifier = nn.Sequential(
                nn.Linear(flattened_size, 128),
                nn.ReLU(),
                nn.Linear(128, 64),
                nn.ReLU(),
                nn.Linear(64, 10)
            )

    def forward(self, x):
        # Input channel adaptation (ensure input_channels is set correctly in main.py MyModel instantiation)
        if x.size(1) != self.input_channels:
            if self.input_channels == 1 and x.size(1) == 3:
                x = x.mean(dim=1, keepdim=True)  # RGB → grayscale
            elif self.input_channels == 3 and x.size(1) == 1:
                x = x.repeat(1, 3, 1, 1)  # Grayscale → RGB
            else:
                print(f"Warning: Input channel mismatch not handled: input {x.size(1)}, expected {self.input_channels}")
        
        x = self.adapter(x)  # Apply adapter first
        x = self.feature_extractor(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)

    def convert_weights(self, incoming_deltas_dict):
        """
        Adapts incoming DELTAS from other nodes to the current model's structure.
        Handles shape mismatches by projection.
        Returns a dictionary of adapted deltas, which should then be applied to the model.
        """
        adapted_deltas = {} # This will store the adapted deltas to be returned
        current_state = self.state_dict() # Get current model's state for shapes and device
        device = next(self.parameters()).device # Get model's current device

        converted, projected = 0, 0
        
        for name, current_param in current_state.items():
            # If a parameter in the current model is not found in the incoming deltas,
            # its delta from that peer is effectively zero.
            if name not in incoming_deltas_dict:
                adapted_deltas[name] = torch.zeros_like(current_param, device=device)
                continue  

            incoming_delta_param = incoming_deltas_dict[name].detach()
            target_shape = current_param.shape # The shape this node's model expects
            
            # Ensure incoming delta is on the same device as the current model's parameters
            incoming_delta_param = incoming_delta_param.to(device)

            print(f"[Akida][Convert] Processing: {name}")
            print(f"  Current model's target shape: {target_shape}")
            print(f"  Incoming delta's shape: {incoming_delta_param.shape}")

            if incoming_delta_param.shape == target_shape:
                adapted_deltas[name] = incoming_delta_param
                converted += 1
                print(f"[Akida][Convert] Direct shape match for {name}.")
            elif incoming_delta_param.dim() == current_param.dim():
                # For all shape mismatches where the number of dimensions is the same, use general projection.
                # Only project if both are floating point parameters
                if incoming_delta_param.dtype.is_floating_point and current_param.dtype.is_floating_point:
                    projected_tensor = MyModel._project_tensor_general(incoming_delta_param, target_shape)
                    if projected_tensor is not None:
                        adapted_deltas[name] = projected_tensor # Projected delta
                        projected += 1
                        print(f"[Akida][Convert] Projected {name}: incoming {incoming_delta_param.shape} -> target {target_shape}.")
                    else:
                        print(f"[Akida][Convert] Skipping invalid projected delta: {name} (returned None). Setting delta to zero.")
                        adapted_deltas[name] = torch.zeros_like(current_param, device=device) # Use zero delta if projection fails
                else:
                    print(f"[Akida][Convert] Skipping {name}: type mismatch for projection (one or both not float). Setting delta to zero.")
                    adapted_deltas[name] = torch.zeros_like(current_param, device=device)
            else:
                print(f"[Akida][Convert] Skipping {name}: dimension count mismatch. Setting delta to zero.")
                adapted_deltas[name] = torch.zeros_like(current_param, device=device)

        # Ensure all parameters from the current model are in adapted_deltas.
        # This handles cases where some parameters might be missing from incoming_deltas_dict
        # but were not caught by the initial `if name not in incoming_deltas_dict` because of the loop structure.
        # This loop also makes sure all returned deltas are on the correct device.
        final_adapted_deltas = {}
        for name, param in current_state.items():
            if name in adapted_deltas:
                final_adapted_deltas[name] = adapted_deltas[name].to(device)
            else:
                final_adapted_deltas[name] = torch.zeros_like(param, device=device)

        print(f"[Akida] Finished adaptation: {converted} direct delta copies, {projected} projected deltas.")
        
        # Return the dictionary of adapted DELTAS
        return final_adapted_deltas

    @staticmethod
    def _project_tensor_general(tensor, target_shape):
        """
        General projection logic to adapt a tensor to a target shape,
        handling different types of layers (Conv2d, Linear, etc.).
        Ensures all operations are done on the tensor's original device.
        """
        try:
            # Initialize projected_tensor on the same device and with the same dtype as the input tensor
            projected_tensor = torch.zeros(target_shape, device=tensor.device, dtype=tensor.dtype)

            # Handle convolution and linear layers (adapt the projections accordingly)
            if len(tensor.shape) == len(target_shape):
                if len(tensor.shape) == 4:  # Conv2d weights (out_channels, in_channels, H, W)
                    # Copy matching portions of the weights to the target shape
                    out_channels_match = min(tensor.shape[0], target_shape[0])
                    in_channels_match = min(tensor.shape[1], target_shape[1])
                    kernel_h_match = min(tensor.shape[2], target_shape[2])
                    kernel_w_match = min(tensor.shape[3], target_shape[3])
                    projected_tensor[:out_channels_match, :in_channels_match, :kernel_h_match, :kernel_w_match].copy_(
                        tensor[:out_channels_match, :in_channels_match, :kernel_h_match, :kernel_w_match]
                    )
                elif len(tensor.shape) == 2:  # Linear layer weights (out_features, in_features)
                    out_features_match = min(tensor.shape[0], target_shape[0])
                    in_features_match = min(tensor.shape[1], target_shape[1])
                    projected_tensor[:out_features_match, :in_features_match].copy_(
                        tensor[:out_features_match, :in_features_match]
                    )
                else:  # Generic case for 1D, 3D, or other matching dim counts (e.g., BatchNorm params, biases)
                    flat_tensor = tensor.view(-1)
                    target_size = np.prod(target_shape)
                    if flat_tensor.numel() > target_size:
                        projected_slice = flat_tensor[:target_size]
                    else:
                        pad = torch.zeros(target_size - flat_tensor.numel(), device=tensor.device, dtype=tensor.dtype)
                        projected_slice = torch.cat([flat_tensor, pad])
                    projected_tensor.copy_(projected_slice.view(target_shape)) # copy_ to the pre-allocated tensor
            else:  # If dimensions do not match, flatten and project
                flat_tensor = tensor.view(-1)
                target_size = np.prod(target_shape)
                if flat_tensor.numel() > target_size:
                    projected_slice = flat_tensor[:target_size]
                else:
                    pad = torch.zeros(target_size - flat_tensor.numel(), device=tensor.device, dtype=tensor.dtype)
                    projected_slice = torch.cat([flat_tensor, pad])
                projected_tensor.copy_(projected_slice.view(target_shape)) # copy_ to the pre-allocated tensor

            projected_tensor = torch.nan_to_num(projected_tensor, nan=0.0, posinf=1e5, neginf=-1e5)
            return projected_tensor
        except Exception as e:
            print(f"Error in _project_tensor_general for incoming tensor shape {tensor.shape} to target_shape {target_shape}: {e}")
            return None