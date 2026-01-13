import torch
import torch.nn as nn
import torch.optim as optim
from model import MyModel  # Ensure this class exists
from data_utils import get_data, get_validation_loader, get_mnist_loader_for_warmup
from config import NODE_ID, NEIGHBORS, DATASET_TYPE_MAP, IP_MAP, PORT_BASE, LEARNING_RATE, TAU1, ROUNDS
from comms import start_server, send_weights, receive_weights, send_failure_alert  # Ensure send_failure_alert is imported
import matplotlib.pyplot as plt
import numpy as np
import os
import socket
import datetime
import gspread
from oauth2client.service_account import ServiceAccountCredentials
import time
import collections # For defaultdict or similar, if needed, though dict is fine

BASE_PATH = "/home/sai/Desktop/achref/DFLtest1"
OPTIMIZER_NAME = "adam"  # or "sgd"
prev_loss, prev_acc = float("inf"), 0.0
torch.backends.cudnn.benchmark = True

# --- Enhanced Parameters (Adjusted for Figure Comparison) ---
INITIAL_CHANNEL_NOISE_VARIANCE = 0.0 # Set to 0.1 as per paper's Figure 6
NOISE_INCREASE_FACTOR_PER_ROUND = 0.0 # Set to 0.0 for constant noise for benchmark
SPARSITY_LEVEL = 1.0 # Set to 1.0 to disable sparsification for base DFLGM comparison


# --- Global variables for differential and momentum updates ---
previous_round_model_weights = None  # Stores model state from start of previous round
momentum_buffer = None              # Stores the accumulated momentum for updates
# IMPORTANT: Manually change this value for each run to generate data for different beta lines (0.1, 0.2, 0.3, 0.4, 0.5, 0.9)
MOMENTUM_BETA = 0.9                # Current beta for THIS run

# Determine device (CPU or GPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# Google Sheets Client Setup
def get_gsheet_client():
    scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
    creds = ServiceAccountCredentials.from_json_keyfile_name("gspread_key.json", scope)  # Ensure gspread_key.json is correct
    return gspread.authorize(creds)


# Modified Logging function for Google Sheets to include momentum_beta
def log_to_google_sheet(round_num, loss, accuracy, variance, optimizer_name, training_mode, connected_peers, momentum_beta, retries=3):
    try:
        client = get_gsheet_client()
        sheet = client.open("Nodes Data").sheet1  # Open first sheet
        
        # Ensure header exists (optional safety)
        if sheet.row_count == 0 or not sheet.get_all_values():
            header = [
                "Node_ID", "round_num", "optimizer_name", "loss", "accuracy",
                "variance", "timestamp", "connected peers", "training mode", "momentum_beta"
            ]
            sheet.append_row(header, value_input_option="USER_ENTERED")
            print("--> [GSheet] Wrote header row")

        # Append training data
        row = [
            NODE_ID,
            round_num,
            optimizer_name.upper(),
            round(loss, 5),
            round(accuracy, 2),
            float(variance),
            datetime.datetime.now().isoformat(),
            len(connected_peers),
            training_mode,
            round(momentum_beta, 2) # Include momentum_beta here
        ]
        sheet.append_row(row, value_input_option="USER_ENTERED")
        print(f"--> [GSheet] Logged Round {round_num} successfully.")
    except Exception as e:
        print(f"[ERROR] Google Sheet logging failed: {e}")
        if retries > 0:
            print(f"      Retrying in 10 seconds... ({retries} attempts left)")
            time.sleep(10)
            log_to_google_sheet(round_num, loss, accuracy, variance, optimizer_name, training_mode, connected_peers, momentum_beta, retries - 1)


# Check for active neighbors in the network
def check_active_neighbors():
    active_list = []
    for neighbor in NEIGHBORS.get(NODE_ID, []):
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.settimeout(1.0)  # Short timeout for connection check
                s.connect((IP_MAP[neighbor], PORT_BASE + neighbor))
            active_list.append(neighbor)
        except Exception as e:
            # Fixed TypeError for send_failure_alert
            send_failure_alert(NODE_ID, f"Node {NODE_ID} could not connect to {neighbor} ({IP_MAP[neighbor]}:{PORT_BASE + neighbor}): Connectivity Check Failed - {str(e)}")
            continue
    return active_list


# --- Advanced Noise Handling ---
def get_current_channel_noise_variance(round_num):
    """
    Returns the channel noise variance for the current round.
    This function can be expanded to implement more complex dynamic noise models.
    """
    # Simple example: Noise variance slightly increases over rounds
    current_variance = INITIAL_CHANNEL_NOISE_VARIANCE + (round_num * NOISE_INCREASE_FACTOR_PER_ROUND)
    return current_variance


def adjust_batch_size_based_on_noise(variance, base_batch_size=128, min_batch_size=32, max_batch_size=256):
    variance = min(1.0, variance)
    noise_factor = max(0, 1 - variance)
    new_batch_size = int(base_batch_size * noise_factor)
    return min(max(new_batch_size, min_batch_size), max_batch_size)


def pretrain_on_mnist(model, loss_fn, val_loader):
    """Pretraining phase for Node 7 using MNIST to warm-up the model."""
    print("[Node 7] Starting warm-up phase using MNIST dataset...")
    warmup_epochs = 30
    mnist_loader_warmup = get_mnist_loader_for_warmup()
    warmup_optimizer = optim.Adam(model.parameters(), lr=0.0005, weight_decay=1e-5)
    
    ckpt_path = os.path.join(BASE_PATH, "akida_pretrained.pt")
    
    if os.path.exists(ckpt_path):
        model.load_state_dict(torch.load(ckpt_path, map_location=device))  # Load to correct device
        print(f"[Node 7] Loaded pretrained model from {ckpt_path}")
    else:
        print(f"[Node 7] Pretraining on MNIST to warm up model...")
        model.train()  # Set model to training mode
        for epoch in range(warmup_epochs):
            correct, total, epoch_loss = 0, 0, 0.0
            for x, y in mnist_loader_warmup:
                x, y = x.to(device), y.to(device)  # Move data to device
                warmup_optimizer.zero_grad()
                output = model(x)
                loss = loss_fn(output, y)
                loss.backward()
                warmup_optimizer.step()

                _, pred = torch.max(output, 1)
                correct += (pred == y).sum().item()
                total += y.size(0)
                epoch_loss += loss.item()

            acc = 100.0 * correct / total
            print(f"[Node 7] Warm-up Epoch {epoch+1}/{warmup_epochs} -> Loss: {epoch_loss:.4f}, Acc: {acc:.2f}%")
        
        warmup_val_acc = evaluate(model, val_loader)
        print(f"[Node 7] Validation Accuracy after Warm-up: {warmup_val_acc:.2f}%")

        os.makedirs(BASE_PATH, exist_ok=True)
        torch.save(model.state_dict(), ckpt_path)
        print(f"[Node {NODE_ID}] Saved pretrained model to {ckpt_path}")


def get_model_weights(model):
    # Ensure weights are on CPU for serialization/transfer to avoid CUDA memory issues
    return {k: v.cpu().clone() for k, v in model.state_dict().items()}


def set_model_weights(model, weights):
    # Ensure weights are moved to the correct device (GPU/CPU) when loaded
    current_state_dict = model.state_dict()
    for k, v in weights.items():
        if k in current_state_dict:
            current_state_dict[k].copy_(v.to(device))  # Use copy_ for in-place update and device transfer
        else:
            print(f"Warning: Attempted to set weight for non-existent parameter: {k}")
    model.load_state_dict(current_state_dict)


def differential_update(local_weights, previous_weights):
    diff_weights = {}
    for name, param in local_weights.items():
        if name in previous_weights and param.dtype.is_floating_point:
            diff_weights[name] = param - previous_weights[name]
        else:
            diff_weights[name] = param.clone()  # Ensure it's a clone and on CPU
    return diff_weights

# --- Efficient Communication (Sparsification) ---
def sparsify_delta(delta: dict, sparsity_level: float) -> dict:
    """
    Applies top-k sparsification to the delta.
    Keeps only the top 'k' (defined by sparsity_level) absolute values for each parameter.
    """
    if sparsity_level >= 1.0: # No sparsification
        return delta
    if sparsity_level <= 0.0: # All zeros
        return {name: torch.zeros_like(param) for name, param in delta.items()}

    sparsified_delta = {}
    for name, param_tensor in delta.items():
        if param_tensor.dtype.is_floating_point:
            # Flatten the tensor to easily find top-k values
            flat_tensor = param_tensor.flatten()
            num_elements = flat_tensor.numel()
            k = int(num_elements * sparsity_level)

            if k == 0 and num_elements > 0: # Ensure at least one element is kept if possible
                k = 1
            if num_elements == 0: # Handle empty tensors
                sparsified_delta[name] = torch.zeros_like(param_tensor)
                continue
            if k >= num_elements: # If k is larger than or equal to num_elements, keep all
                sparsified_delta[name] = param_tensor.clone()
                continue

            # Get the top-k absolute values
            top_k_values, _ = torch.topk(flat_tensor.abs(), k)
            
            # Find the threshold: the smallest value among the top-k
            threshold = top_k_values[-1] if top_k_values.numel() > 0 else 0.0

            # Create a mask for elements greater than or equal to the threshold
            mask = flat_tensor.abs() >= threshold
            
            # Apply the mask: elements below threshold become zero
            sparsified_flat_tensor = torch.zeros_like(flat_tensor)
            sparsified_flat_tensor[mask] = flat_tensor[mask]
            
            sparsified_delta[name] = sparsified_flat_tensor.reshape(param_tensor.shape)
        else:
            # Non-floating point parameters are not sparsified
            sparsified_delta[name] = param_tensor.clone()
    return sparsified_delta


def metropolis_average(local_delta, received_deltas_with_nid, neighbors):
    """
    Aggregates received deltas (and local delta) using Metropolis-Hastings averaging.
    Args:
        local_delta (dict): The delta of the current node (on CPU). This should be the NOISY and potentially SPARSIFIED local delta.
        received_deltas_with_nid (list): List of (delta, node_id) tuples from neighbors (on CPU). These should be noisy and potentially sparsified deltas.
        neighbors (dict): Dictionary mapping node IDs to their neighbor lists.
    Returns:
        dict: Aggregated delta (on CPU).
    """
    # Combine local delta with received deltas for averaging
    all_deltas_data = [(local_delta, NODE_ID)] + received_deltas_with_nid
    
    # Calculate degrees for Metropolis-Hastings weights
    degrees = {nid: len(neighbors.get(nid, [])) for _, nid in all_deltas_data}
    d_i = degrees.get(NODE_ID, 0) # Degree of current node

    mh_weights = {}
    total_weight = 0.0
    for _, nid in all_deltas_data:
        d_j = degrees.get(nid, 0)
        # Metropolis-Hastings weight formula: wij = 1 / (1 + max(deg_i, deg_j))
        w = 1.0 / (1.0 + max(d_i, d_j)) 
        mh_weights[nid] = w
        total_weight += w
    
    # Normalize weights
    for nid in mh_weights:
        if total_weight > 0: # Avoid division by zero
            mh_weights[nid] /= total_weight
        else: # Fallback: if no peers and only self-data, assign 1.0 to self, 0.0 otherwise
            mh_weights[nid] = 1.0 if nid == NODE_ID else 0.0

    # Initialize aggregated delta with zeros, on CPU as input deltas are on CPU
    if local_delta:
        # Use first tensor's device from local_delta, which should be CPU from get_model_weights
        sample_tensor_device = next(iter(local_delta.values())).device
    else:
        # If local_delta is empty (e.g., first round for a very simple model without params), default to CPU
        sample_tensor_device = torch.device('cpu') 

    aggregated_delta = {k: torch.zeros_like(v, device=sample_tensor_device) for k, v in local_delta.items()}
    
    # Perform weighted average
    for name in aggregated_delta:
        if aggregated_delta[name].dtype.is_floating_point: # Only average float parameters
            for delta_dict, nid in all_deltas_data:
                if name in delta_dict:
                    # Ensure delta_dict[name] is on the same device as aggregated_delta[name] (CPU)
                    aggregated_delta[name] += delta_dict[name].to(aggregated_delta[name].device) * mh_weights[nid]
        else:
            # For non-floating point parameters (e.g., BatchNorm non-trainable params),
            # keep the local node's delta (its value from differential_update).
            # Ensure it's on the correct device.
            if name in local_delta:
                aggregated_delta[name] = local_delta[name].clone().to(aggregated_delta[name].device)
            # If a parameter is not in local_delta (e.g., if differential_update skipped it),
            # it implies zero delta from this node. Keep zeros_like.
            

    print(f"[Node {NODE_ID}] Aggregated deltas using Metropolis weights: { {k: f'{v:.3f}' for k,v in mh_weights.items()} }")
    return aggregated_delta


def apply_momentum_and_update_model(aggregated_delta, model):
    """
    Applies the aggregated delta (with momentum) to the model's parameters.
    """
    global momentum_buffer, MOMENTUM_BETA

    # Initialize momentum_buffer if it's the first time or if model parameters change
    if momentum_buffer is None:
        momentum_buffer = {name: torch.zeros_like(param, device=device) for name, param in model.state_dict().items()}

    with torch.no_grad():
        for name, delta_tensor_cpu in aggregated_delta.items(): # delta_tensor_cpu is on CPU
            # Ensure delta_tensor is on the same device as the model before processing
            delta_tensor = delta_tensor_cpu.to(device)

            if name in model.state_dict() and model.state_dict()[name].dtype.is_floating_point:
                # Ensure momentum_buffer tensor for 'name' is initialized and on the correct device
                if name not in momentum_buffer or momentum_buffer[name].shape != delta_tensor.shape or momentum_buffer[name].device != delta_tensor.device:
                    momentum_buffer[name] = torch.zeros_like(delta_tensor, device=device)
                
                # Apply momentum to the delta
                momentum_buffer[name].mul_(MOMENTUM_BETA).add_(delta_tensor, alpha=(1 - MOMENTUM_BETA))
                
                # Update model weights with the momentum-adjusted delta
                model.state_dict()[name].add_(momentum_buffer[name], alpha=LEARNING_RATE)
            elif name in model.state_dict(): # For non-floating point parameters, apply directly if they exist
                model.state_dict()[name].copy_(delta_tensor)


def compute_variance(weight_snapshots):
    """
    Computes the variance of the L2 norm of the difference between consecutive 
    full model weight snapshots. Used for plotting overall model change.
    """
    if len(weight_snapshots) < 2:
        return 0.0
    # Compute the L2 norm of the difference between consecutive weight snapshots
    deltas_norm = [np.linalg.norm(weight_snapshots[i] - weight_snapshots[i-1]) for i in range(1, len(weight_snapshots))]
    return np.var(deltas_norm) if deltas_norm else 0.0

def save_model_checkpoint(model):
    os.makedirs(BASE_PATH, exist_ok=True)
    path = os.path.join(BASE_PATH, f"model_node{NODE_ID}_final.pt")
    # Save model state dictionary. get_model_weights ensures it's on CPU.
    torch.save(model.state_dict(), path)
    print(f"[Node {NODE_ID}] Model checkpoint saved at {path}")

def save_log_and_plot(loss_log, acc_log):
    os.makedirs(BASE_PATH, exist_ok=True)
    log_path = os.path.join(BASE_PATH, f"node{NODE_ID}_metrics_log.txt")
    plot_path = os.path.join(BASE_PATH, f"training_plot_node{NODE_ID}.png")

    with open(log_path, "w") as f:
        f.write("Round,Loss,Accuracy\n")
        for i, (loss, acc) in enumerate(zip(loss_log, acc_log)):
            f.write(f"{i+1},{loss:.4f},{acc:.2f}\n")

    rounds = list(range(1, len(loss_log) + 1))
    fig, ax1 = plt.subplots()
    ax1.set_title(f"Node {NODE_ID} Training Metrics")
    ax1.set_xlabel("Round")
    ax1.set_ylabel("Loss", color='tab:blue')
    ax1.plot(rounds, loss_log, marker='o', color='tab:blue')
    ax1.tick_params(axis='y', labelcolor='tab:blue')

    ax2 = ax1.twinx()
    ax2.set_ylabel("Accuracy (%)", color='tab:green')
    ax2.plot(rounds, acc_log, marker='x', color='tab:green')
    ax2.tick_params(axis='y', labelcolor='tab:green')

    fig.tight_layout()
    plt.grid(True)
    plt.savefig(plot_path)
    print(f"[Node {NODE_ID}] Saved plot to {plot_path}")
    plt.close()


def compute_and_plot_weight_variance(weight_snapshots):
    os.makedirs(BASE_PATH, exist_ok=True)
    path = os.path.join(BASE_PATH, f"node{NODE_ID}_Param_variance.png")
    # Calculate the L2 norm of the difference between consecutive weight snapshots
    deltas = [np.linalg.norm(weight_snapshots[i] - weight_snapshots[i - 1]) for i in range(1, len(weight_snapshots))]
    if not deltas:
        return
    plt.figure()
    plt.plot(range(2, len(weight_snapshots) + 1), deltas, marker='s', color='red')
    plt.yscale("log")
    plt.xlabel("Round")
    plt.ylabel("Δ Weight Norm (log scale)") # Changed label for clarity
    plt.title(f"Node {NODE_ID} Parameter Update Norm Over Rounds")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(path)
    print(f"[Node {NODE_ID}] Saved variance Parameter plot to {path}")
    plt.close()


# Main Training Loop
def train():
    global prev_loss, prev_acc, previous_round_model_weights, momentum_buffer

    start_server()  # Ensure server start logic is implemented in comms.py

    # Model and Data Loader Initialization
    dataset_type = DATASET_TYPE_MAP.get(NODE_ID, "CIFAR10")  # Default to CIFAR10 if not found in the map
    if dataset_type == "MNIST":
        input_channels = 1  # Grayscale images
    elif dataset_type == "CIFAR10":
        input_channels = 3  # RGB images
    else:
        input_channels = 3  # Default to RGB if an unknown dataset is used

    model = MyModel(input_channels=input_channels).to(device) # Move model to device immediately
    
    # Initialize optimizer and loss function *before* pretraining
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-5) if OPTIMIZER_NAME == "adam" \
        else optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=0.9, weight_decay=1e-4, nesterov=True)
    loss_fn = nn.CrossEntropyLoss() # Fixed typo here
    
    val_loader = get_validation_loader(peer=True)  # Ensure get_validation_loader() is properly implemented

    if NODE_ID == 7:
        pretrain_on_mnist(model, loss_fn, val_loader) # Pass loss_fn and val_loader
        data_loader = iter(get_data(batch_size=64))  # Node 7 gets its own data loader with smaller batch size
    else:
        data_loader = iter(get_data(batch_size=128))  # Use the default batch size for CIFAR-10

    loss_log, acc_log = [], []
    
    # Initialize previous_round_model_weights at the very beginning
    # This represents the initial "global" model for the first round's delta calculation
    previous_round_model_weights = get_model_weights(model) 
    
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=ROUNDS)

    # Initial snapshot before any training
    weight_snapshots = [torch.cat([p.flatten().cpu() for p in model.parameters()]).detach().numpy()]

    for r in range(ROUNDS):
        print(f"\n[Node {NODE_ID}] --- Round {r+1}/{ROUNDS} ---")
        
        # Capture model state *before* local training for this round's differential update
        # This is the 'global' model state received/aggregated from the previous round
        start_round_model_weights = get_model_weights(model) 

        model.train()
        round_loss, correct, total = 0.0, 0, 0

        # Training Loop for TAU1 local epochs
        for _ in range(TAU1):
            try:
                x, y = next(data_loader)
            except StopIteration:
                data_loader = iter(get_data()) # Reinitialize data loader for next epoch
                x, y = next(data_loader)
            
            x, y = x.to(device), y.to(device) # Move data to device

            optimizer.zero_grad()
            output = model(x)
            loss = loss_fn(output, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

            _, pred = torch.max(output, 1)
            correct += (pred == y).sum().item()
            total += y.size(0)
            round_loss += loss.item()

        avg_loss = round_loss / TAU1
        accuracy = 100.0 * correct / total if total > 0 else 0.0
        
        # **Performance check**: Compare current accuracy and loss with previous ones
        performance_improved = (avg_loss < prev_loss) or (accuracy > prev_acc)

        if performance_improved:
            print(f"[Node {NODE_ID}] Performance improved. Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%")
        else:
            print(f"[Node {NODE_ID}] No improvement in performance. Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%")
        
        # **Validation accuracy**: Print validation accuracy after each round
        val_acc = evaluate(model, val_loader)
        print(f"[Node {NODE_ID}] Validation Accuracy: {val_acc:.2f}%")

        # --- Communication Phase for DELTAS ---
        active_list = check_active_neighbors()
        print(f"[Node {NODE_ID}] Active neighbors: {len(active_list)}/{len(NEIGHBORS.get(NODE_ID, []))} -> {active_list}")
        
        # Calculate local delta: current_local_model - start_of_round_global_model
        current_local_weights = get_model_weights(model) # Get current model state (on CPU)
        local_delta = differential_update(current_local_weights, start_round_model_weights)
        
        # --- Advanced Noise Handling: Apply dynamic channel noise ---
        current_channel_noise_std = get_current_channel_noise_variance(r + 1)**0.5 # std_dev = sqrt(variance)
        noisy_local_delta = {}
        for name, param_tensor in local_delta.items():
            if param_tensor.dtype.is_floating_point:
               noise = torch.randn_like(param_tensor) * current_channel_noise_std
               noisy_local_delta[name] = param_tensor + noise
            else:
               noisy_local_delta[name] = param_tensor.clone() # Non-floating point params don't get noise
        
        # --- Efficient Communication: Sparsify the noisy delta ---
        sparsified_noisy_local_delta = sparsify_delta(noisy_local_delta, SPARSITY_LEVEL)

        # --- Send only the sparsified and noisy delta ---
        if performance_improved and active_list:
            print(f"[Node {NODE_ID}] Performance improved. Sending sparsified and noisy local delta to peers.")
            # Note: For this specific figure comparison, we are NOT sending reliability score explicitly here
            # as the base DFLGM doesn't inherently use it for the momentum beta comparison plots.
            send_weights(sparsified_noisy_local_delta, target_nodes=active_list)
        elif not active_list:
            print(f"[Node {NODE_ID}] No connected peers. Skipping delta sharing.")
        else:
            print(f"[Node {NODE_ID}] No performance improvement. Not sending local delta.")
        
        did_aggregate = False
        variance = 0.0 # Initialize variance for logging

        # Receive deltas from peers (these are the raw deltas collected by the server thread)
        # Note: For this specific figure comparison, we are NOT expecting reliability score implicitly here.
        # Ensure comms.py's receive_weights returns (delta_dict, nid) not (delta_dict, nid, reliability_score) if you only need base DFLGM comparison.
        # If your comms.py has been updated for reliability, it will return the 3-tuple, but the metropolis_average should be adjusted.
        
        # Assuming comms.py is the *original* one you provided where receive_weights returns (delta_dict, nid)
        received_deltas_from_peers = receive_weights(min_expected=0, wait_time=10) 

        if received_deltas_from_peers:
            print(f"[Node {NODE_ID}] Received deltas from: {[nid for _, nid in received_deltas_from_peers]}")
            print(f"[Node {NODE_ID}] Processed {len(received_deltas_from_peers)} incoming deltas.")

            # Prepare deltas for aggregation: convert if needed, and prepare for variance calc
            valid_received_deltas_with_nid = []
            diffs_for_variance = [] # To calculate variance of received deltas relative to local delta
            
            for peer_delta_dict, nid in received_deltas_from_peers:
                try:
                    # model.convert_weights is used here to adapt the structure of incoming deltas
                    # It ensures deltas have the correct shapes for this node's model.
                    adapted_peer_delta = model.convert_weights(peer_delta_dict) 
                    
                    if adapted_peer_delta is not None:
                        # Ensure both sparsified_noisy_local_delta and adapted_peer_delta are flattened for variance calc
                        # and that they only include floating-point parameters
                        local_processed_delta_flat = torch.cat([v.flatten() for k,v in sparsified_noisy_local_delta.items() if v.dtype.is_floating_point])
                        adapted_peer_delta_flat = torch.cat([v.flatten() for k,v in adapted_peer_delta.items() if v.dtype.is_floating_point])
                        
                        # Pad smaller tensor if sizes differ for comparison
                        max_len = max(local_processed_delta_flat.numel(), adapted_peer_delta_flat.numel())
                        
                        if max_len > 0:
                            if local_processed_delta_flat.numel() < max_len:
                                local_processed_delta_flat = torch.cat([local_processed_delta_flat, torch.zeros(max_len - local_processed_delta_flat.numel(), device=local_processed_delta_flat.device)])
                            if adapted_peer_delta_flat.numel() < max_len:
                                adapted_peer_delta_flat = torch.cat([adapted_peer_delta_flat, torch.zeros(max_len - adapted_peer_delta_flat.numel(), device=adapted_peer_delta_flat.device)])

                            diff_for_variance = local_processed_delta_flat - adapted_peer_delta_flat
                            diffs_for_variance.append(diff_for_variance)
                            valid_received_deltas_with_nid.append((adapted_peer_delta, nid))
                        else:
                            print(f"[Node {NODE_ID}] Skipping peer {nid} delta for variance due to empty/incompatible parameters.")
                    else: 
                        print(f"[Node {NODE_ID}] Skipping invalid adapted delta from neighbors {nid}.") 
                        
                except Exception as e:
                    print(f"[Node {NODE_ID}] Error processing delta from node {nid}: {e}")
            
            if valid_received_deltas_with_nid:
                # Calculate variance of received deltas relative to local delta
                if diffs_for_variance:
                    diffs_tensor = torch.stack(diffs_for_variance)
                    variance = torch.mean(torch.sum(diffs_tensor ** 2, dim=1)).item()
                else:
                    variance = 0.0 # No valid diffs to calculate variance from
                
                # Aggregate deltas using Metropolis averaging (now using sparsified_noisy_local_delta)
                # Note: `metropolis_average` is designed to handle this as it treats local_delta and received_deltas similarly.
                aggregated_delta = metropolis_average(sparsified_noisy_local_delta, valid_received_deltas_with_nid, NEIGHBORS)
                
                # Apply momentum and update the model using the aggregated delta
                apply_momentum_and_update_model(aggregated_delta, model)
                did_aggregate = True
            else:
                print(f"[Node {NODE_ID}] No valid deltas received for aggregation. Model not updated via aggregation.")
        else:
            print(f"[Node {NODE_ID}] No deltas received. Continuing solo :( .")
        
        # After aggregation (or solo training), the current model weights (now updated)
        # become the 'previous' for the next round's delta calculation.
        previous_round_model_weights = get_model_weights(model) 

        # Log current full model snapshot for overall variance tracking
        weight_snapshots.append(torch.cat([p.flatten().cpu() for p in model.parameters()]).detach().numpy())

        # Logging to Google Sheet including MOMENTUM_BETA
        log_to_google_sheet(
            round_num=r + 1,
            loss=avg_loss,
            accuracy=val_acc,
            variance=variance,
            optimizer_name=OPTIMIZER_NAME,
            training_mode="collaborative" if did_aggregate else "solo",
            connected_peers=active_list,
            momentum_beta=MOMENTUM_BETA # Pass the current momentum beta
        )

        loss_log.append(avg_loss)
        acc_log.append(val_acc)
        prev_loss, prev_acc = avg_loss, accuracy
        scheduler.step()

    save_log_and_plot(loss_log, acc_log)
    compute_and_plot_weight_variance(weight_snapshots) # This is variance of full model over rounds
    save_model_checkpoint(model)
    print(f"[Node {NODE_ID}] Training completed.")


def evaluate(model, loader):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device) # Move data to model's device
            out = model(x)
            _, pred = torch.max(out, 1)
            correct += (pred == y).sum().item()
            total += y.size(0)
    return 100.0 * correct / total if total > 0 else 0.0


if __name__ == "__main__":
    try:
        train()
    except KeyboardInterrupt:
        print(f"\n[Node {NODE_ID}] Training interrupted. Marking node as killed...")
        try:
            client = get_gsheet_client()
            sheet = client.open("Nodes Data").sheet1
            row = [
                NODE_ID,
                "KILLED",
                OPTIMIZER_NAME.upper(),
                "-", "-", "-",  # No loss, accuracy, variance
                datetime.datetime.now().isoformat(),
                "-", "interrupted",
                MOMENTUM_BETA # Log momentum beta even if killed
            ]
            sheet.append_row(row, value_input_option="USER_ENTERED")
            print("--> [GSheet] Logged KILLED node.")
        except Exception as e:
            print(f"[ERROR] Failed to log killed node: {e}")