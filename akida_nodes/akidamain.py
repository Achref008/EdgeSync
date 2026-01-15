import torch
import torch.nn as nn
import torch.optim as optim
from model import MyModel
from data_utils import get_data, get_validation_loader, get_mnist_loader_for_warmup
from config import NODE_ID, NEIGHBORS, DATASET_TYPE_MAP, IP_MAP, PORT_BASE, LEARNING_RATE, TAU1, ROUNDS
from comms import start_server, send_weights, receive_weights, send_failure_alert
import matplotlib.pyplot as plt
import numpy as np
import os
import socket
import datetime
import gspread
from oauth2client.service_account import ServiceAccountCredentials
import time
import collections

# -------------------------------------------------------------------------
# Runtime configuration and experiment constants
# -------------------------------------------------------------------------
BASE_PATH = "/home/sai/Desktop/achref/DFLtest1"
OPTIMIZER_NAME = "adam"  # "adam" or "sgd"
prev_loss, prev_acc = float("inf"), 0.0
torch.backends.cudnn.benchmark = True

# -------------------------------------------------------------------------
# Communication-channel modeling and delta transmission controls
# -------------------------------------------------------------------------
INITIAL_CHANNEL_NOISE_VARIANCE = 0.0
NOISE_INCREASE_FACTOR_PER_ROUND = 0.0
SPARSITY_LEVEL = 1.0

# -------------------------------------------------------------------------
# State tracking for delta-based training and momentum-based stabilization
# -------------------------------------------------------------------------
previous_round_model_weights = None
momentum_buffer = None

# Set this value per experiment run when generating beta-comparison plots
MOMENTUM_BETA = 0.9

# -------------------------------------------------------------------------
# Device selection
# -------------------------------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# -------------------------------------------------------------------------
# Google Sheets client: used for run logging across nodes
# -------------------------------------------------------------------------
def get_gsheet_client():
    scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
    creds = ServiceAccountCredentials.from_json_keyfile_name("gspread_key.json", scope)
    return gspread.authorize(creds)


# -------------------------------------------------------------------------
# Google Sheets logging: one row per round including beta and connectivity
# -------------------------------------------------------------------------
def log_to_google_sheet(round_num, loss, accuracy, variance, optimizer_name,
                        training_mode, connected_peers, momentum_beta, retries=3):
    try:
        client = get_gsheet_client()
        sheet = client.open("Nodes Data").sheet1

        # Optional safety: ensure a header exists before appending run data
        if sheet.row_count == 0 or not sheet.get_all_values():
            header = [
                "Node_ID", "round_num", "optimizer_name", "loss", "accuracy",
                "variance", "timestamp", "connected peers", "training mode", "momentum_beta"
            ]
            sheet.append_row(header, value_input_option="USER_ENTERED")
            print("--> [GSheet] Wrote header row")

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
            round(momentum_beta, 2)
        ]
        sheet.append_row(row, value_input_option="USER_ENTERED")
        print(f"--> [GSheet] Logged Round {round_num} successfully.")

    except Exception as e:
        print(f"[ERROR] Google Sheet logging failed: {e}")
        if retries > 0:
            print(f"      Retrying in 10 seconds... ({retries} attempts left)")
            time.sleep(10)
            log_to_google_sheet(round_num, loss, accuracy, variance, optimizer_name,
                                training_mode, connected_peers, momentum_beta, retries - 1)


# -------------------------------------------------------------------------
# Network connectivity probe: checks which neighbors are reachable this round
# -------------------------------------------------------------------------
def check_active_neighbors():
    active_list = []
    for neighbor in NEIGHBORS.get(NODE_ID, []):
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.settimeout(1.0)
                s.connect((IP_MAP[neighbor], PORT_BASE + neighbor))
            active_list.append(neighbor)
        except Exception as e:
            send_failure_alert(
                NODE_ID,
                f"Node {NODE_ID} could not connect to {neighbor} "
                f"({IP_MAP[neighbor]}:{PORT_BASE + neighbor}): Connectivity Check Failed - {str(e)}"
            )
            continue
    return active_list


# -------------------------------------------------------------------------
# Channel noise schedule: returns variance used to perturb transmitted deltas
# -------------------------------------------------------------------------
def get_current_channel_noise_variance(round_num):
    """
    Defines how channel noise changes over time.
    With NOISE_INCREASE_FACTOR_PER_ROUND = 0, noise is constant across rounds.
    """
    return INITIAL_CHANNEL_NOISE_VARIANCE + (round_num * NOISE_INCREASE_FACTOR_PER_ROUND)


def adjust_batch_size_based_on_noise(variance, base_batch_size=128, min_batch_size=32, max_batch_size=256):
    """
    Optional utility to reduce batch size under high-noise conditions.
    Not used directly in the current training loop unless called explicitly.
    """
    variance = min(1.0, variance)
    noise_factor = max(0, 1 - variance)
    new_batch_size = int(base_batch_size * noise_factor)
    return min(max(new_batch_size, min_batch_size), max_batch_size)


# -------------------------------------------------------------------------
# Warm-up routine: Node 7 can pretrain on MNIST to stabilize initialization
# -------------------------------------------------------------------------
def pretrain_on_mnist(model, loss_fn, val_loader):
    print("[Node 7] Starting warm-up phase using MNIST dataset...")
    warmup_epochs = 30
    mnist_loader_warmup = get_mnist_loader_for_warmup()
    warmup_optimizer = optim.Adam(model.parameters(), lr=0.0005, weight_decay=1e-5)

    ckpt_path = os.path.join(BASE_PATH, "akida_pretrained.pt")

    if os.path.exists(ckpt_path):
        model.load_state_dict(torch.load(ckpt_path, map_location=device))
        print(f"[Node 7] Loaded pretrained model from {ckpt_path}")
    else:
        print("[Node 7] Pretraining on MNIST to warm up model...")
        model.train()
        for epoch in range(warmup_epochs):
            correct, total, epoch_loss = 0, 0, 0.0
            for x, y in mnist_loader_warmup:
                x, y = x.to(device), y.to(device)
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


# -------------------------------------------------------------------------
# Model state helpers: CPU transfer for safe serialization and networking
# -------------------------------------------------------------------------
def get_model_weights(model):
    return {k: v.cpu().clone() for k, v in model.state_dict().items()}


def set_model_weights(model, weights):
    current_state_dict = model.state_dict()
    for k, v in weights.items():
        if k in current_state_dict:
            current_state_dict[k].copy_(v.to(device))
        else:
            print(f"Warning: Attempted to set weight for non-existent parameter: {k}")
    model.load_state_dict(current_state_dict)


# -------------------------------------------------------------------------
# Delta computation: local weight change relative to start-of-round state
# -------------------------------------------------------------------------
def differential_update(local_weights, previous_weights):
    diff_weights = {}
    for name, param in local_weights.items():
        if name in previous_weights and param.dtype.is_floating_point:
            diff_weights[name] = param - previous_weights[name]
        else:
            diff_weights[name] = param.clone()
    return diff_weights


# -------------------------------------------------------------------------
# Delta sparsification: optional top-k compression for bandwidth reduction
# -------------------------------------------------------------------------
def sparsify_delta(delta: dict, sparsity_level: float) -> dict:
    """
    Applies top-k sparsification per parameter tensor.

    - sparsity_level = 1.0 keeps all values (no compression)
    - sparsity_level = 0.0 sends all zeros
    """
    if sparsity_level >= 1.0:
        return delta
    if sparsity_level <= 0.0:
        return {name: torch.zeros_like(param) for name, param in delta.items()}

    sparsified_delta = {}
    for name, param_tensor in delta.items():
        if param_tensor.dtype.is_floating_point:
            flat_tensor = param_tensor.flatten()
            num_elements = flat_tensor.numel()
            k = int(num_elements * sparsity_level)

            if k == 0 and num_elements > 0:
                k = 1
            if num_elements == 0:
                sparsified_delta[name] = torch.zeros_like(param_tensor)
                continue
            if k >= num_elements:
                sparsified_delta[name] = param_tensor.clone()
                continue

            top_k_values, _ = torch.topk(flat_tensor.abs(), k)
            threshold = top_k_values[-1] if top_k_values.numel() > 0 else 0.0

            mask = flat_tensor.abs() >= threshold

            sparsified_flat_tensor = torch.zeros_like(flat_tensor)
            sparsified_flat_tensor[mask] = flat_tensor[mask]

            sparsified_delta[name] = sparsified_flat_tensor.reshape(param_tensor.shape)
        else:
            sparsified_delta[name] = param_tensor.clone()

    return sparsified_delta


# -------------------------------------------------------------------------
# Metropolis-Hastings aggregation: weighted averaging based on node degrees
# -------------------------------------------------------------------------
def metropolis_average(local_delta, received_deltas_with_nid, neighbors):
    """
    Aggregates deltas using Metropolis-Hastings consensus weights.

    Inputs are dictionaries of tensors on CPU:
      - local_delta: current node delta (after noise and sparsification, if enabled)
      - received_deltas_with_nid: list of (delta_dict, node_id) from neighbors
    """
    all_deltas_data = [(local_delta, NODE_ID)] + received_deltas_with_nid

    degrees = {nid: len(neighbors.get(nid, [])) for _, nid in all_deltas_data}
    d_i = degrees.get(NODE_ID, 0)

    mh_weights = {}
    total_weight = 0.0
    for _, nid in all_deltas_data:
        d_j = degrees.get(nid, 0)
        w = 1.0 / (1.0 + max(d_i, d_j))
        mh_weights[nid] = w
        total_weight += w

    for nid in mh_weights:
        if total_weight > 0:
            mh_weights[nid] /= total_weight
        else:
            mh_weights[nid] = 1.0 if nid == NODE_ID else 0.0

    if local_delta:
        sample_tensor_device = next(iter(local_delta.values())).device
    else:
        sample_tensor_device = torch.device('cpu')

    aggregated_delta = {k: torch.zeros_like(v, device=sample_tensor_device) for k, v in local_delta.items()}

    for name in aggregated_delta:
        if aggregated_delta[name].dtype.is_floating_point:
            for delta_dict, nid in all_deltas_data:
                if name in delta_dict:
                    aggregated_delta[name] += delta_dict[name].to(aggregated_delta[name].device) * mh_weights[nid]
        else:
            if name in local_delta:
                aggregated_delta[name] = local_delta[name].clone().to(aggregated_delta[name].device)

    print(f"[Node {NODE_ID}] Aggregated deltas using Metropolis weights: "
          f"{ {k: f'{v:.3f}' for k, v in mh_weights.items()} }")
    return aggregated_delta


# -------------------------------------------------------------------------
# Momentum + weight update: smooths noisy updates before applying to model
# -------------------------------------------------------------------------
def apply_momentum_and_update_model(aggregated_delta, model):
    """
    Applies aggregated deltas to the model with exponential smoothing:
      momentum_buffer = beta * momentum_buffer + (1 - beta) * delta
    Then updates parameters using LEARNING_RATE scaling.
    """
    global momentum_buffer, MOMENTUM_BETA

    if momentum_buffer is None:
        momentum_buffer = {name: torch.zeros_like(param, device=device) for name, param in model.state_dict().items()}

    with torch.no_grad():
        for name, delta_tensor_cpu in aggregated_delta.items():
            delta_tensor = delta_tensor_cpu.to(device)

            if name in model.state_dict() and model.state_dict()[name].dtype.is_floating_point:
                if (name not in momentum_buffer
                        or momentum_buffer[name].shape != delta_tensor.shape
                        or momentum_buffer[name].device != delta_tensor.device):
                    momentum_buffer[name] = torch.zeros_like(delta_tensor, device=device)

                momentum_buffer[name].mul_(MOMENTUM_BETA).add_(delta_tensor, alpha=(1 - MOMENTUM_BETA))
                model.state_dict()[name].add_(momentum_buffer[name], alpha=LEARNING_RATE)

            elif name in model.state_dict():
                model.state_dict()[name].copy_(delta_tensor)


# -------------------------------------------------------------------------
# Metrics helpers: track global model movement and produce plots/checkpoints
# -------------------------------------------------------------------------
def compute_variance(weight_snapshots):
    if len(weight_snapshots) < 2:
        return 0.0
    deltas_norm = [np.linalg.norm(weight_snapshots[i] - weight_snapshots[i - 1]) for i in range(1, len(weight_snapshots))]
    return np.var(deltas_norm) if deltas_norm else 0.0


def save_model_checkpoint(model):
    os.makedirs(BASE_PATH, exist_ok=True)
    path = os.path.join(BASE_PATH, f"model_node{NODE_ID}_final.pt")
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

    deltas = [np.linalg.norm(weight_snapshots[i] - weight_snapshots[i - 1]) for i in range(1, len(weight_snapshots))]
    if not deltas:
        return

    plt.figure()
    plt.plot(range(2, len(weight_snapshots) + 1), deltas, marker='s', color='red')
    plt.yscale("log")
    plt.xlabel("Round")
    plt.ylabel("Δ Weight Norm (log scale)")
    plt.title(f"Node {NODE_ID} Parameter Update Norm Over Rounds")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(path)
    print(f"[Node {NODE_ID}] Saved variance Parameter plot to {path}")
    plt.close()


# -------------------------------------------------------------------------
# Main training loop: local training + delta exchange + aggregation + logging
# -------------------------------------------------------------------------
def train():
    global prev_loss, prev_acc, previous_round_model_weights, momentum_buffer

    start_server()

    dataset_type = DATASET_TYPE_MAP.get(NODE_ID, "CIFAR10")
    if dataset_type == "MNIST":
        input_channels = 1
    elif dataset_type == "CIFAR10":
        input_channels = 3
    else:
        input_channels = 3

    model = MyModel(input_channels=input_channels).to(device)

    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-5) if OPTIMIZER_NAME == "adam" \
        else optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=0.9, weight_decay=1e-4, nesterov=True)

    loss_fn = nn.CrossEntropyLoss()

    val_loader = get_validation_loader(peer=True)

    if NODE_ID == 7:
        pretrain_on_mnist(model, loss_fn, val_loader)
        data_loader = iter(get_data(batch_size=64))
    else:
        data_loader = iter(get_data(batch_size=128))

    loss_log, acc_log = [], []

    # First reference snapshot used for round-1 delta computation
    previous_round_model_weights = get_model_weights(model)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=ROUNDS)

    # Snapshot tracking of full model movement across rounds
    weight_snapshots = [torch.cat([p.flatten().cpu() for p in model.parameters()]).detach().numpy()]

    for r in range(ROUNDS):
        print(f"\n[Node {NODE_ID}] --- Round {r+1}/{ROUNDS} ---")

        # Model state at the start of the round (reference for local delta)
        start_round_model_weights = get_model_weights(model)

        model.train()
        round_loss, correct, total = 0.0, 0, 0

        # Local training: TAU1 batches per communication round
        for _ in range(TAU1):
            try:
                x, y = next(data_loader)
            except StopIteration:
                data_loader = iter(get_data())
                x, y = next(data_loader)

            x, y = x.to(device), y.to(device)

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

        # Local improvement check is used to decide whether to transmit the delta
        performance_improved = (avg_loss < prev_loss) or (accuracy > prev_acc)

        if performance_improved:
            print(f"[Node {NODE_ID}] Performance improved. Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%")
        else:
            print(f"[Node {NODE_ID}] No improvement in performance. Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%")

        val_acc = evaluate(model, val_loader)
        print(f"[Node {NODE_ID}] Validation Accuracy: {val_acc:.2f}%")

        # Connectivity: only reachable neighbors will participate this round
        active_list = check_active_neighbors()
        print(f"[Node {NODE_ID}] Active neighbors: {len(active_list)}/{len(NEIGHBORS.get(NODE_ID, []))} -> {active_list}")

        # Local delta = current_model - start_of_round_model
        current_local_weights = get_model_weights(model)
        local_delta = differential_update(current_local_weights, start_round_model_weights)

        # Apply channel noise to float tensors (models transmission disturbance)
        current_channel_noise_std = get_current_channel_noise_variance(r + 1) ** 0.5
        noisy_local_delta = {}
        for name, param_tensor in local_delta.items():
            if param_tensor.dtype.is_floating_point:
                noise = torch.randn_like(param_tensor) * current_channel_noise_std
                noisy_local_delta[name] = param_tensor + noise
            else:
                noisy_local_delta[name] = param_tensor.clone()

        # Optional: sparsify to reduce payload size during exchange
        sparsified_noisy_local_delta = sparsify_delta(noisy_local_delta, SPARSITY_LEVEL)

        # Delta transmission policy: only send if improved and peers exist
        if performance_improved and active_list:
            print(f"[Node {NODE_ID}] Performance improved. Sending sparsified and noisy local delta to peers.")
            send_weights(sparsified_noisy_local_delta, target_nodes=active_list)
        elif not active_list:
            print(f"[Node {NODE_ID}] No connected peers. Skipping delta sharing.")
        else:
            print(f"[Node {NODE_ID}] No performance improvement. Not sending local delta.")

        did_aggregate = False
        variance = 0.0

        # Receive deltas collected by the server thread within the time window
        received_deltas_from_peers = receive_weights(min_expected=0, wait_time=10)

        if received_deltas_from_peers:
            print(f"[Node {NODE_ID}] Received deltas from: {[nid for _, nid in received_deltas_from_peers]}")
            print(f"[Node {NODE_ID}] Processed {len(received_deltas_from_peers)} incoming deltas.")

            # Prepare deltas for aggregation and compute a similarity metric for logging
            valid_received_deltas_with_nid = []
            diffs_for_variance = []

            for peer_delta_dict, nid in received_deltas_from_peers:
                try:
                    # Shape adaptation for heterogeneous models (projection if needed)
                    adapted_peer_delta = model.convert_weights(peer_delta_dict)

                    if adapted_peer_delta is not None:
                        local_processed_delta_flat = torch.cat([
                            v.flatten() for k, v in sparsified_noisy_local_delta.items()
                            if v.dtype.is_floating_point
                        ])
                        adapted_peer_delta_flat = torch.cat([
                            v.flatten() for k, v in adapted_peer_delta.items()
                            if v.dtype.is_floating_point
                        ])

                        max_len = max(local_processed_delta_flat.numel(), adapted_peer_delta_flat.numel())
                        if max_len > 0:
                            if local_processed_delta_flat.numel() < max_len:
                                local_processed_delta_flat = torch.cat([
                                    local_processed_delta_flat,
                                    torch.zeros(max_len - local_processed_delta_flat.numel(), device=local_processed_delta_flat.device)
                                ])
                            if adapted_peer_delta_flat.numel() < max_len:
                                adapted_peer_delta_flat = torch.cat([
                                    adapted_peer_delta_flat,
                                    torch.zeros(max_len - adapted_peer_delta_flat.numel(), device=adapted_peer_delta_flat.device)
                                ])

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
                if diffs_for_variance:
                    diffs_tensor = torch.stack(diffs_for_variance)
                    variance = torch.mean(torch.sum(diffs_tensor ** 2, dim=1)).item()
                else:
                    variance = 0.0

                # Metropolis aggregation combines local delta and received deltas
                aggregated_delta = metropolis_average(
                    sparsified_noisy_local_delta,
                    valid_received_deltas_with_nid,
                    NEIGHBORS
                )

                # Apply momentum-smoothed update to the model parameters
                apply_momentum_and_update_model(aggregated_delta, model)
                did_aggregate = True
            else:
                print(f"[Node {NODE_ID}] No valid deltas received for aggregation. Model not updated via aggregation.")
        else:
            print(f"[Node {NODE_ID}] No deltas received. Continuing solo :( .")

        previous_round_model_weights = get_model_weights(model)

        weight_snapshots.append(torch.cat([p.flatten().cpu() for p in model.parameters()]).detach().numpy())

        log_to_google_sheet(
            round_num=r + 1,
            loss=avg_loss,
            accuracy=val_acc,
            variance=variance,
            optimizer_name=OPTIMIZER_NAME,
            training_mode="collaborative" if did_aggregate else "solo",
            connected_peers=active_list,
            momentum_beta=MOMENTUM_BETA
        )

        loss_log.append(avg_loss)
        acc_log.append(val_acc)
        prev_loss, prev_acc = avg_loss, accuracy
        scheduler.step()

    save_log_and_plot(loss_log, acc_log)
    compute_and_plot_weight_variance(weight_snapshots)
    save_model_checkpoint(model)
    print(f"[Node {NODE_ID}] Training completed.")


# -------------------------------------------------------------------------
# Evaluation routine: accuracy on a given DataLoader
# -------------------------------------------------------------------------
def evaluate(model, loader):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            _, pred = torch.max(out, 1)
            correct += (pred == y).sum().item()
            total += y.size(0)
    return 100.0 * correct / total if total > 0 else 0.0


# -------------------------------------------------------------------------
# Entrypoint: run training and log node interruption if stopped manually
# -------------------------------------------------------------------------
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
                "-", "-", "-",
                datetime.datetime.now().isoformat(),
                "-", "interrupted",
                MOMENTUM_BETA
            ]
            sheet.append_row(row, value_input_option="USER_ENTERED")
            print("--> [GSheet] Logged KILLED node.")
        except Exception as e:
            print(f"[ERROR] Failed to log killed node: {e}")
