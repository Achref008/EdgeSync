import socket
import pickle
import threading
import time
import zlib
from config import NODE_ID, NEIGHBORS, IP_MAP, PORT_BASE
import torch

# -------------------------------------------------------------------------
# Shared receive buffer and synchronization
# -------------------------------------------------------------------------
# The server thread appends incoming (delta_dict, sender_id) tuples to this list.
# The training loop fetches and clears it once per round via receive_weights().
received_weights_buffer = []

# Stores last known weights of failed nodes (optional recovery mechanism).
last_known_weights = {}

# Ensures thread-safe access to received_weights_buffer and last_known_weights.
lock = threading.Lock()


# -------------------------------------------------------------------------
# Serialization helpers (pickle) + compression (zlib)
# -------------------------------------------------------------------------
def serialize(data):
    """Serializes a Python object to bytes for network transmission."""
    return pickle.dumps(data)


def deserialize(data):
    """Deserializes bytes back into a Python object."""
    return pickle.loads(data)


# -------------------------------------------------------------------------
# Socket utility: read exactly N bytes or fail
# -------------------------------------------------------------------------
def recv_all(sock, num_bytes):
    """
    Receives exactly num_bytes from a socket.
    Raises ConnectionError if the connection closes before all bytes are read.
    """
    data = b''
    while len(data) < num_bytes:
        chunk = sock.recv(num_bytes - len(data))
        if not chunk:
            raise ConnectionError("Socket connection broken before receiving all data.")
        data += chunk
    return data


# -------------------------------------------------------------------------
# Sender: transmit a delta dictionary to neighbor nodes
# -------------------------------------------------------------------------
def send_weights(weights_to_send, target_nodes=None, max_retries=3):
    """
    Sends a precomputed delta dictionary to target nodes.

    This function assumes:
      - main.py already computed the delta (and applied any noise/sparsification)
      - this module only handles transport (serialize -> compress -> send)
    """
    if target_nodes is None:
        target_nodes = NEIGHBORS.get(NODE_ID, [])

    raw_data = serialize((NODE_ID, weights_to_send))
    compressed_data = zlib.compress(raw_data)

    # Prefix payload with 4-byte length header for framed reads on the receiver side
    msg_len = len(compressed_data).to_bytes(4, byteorder='big')
    payload = msg_len + compressed_data

    for neighbor in target_nodes:
        ip = IP_MAP[neighbor]
        port = PORT_BASE + neighbor

        retries = 0
        backoff_time = 1

        while retries < max_retries:
            try:
                with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                    s.settimeout(5.0)
                    s.connect((ip, port))
                    s.sendall(payload)
                    s.shutdown(socket.SHUT_WR)

                print(f"[Node {NODE_ID}] Sent data to Node {neighbor}")
                break

            except Exception as e:
                retries += 1
                print(f"[Node {NODE_ID}] Failed to send data to Node {neighbor}: {e}")

                if retries < max_retries:
                    print(f"Retrying ({retries}/{max_retries}) in {backoff_time} seconds...")
                    time.sleep(backoff_time)
                    backoff_time = min(backoff_time * 2, 10)
                else:
                    print(f"[Node {NODE_ID}] Max retries reached. Failed to send data to Node {neighbor}.")
                    break


# -------------------------------------------------------------------------
# Receiver: collect deltas placed into the buffer by the server thread
# -------------------------------------------------------------------------
def receive_weights(min_expected=0, wait_time=10):
    """
    Retrieves (delta_dict, sender_id) tuples from the shared buffer.

    - Does not open sockets.
    - Used by the training loop once per round.
    - Waits up to wait_time seconds to gather messages arriving asynchronously.
    """
    global received_weights_buffer

    start_time = time.time()
    collected_weights = []

    while time.time() - start_time < wait_time:
        with lock:
            if received_weights_buffer:
                collected_weights.extend(received_weights_buffer)
                received_weights_buffer.clear()

        if min_expected > 0 and len(collected_weights) >= min_expected:
            break

        time.sleep(0.1)

    with lock:
        collected_weights.extend(received_weights_buffer)
        received_weights_buffer.clear()

    print(f"[Node {NODE_ID}] Collected {len(collected_weights)} incoming items this round.")
    return collected_weights


# -------------------------------------------------------------------------
# Failure handling: optional access to cached weights from failed nodes
# -------------------------------------------------------------------------
def recover_from_failure(node_id):
    """
    Returns the last cached weights of node_id if available.
    This is an optional mechanism if a node becomes unreachable mid-run.
    """
    try:
        last_weights = last_known_weights.get(node_id)
        if last_weights:
            print(f"[Node {NODE_ID}] Recovering from failure. Using last known weights for Node {node_id}.")
            return last_weights

        print(f"[Node {NODE_ID}] No last known weights found for Node {node_id}.")
        return None

    except Exception as e:
        print(f"[Node {NODE_ID}] Error recovering from failure for Node {node_id}: {e}")
        return None


def send_failure_alert(node_id, message):
    """
    Reports a failure event. This is a stub that can be replaced by
    a real notification system (logging service, messaging, etc.).
    """
    try:
        print(f"[ALERT] Node {node_id}: {message}")
    except Exception as e:
        print(f"[Node {NODE_ID}] Error sending failure alert: {e}")


# -------------------------------------------------------------------------
# Server thread: persistent listener to receive deltas from peers
# -------------------------------------------------------------------------
def start_server():
    """
    Starts a daemon server thread that listens on (PORT_BASE + NODE_ID).

    Expected incoming payload format:
      - 4-byte big-endian length prefix
      - zlib-compressed pickle((sender_id, delta_dict))
    """
    def listen():
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            s.bind(('0.0.0.0', PORT_BASE + NODE_ID))
            s.listen()

            print(f"[Node {NODE_ID}] Server started, listening on port {PORT_BASE + NODE_ID}")

            while True:
                conn, addr = s.accept()
                with conn:
                    try:
                        msg_len_bytes = recv_all(conn, 4)
                        msg_len = int.from_bytes(msg_len_bytes, byteorder='big')

                        data = recv_all(conn, msg_len)

                        decompressed_data = zlib.decompress(data)
                        sender_id, received_delta = deserialize(decompressed_data)

                        with lock:
                            received_weights_buffer.append((received_delta, sender_id))

                        print(f"[Node {NODE_ID}] Received delta from Node {sender_id} at {addr[0]}")

                    except Exception as e:
                        print(f"[Node {NODE_ID}] Error processing incoming data from {addr[0]}: {e}")

    thread = threading.Thread(target=listen, daemon=True)
    thread.start()
