import socket
import pickle
import threading
import time
import zlib

from config import NODE_ID, NEIGHBORS, IP_MAP, PORT_BASE


# =============================================================================
# Runtime Shared State (server thread <-> training loop)
# =============================================================================
# The server runs continuously in a background thread and appends incoming
# updates to an in-memory buffer. The training loop periodically pulls and clears
# this buffer once per round.
received_weights_buffer = []

# Optional storage for the last known parameters/deltas of nodes that became
# unreachable, enabling simple fault recovery strategies.
last_known_weights = {}

# Protects shared state against race conditions between threads.
lock = threading.Lock()


# =============================================================================
# Serialization helpers (Python objects <-> bytes)
# =============================================================================
def serialize(data):
    """Serialize a Python object into bytes for network transport."""
    return pickle.dumps(data, protocol=pickle.HIGHEST_PROTOCOL)


def deserialize(data):
    """Deserialize bytes back into the original Python object."""
    return pickle.loads(data)


# =============================================================================
# TCP framing helper (receive exact number of bytes)
# =============================================================================
def recv_all(sock, num_bytes):
    """
    Receive exactly `num_bytes` from a TCP socket.

    TCP is a stream protocol: a single send() may arrive in multiple recv() calls.
    This function loops until the requested amount is received or the connection
    closes unexpectedly.
    """
    data = b""
    while len(data) < num_bytes:
        chunk = sock.recv(num_bytes - len(data))
        if not chunk:
            raise ConnectionError("Socket closed before receiving expected bytes.")
        data += chunk
    return data


# =============================================================================
# Outgoing communication: push delta updates to neighbors
# =============================================================================
def send_weights(weights_to_send, target_nodes=None, max_retries=3):
    """
    Send a model update payload to one or more target nodes.

    Expected payload format:
      - weights_to_send: typically a dict of parameter deltas computed in main.py

    Notes:
      - This function only transmits the payload.
      - It does NOT compute deltas, apply momentum, or modify weights.
      - Uses length-prefixed framing + zlib compression.
    """
    if target_nodes is None:
        target_nodes = NEIGHBORS.get(NODE_ID, [])

    # Frame format: [4-byte length][zlib-compressed pickle(sender_id, payload)]
    raw_data = serialize((NODE_ID, weights_to_send))
    compressed = zlib.compress(raw_data)

    msg_len = len(compressed).to_bytes(4, byteorder="big")
    payload = msg_len + compressed

    for neighbor in target_nodes:
        ip = IP_MAP[neighbor]
        port = PORT_BASE + neighbor

        retries = 0
        backoff = 1

        while retries < max_retries:
            try:
                with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                    s.settimeout(5.0)  # Avoid hanging on unreachable nodes
                    s.connect((ip, port))
                    s.sendall(payload)
                    s.shutdown(socket.SHUT_WR)  # Cleanly finish sending

                print(f"[Node {NODE_ID}] Sent payload to Node {neighbor}")
                break

            except Exception as e:
                retries += 1
                print(f"[Node {NODE_ID}] Send failed to Node {neighbor}: {e}")

                if retries < max_retries:
                    print(f"[Node {NODE_ID}] Retry {retries}/{max_retries} in {backoff}s")
                    time.sleep(backoff)
                    backoff = min(backoff * 2, 10)  # Exponential backoff cap
                else:
                    print(f"[Node {NODE_ID}] Giving up on Node {neighbor} after {max_retries} retries")


# =============================================================================
# Incoming communication: collect received updates from the buffer
# =============================================================================
def receive_weights(min_expected=0, wait_time=10):
    """
    Collect incoming payloads that the server thread already received.

    Parameters
    ----------
    min_expected : int
        If > 0, return early once at least this many payloads are collected.
    wait_time : float
        Maximum time to wait (seconds) while polling the shared buffer.

    Returns
    -------
    list of tuples:
        Each item is (received_payload, sender_id)
        where received_payload is typically a dict of deltas.
    """
    global received_weights_buffer

    start = time.time()
    collected = []

    while time.time() - start < wait_time:
        with lock:
            if received_weights_buffer:
                collected.extend(received_weights_buffer)
                received_weights_buffer.clear()

        if min_expected > 0 and len(collected) >= min_expected:
            break

        time.sleep(0.1)  # Prevent CPU busy-waiting

    # Final drain after timeout
    with lock:
        collected.extend(received_weights_buffer)
        received_weights_buffer.clear()

    print(f"[Node {NODE_ID}] Collected {len(collected)} incoming items this round.")
    return collected


# =============================================================================
# Fault handling utilities
# =============================================================================
def recover_from_failure(node_id):
    """
    Return the last known payload for a failed/unreachable node (if available).

    This is a simple mechanism to keep training stable under intermittent
    connectivity by substituting a cached value when needed.
    """
    try:
        last_payload = last_known_weights.get(node_id)
        if last_payload is not None:
            print(f"[Node {NODE_ID}] Using last known payload for Node {node_id}.")
            return last_payload

        print(f"[Node {NODE_ID}] No cached payload found for Node {node_id}.")
        return None

    except Exception as e:
        print(f"[Node {NODE_ID}] Recovery error for Node {node_id}: {e}")
        return None


def send_failure_alert(node_id, message):
    """
    Hook for failure notifications.

    Replace this print with logging/telemetry integration if needed
    (e.g., Prometheus, MQTT, email, Slack).
    """
    try:
        print(f"[ALERT] Node {node_id}: {message}")
    except Exception as e:
        print(f"[Node {NODE_ID}] Alert error: {e}")


# =============================================================================
# Server: persistent listener thread (receives payloads and buffers them)
# =============================================================================
def start_server():
    """
    Start a background TCP server that continuously receives payloads.

    Incoming frame format:
      - 4 bytes: big-endian payload length
      - N bytes: zlib-compressed pickle(sender_id, payload)

    Received items are appended to `received_weights_buffer` as:
      (payload, sender_id)
    """

    def listen():
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            s.bind(("0.0.0.0", PORT_BASE + NODE_ID))
            s.listen()

            print(f"[Node {NODE_ID}] Server listening on port {PORT_BASE + NODE_ID}")

            while True:
                conn, addr = s.accept()
                with conn:
                    try:
                        # Read framed message length then payload bytes
                        msg_len_bytes = recv_all(conn, 4)
                        msg_len = int.from_bytes(msg_len_bytes, byteorder="big")

                        data = recv_all(conn, msg_len)

                        # Decode payload
                        decompressed = zlib.decompress(data)
                        sender_id, received_payload = deserialize(decompressed)

                        # Buffer for the training loop to process
                        with lock:
                            received_weights_buffer.append((received_payload, sender_id))

                        print(f"[Node {NODE_ID}] Received payload from Node {sender_id} ({addr[0]})")

                    except Exception as e:
                        print(f"[Node {NODE_ID}] Error processing data from {addr[0]}: {e}")

    thread = threading.Thread(target=listen, daemon=True)
    thread.start()
