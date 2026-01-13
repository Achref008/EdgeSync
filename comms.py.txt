import socket
import pickle
import threading
import time
import zlib
from config import NODE_ID, NEIGHBORS, IP_MAP, PORT_BASE
import torch  # Import torch for .clone() if needed, though mostly handled in main.py

# Global list to store weights received by the server thread
received_weights_buffer = []
# prev_weights and momentum are now explicitly managed by main.py
last_known_weights = {}  # Store last known weights of failed nodes

# Lock for thread-safe access to received_weights_buffer
lock = threading.Lock()

def serialize(data):
    """Converts a Python object into a byte stream."""
    return pickle.dumps(data)

def deserialize(data):
    """Converts a byte stream back into a Python object."""
    return pickle.loads(data)

def recv_all(sock, num_bytes):
    """Helper function to receive a specific number of bytes from a socket."""
    data = b''
    while len(data) < num_bytes:
        chunk = sock.recv(num_bytes - len(data))
        if not chunk:
            raise ConnectionError("Socket connection broken before receiving all data.")
        data += chunk
    return data

def send_weights(weights_to_send, target_nodes=None, max_retries=3):
    """
    Sends the provided data (expected to be a dictionary of deltas from main.py)
    to target nodes. This function does NOT calculate differentials or apply momentum.
    """
    if target_nodes is None:
        target_nodes = NEIGHBORS.get(NODE_ID, [])
    
    # Serialize and compress the data (expected to be a delta dictionary)
    raw_data = serialize((NODE_ID, weights_to_send))
    compressed_data = zlib.compress(raw_data)
    msg_len = len(compressed_data).to_bytes(4, byteorder='big')  
    payload = msg_len + compressed_data

    # Send the compressed payload to all the neighbors
    for neighbor in target_nodes:
        ip = IP_MAP[neighbor]
        port = PORT_BASE + neighbor
        retries = 0
        backoff_time = 1  # Reset backoff for each neighbor
        while retries < max_retries:
            try:
                with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                    s.settimeout(5.0)  # Connection timeout
                    s.connect((ip, port))
                    s.sendall(payload)
                    
                    s.shutdown(socket.SHUT_WR)  # Graceful close of write side
                print(f"[Node {NODE_ID}] Sent data to Node {neighbor}")
                break  
            except Exception as e:
                retries += 1
                print(f"[Node {NODE_ID}] Failed to send data to Node {neighbor}: {e}")
                if retries < max_retries:
                    print(f"Retrying ({retries}/{max_retries}) in {backoff_time} seconds...")
                    time.sleep(backoff_time)
                    backoff_time = min(backoff_time * 2, 10)  # Exponential backoff
                else:
                    print(f"[Node {NODE_ID}] Max retries reached. Failed to send data to Node {neighbor}.")
                    break  

def receive_weights(min_expected=0, wait_time=10):
    """
    Retrieves weights (deltas) from the shared buffer populated by the server thread.
    This function does NOT initiate new socket connections. It just collects from the buffer.
    """
    global received_weights_buffer
    
    start_time = time.time()
    collected_weights = []

    # Wait for the specified time, periodically checking the buffer
    while time.time() - start_time < wait_time:
        with lock:
            if received_weights_buffer:
                # Add all current items from buffer to collected_weights
                collected_weights.extend(received_weights_buffer)
                received_weights_buffer.clear()  # Clear buffer after collecting them
        
        # If enough weights are collected (and min_expected is set), break early
        if min_expected > 0 and len(collected_weights) >= min_expected:
            break
        
        # Small sleep to prevent busy-waiting
        time.sleep(0.1) 
    
    # Collect any remaining weights after the wait_time has passed
    with lock:
        collected_weights.extend(received_weights_buffer)
        received_weights_buffer.clear()
        
    print(f"[Node {NODE_ID}] Collected {len(collected_weights)} incoming items this round.")
    return collected_weights

def recover_from_failure(node_id):
    """This function retrieves the last known weights of a node in case of failure."""
    try:
        last_weights = last_known_weights.get(node_id)
        if last_weights:
            print(f"[Node {NODE_ID}] Recovering from failure. Using last known weights for Node {node_id}.")
            return last_weights
        else:
            print(f"[Node {NODE_ID}] No last known weights found for Node {node_id}.")
            return None
    except Exception as e:
        print(f"[Node {NODE_ID}] Error recovering from failure for Node {node_id}: {e}")
        return None

def send_failure_alert(node_id, message):
    """This function sends a failure alert via a notification system."""
    try:
        print(f"[ALERT] Node {node_id}: {message}")
    except Exception as e:
        print(f"[Node {NODE_ID}] Error sending failure alert: {e}")

def start_server():
    """Starts the persistent server thread to listen for incoming weights (deltas)."""
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
                        sender_id, received_delta = deserialize(decompressed_data)  # Expecting a delta dictionary
                        
                        with lock:
                            received_weights_buffer.append((received_delta, sender_id))
                        print(f"[Node {NODE_ID}] Received delta from Node {sender_id} at {addr[0]}")

                    except Exception as e:
                        print(f"[Node {NODE_ID}] Error processing incoming data from {addr[0]}: {e}")

    thread = threading.Thread(target=listen, daemon=True)
    thread.start()
