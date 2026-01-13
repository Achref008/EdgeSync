NODE_ID = 7
TOTAL_NODES = 5

NEIGHBORS = {
    1: [2, 3, 4, 7],
    2: [1, 3, 4, 7],
    3: [1, 2, 4, 7],
    4: [1, 2, 3, 7],
    7: [1, 2, 3, 4]
}

PEER_VALIDATION_MAP = {
    1: 2,
    2: 3,
    3: 1,
    4: 2,
    7: 1
    
}

IP_MAP = {
    1: "10.0.1.2",
    2: "10.0.2.2",
    3: "10.0.3.2",
    4: "10.0.4.2",
    7: "10.0.7.2"
}


DATASET_TYPE_MAP = {
    1: 'CIFAR10',       # Jetson 1
    2: 'CIFAR10',       # Jetson 2
    3: 'CIFAR10',       # Jetson 4 
    4: 'CIFAR10',       # Jetson 4
    7: 'MNIST'          # Akida
}

DIRICHLET_ALPHA = 0.1


CENTRAL_NODE_HOST = "10.1.1.1"
CENTRAL_NODE_USER = "sai"
CENTRAL_RESULTS_DIR = "/home/sai/Desktop/achref/DFLtest1/central_node"

PORT_BASE = 5000

AKIDA_CIFAR_RATIO = 0.9
TAU1 = 50         
ROUNDS = 100      
LEARNING_RATE = 1e-3 
 