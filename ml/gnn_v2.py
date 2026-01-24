import os
os.environ["TF_USE_LEGACY_KERAS"] = "1"  # For TF2.16+.

import numpy as np
import tensorflow as tf
import tensorflow_gnn as tfgnn
from sklearn.neighbors import kneighbors_graph
import matplotlib.pyplot as plt

# Set seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

def generate_synthetic_data(n_samples=1000, missing_rate=0.5):
    """
    Generates a noisy sine wave. 
    X are the coordinates, Y is the value to predict.
    """
    # 1. Generate covariates (X) and target (Y)
    x = np.linspace(0, 10, n_samples).reshape(-1, 1)
    # y = sin(x) + noise
    noise = np.random.normal(0, 0.1, size=(n_samples, 1))
    y = np.sin(x) + noise
    
    # 2. Create a mask for missing data
    # 1.0 = Observed (Training), 0.0 = Missing (Imputation target)
    mask = np.random.binomial(1, 1 - missing_rate, size=(n_samples, 1)).astype(np.float32)
    
    # For the input features, we give the GNN the X coordinates.
    # We DO NOT give it Y. The GNN must infer Y based on graph neighbors.
    node_features = x.astype(np.float32)
    
    return node_features, y.astype(np.float32), mask

def build_graph_tensor(node_features, y_all, mask, k_neighbors=5):
    """
    Converts raw data into a TF-GNN GraphTensor.
    Builds edges based on K-Nearest Neighbors of the features.
    """
    num_nodes = node_features.shape[0]

    # 1. Create Edges using K-Nearest Neighbors (KNN)
    # We connect nodes that have similar feature values (X)
    print(f"Building KNN graph (k={k_neighbors})...")
    adj_matrix = kneighbors_graph(node_features, k_neighbors, mode='connectivity', include_self=False)
    
    # Extract source and target indices from the sparse matrix
    sources, targets = adj_matrix.nonzero()
    
    # 2. Define the Node Set
    # We attach the features (X), the targets (Y), and the mask to the nodes.
    # Note: In a real deployment, you wouldn't pass the ground truth of missing data 
    # into the 'features', but here we store Y in the graph for loss calculation convenience.
    nodes = tfgnn.NodeSet.from_fields(
        sizes=tf.constant([num_nodes]),
        features={
            "features": tf.constant(node_features),
            "targets": tf.constant(y_all),
            "mask": tf.constant(mask)
        }
    )

    # 3. Define the Edge Set
    # Adjacency defines the structure. 
    adjacency = tfgnn.Adjacency.from_indices(
        source=("nodes", tf.constant(sources, dtype=tf.int32)),
        target=("nodes", tf.constant(targets, dtype=tf.int32)),
    )
    
    edges = tfgnn.EdgeSet.from_fields(
        sizes=tf.constant([len(sources)]),
        adjacency=adjacency
    )

    # 4. Construct GraphTensor
    graph_tensor = tfgnn.GraphTensor.from_pieces(
        node_sets={"nodes": nodes},
        edge_sets={"edges": edges}
    )
    
    return graph_tensor

def build_gnn_model(graph_spec):
    """
    Builds a Keras model using TF-GNN layers.
    Architecture: Graph Convolution (GCN) -> Dense -> Regression Output
    """
    input_graph = tf.keras.layers.Input(type_spec=graph_spec)
    
    # Extract features from the 'nodes' set
    x = input_graph.node_sets["nodes"]["features"]
    
    # --- GNN Layer 1 ---
    # GCNConv propagates information from neighbors to the node
    gnn_layer_1 = tfgnn.keras.layers.GCNConv(units=32, activation="relu")
    x = gnn_layer_1(input_graph, node_set_name="nodes", edge_set_name="edges", node_features=x)
    
    # --- GNN Layer 2 ---
    gnn_layer_2 = tfgnn.keras.layers.GCNConv(units=16, activation="relu")
    x = gnn_layer_2(input_graph, node_set_name="nodes", edge_set_name="edges", node_features=x)

    # --- Readout / Prediction Head ---
    # Standard Dense layers to map node embeddings to a single regression value
    x = tf.keras.layers.Dense(16, activation="relu")(x)
    predictions = tf.keras.layers.Dense(1)(x) # Linear output for regression
    
    return tf.keras.Model(inputs=input_graph, outputs=predictions)

def run_experiment():
    # 1. Prepare Data
    print("--- Generating Data ---")
    features, targets, mask = generate_synthetic_data(n_samples=500, missing_rate=0.6)
    
    # 2. Build Graph
    print("--- Building GraphTensor ---")
    graph = build_graph_tensor(features, targets, mask, k_neighbors=10)
    
    # TF-GNN models usually expect a batch dimension, even if batch_size=1 (whole graph)
    # We expand dims to simulate a batch of 1 graph.
    graph_batch = tf.expand_dims(graph, axis=0)
    
    # 3. Build Model
    print("--- Building Model ---")
    model = build_gnn_model(graph.spec)
    
    # 4. Define Loss and Optimizer
    # Key Statistical Concept: We only compute loss on Observed Nodes (mask == 1).
    # We use sample_weight to zero out loss for missing nodes.
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.01),
        loss='mse'
    )
    
    # Prepare inputs for Keras fit
    # Input: The graph structure and features
    # Target: The known Y values (targets)
    # Sample Weight: The mask (0 for missing, 1 for observed)
    
    # Note: TF-GNN data handling usually extracts targets internally or via datasets,
    # but for this simple single-graph example, we pass arrays explicitly.
    # Since batch size is 1 (1 graph), we need to reshape targets/weights to match 
    # the flattened node output of the model.
    train_targets = np.expand_dims(targets, axis=0) # Shape (1, N, 1)
    train_weights = np.expand_dims(mask, axis=0)    # Shape (1, N, 1)
    
    print("--- Training ---")
    history = model.fit(
        graph_batch, 
        train_targets, 
        sample_weight=train_weights,
        epochs=200, 
        verbose=0
    )
    print(f"Final Training Loss (MSE on observed): {history.history['loss'][-1]:.4f}")
    
    # 5. Predict / Impute
    print("--- Imputing Missing Data ---")
    predicted_all = model.predict(graph_batch)
    
    # Remove batch dim
    predicted_all = np.squeeze(predicted_all)
    
    # 6. Evaluate on Missing Data (The "Test" set)
    # Invert mask: 1 where data was missing, 0 where it was observed
    missing_mask = (1 - mask).flatten().astype(bool)
    
    truth_missing = targets.flatten()[missing_mask]
    pred_missing = predicted_all[missing_mask]
    
    mse_missing = np.mean((truth_missing - pred_missing)**2)
    print(f"MSE on Missing Data (Imputation Accuracy): {mse_missing:.4f}")

    # 7. Visualization
    plt.figure(figsize=(10, 6))
    
    # Plot real underlying function
    plt.plot(features, targets, color='gray', alpha=0.3, label='Ground Truth (All)')
    
    # Plot observed training points
    observed_mask = mask.flatten().astype(bool)
    plt.scatter(features[observed_mask], targets[observed_mask], 
                color='blue', label='Observed (Train)', s=10)
    
    # Plot imputed points
    plt.scatter(features[missing_mask], pred_missing, 
                color='red', label='Imputed (Predicted)', s=10)
    
    plt.title("GNN Imputation: Recovering Missing Sine Wave Data")
    plt.legend()
    plt.show()

if __name__ == "__main__":
    run_experiment()

