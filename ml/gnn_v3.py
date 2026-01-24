import os
os.environ["TF_USE_LEGACY_KERAS"] = "1"  # For TF2.16+.

import tensorflow as tf
import tensorflow_gnn as tfgnn
import numpy as np
import matplotlib.pyplot as plt

# ==========================================
# 1. Generate Data with Missing Values
# ==========================================
def generate_graph_data(num_nodes=30, missing_rate=0.3):
    # Create simple node indices and edges (a ring structure for simplicity)
    sources = np.arange(num_nodes)
    targets = (np.arange(num_nodes) + 1) % num_nodes
    
    # Ground Truth: A sine wave pattern
    x_vals = np.linspace(0, 4 * np.pi, num_nodes)
    ground_truth = np.sin(x_vals).astype(np.float32).reshape(-1, 1)
    
    # Create a mask: 1 = Known (Train), 0 = Missing (Predict)
    # We randomly hide some nodes
    rand_vals = np.random.rand(num_nodes, 1)
    mask = (rand_vals > missing_rate).astype(np.float32)
    
    # Input Features: The ground truth, but we Zero out the missing values
    # The GNN must look at neighbors to recover these zeros.
    input_features = ground_truth * mask
    
    # Create the TF-GNN GraphTensor
    graph = tfgnn.GraphTensor.from_pieces(
        node_sets={
            "nodes": {
                "features": input_features,
                "ground_truth": ground_truth,
                "mask": mask
            }
        },
        edge_sets={
            "edges": {
                "source": sources,
                "target": targets
            }
        }
    )
    return graph, ground_truth, mask

# Generate the data
graph_data, gt_raw, mask_raw = generate_graph_data()

exit()
print(f"Graph generated: {int(np.sum(mask_raw))} training nodes, {int(len(mask_raw) - np.sum(mask_raw))} missing nodes.")

# ==========================================
# 2. Create TF-GNN Model
# ==========================================

def build_model(graph_spec):
    input_graph = tf.keras.Input(type_spec=graph_spec)
    
    # 1. Initial Feature Mapping: Project scalar input to higher dim
    # We target the 'features' attribute of the 'nodes' set
    graph = tfgnn.keras.layers.MapFeatures(
        node_sets_fn=lambda node_set, node_set_name: 
            tf.keras.layers.Dense(16, activation="relu")(node_set["features"])
    )(input_graph)

    # 2. Graph Convolution (Message Passing)
    # This allows nodes to aggregate information from their neighbors
    # We stack two layers to allow info to travel 2 hops
    graph = tfgnn.keras.layers.GCNHomographUpdate(
        units=16, 
        activation="relu"
    )(graph)
    
    graph = tfgnn.keras.layers.GCNHomographUpdate(
        units=16, 
        activation="relu"
    )(graph)

    # 3. Readout: Extract the updated node states
    node_features = graph.node_sets["nodes"][tfgnn.HIDDEN_STATE]
    
    # 4. Final Prediction: Project back to 1 dimension (scalar value)
    output = tf.keras.layers.Dense(1)(node_features)
    
    return tf.keras.Model(inputs=input_graph, outputs=output)

model = build_model(graph_data.spec)

# ==========================================
# 3. Custom Loss Function (Masked MSE)
# ==========================================

def masked_mse_loss(y_true, y_pred):
    """
    y_true shape: [Batch, 2] -> [Ground_Truth_Value, Mask_Value]
    y_pred shape: [Batch, 1] -> [Predicted_Value]
    """
    # Separate the actual target value and the mask
    # We packed them together in the training loop setup below
    targets = y_true[:, 0:1]
    mask = y_true[:, 1:2]
    
    # Calculate squared error
    squared_error = tf.square(targets - y_pred)
    
    # Apply mask: Zero out error for missing nodes
    masked_error = squared_error * mask
    
    # Average only over the number of visible nodes (avoid div by zero)
    loss = tf.reduce_sum(masked_error) / (tf.reduce_sum(mask) + 1e-6)
    return loss

# Prepare 'y_true' for Keras fit. 
# We combine Ground Truth and Mask into one array so the Loss function can see both.
y_train_combined = np.hstack([gt_raw, mask_raw])

# Compile
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.01), loss=masked_mse_loss)

# Train
# Note: TF-GNN inputs usually expect a batch dimension. We wrap our single graph in a batch of 1.
train_ds = tf.data.Dataset.from_tensors((graph_data, y_train_combined))

print("\nStarting Training...")
history = model.fit(train_ds, epochs=100, verbose=0)
print(f"Final Loss: {history.history['loss'][-1]:.4f}")

# ==========================================
# 4. Predict Missing Values
# ==========================================

# Run the model on the graph
predictions = model.predict(train_ds)

# Visualization and Evaluation
print("\n--- Results ---")
print(f"{'Node':<5} | {'Type':<8} | {'True':<8} | {'Pred':<8} | {'Diff':<8}")
print("-" * 50)

for i in range(len(predictions)):
    is_missing = mask_raw[i] == 0
    type_str = "MISSING" if is_missing else "Train"
    
    # Only printing a few for brevity, but ensuring we show missing ones
    if is_missing or i < 5: 
        actual = gt_raw[i][0]
        pred = predictions[i][0]
        diff = abs(actual - pred)
        print(f"{i:<5} | {type_str:<8} | {actual:<8.4f} | {pred:<8.4f} | {diff:<8.4f}")

# Calculate Error on specifically the missing nodes
missing_indices = np.where(mask_raw == 0)[0]
missing_true = gt_raw[missing_indices]
missing_pred = predictions[missing_indices]
mse_missing = np.mean((missing_true - missing_pred)**2)

print(f"\nMSE on Missing Nodes: {mse_missing:.4f}")




