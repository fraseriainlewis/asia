import os
os.environ["TF_USE_LEGACY_KERAS"] = "1"  # For TF2.16+.

import tensorflow as tf
import tensorflow_gnn as tfgnn

# 1. THE DATASET
# Patients: P0, P1, P2
# Features: F0 (Age), F1 (BMI)
# (P1, F1) is missing!
data = [
    {"p": 0, "f": 0, "val": 45.0}, # P0 Age
    {"p": 0, "f": 1, "val": 22.0}, # P0 BMI
    {"p": 1, "f": 0, "val": 50.0}, # P1 Age
    # {"p": 1, "f": 1, "val": ???}, # P1 BMI is MISSING
    {"p": 2, "f": 0, "val": 48.0}, # P2 Age
    {"p": 2, "f": 1, "val": 25.0}, # P2 BMI
]

# 2. DEFINE SCHEMA
# We have two types of nodes (Bipartite) and one type of edge
schema_pbtxt = """
node_sets {
  key: "patient"
  value { features { key: "id" value { dtype: DT_INT32 shape { dim { size: 1 } } } } }
}
node_sets {
  key: "feature"
  value { features { key: "id" value { dtype: DT_INT32 shape { dim { size: 1 } } } } }
}
edge_sets {
  key: "observation"
  value {
    source: "patient"
    target: "feature"
    features { key: "value" value { dtype: DT_FLOAT shape { dim { size: 1 } } } }
  }
}
"""
schema = tfgnn.parse_schema(schema_pbtxt)

# 3. BUILD THE GRAPHTENSOR
# Convert our list into tensors
p_indices = tf.constant([d["p"] for d in data], dtype=tf.int32)
f_indices = tf.constant([d["f"] for d in data], dtype=tf.int32)
v_values = tf.constant([[d["val"]] for d in data], dtype=tf.float32)

graph = tfgnn.GraphTensor.from_pieces(
    node_sets={
        "patient": tfgnn.NodeSet.from_fields(
            sizes=tf.constant([3]),
            features={"id": tf.expand_dims(tf.range(3), -1)}
        ),
        "feature": tfgnn.NodeSet.from_fields(
            sizes=tf.constant([2]),
            features={"id": tf.expand_dims(tf.range(2), -1)}
        )
    },
    edge_sets={
        "observation": tfgnn.EdgeSet.from_fields(
            sizes=tf.constant([len(data)]),
            adjacency=tfgnn.Adjacency.from_indices(
                source=("patient", p_indices),
                target=("feature", f_indices)
            ),
            features={"value": v_values}
        )
    }
)

# 4. THE GNN MODEL
def build_model(graph_spec):
    input_graph = tf.keras.layers.Input(type_spec=graph_spec)
    
    # Init: Turn IDs into hidden state vectors
    def init_node_state(node_set, node_set_name):
        return tf.keras.layers.Embedding(5, 16)(node_set["id"])

    graph_state = tfgnn.keras.layers.MapFeatures(
        node_sets={"patient": init_node_state, "feature": init_node_state}
    )(input_graph)

    # Message Passing: Feature -> Patient
    # This tells the patient about their clinical values
    graph_state = tfgnn.keras.layers.GraphUpdate(
        node_sets={
            "patient": tfgnn.keras.layers.NodeSetUpdate(
                {"observation": tfgnn.keras.layers.SimpleConv(
                    tf.keras.layers.Dense(16, "relu"), "mean", edge_input_feature="value")},
                tfgnn.keras.layers.NextStateFromConcat(tf.keras.layers.Dense(16))
            )
        }
    )(graph_state)

    return tf.keras.Model(input_graph, graph_state)

# 5. THE PREDICTION HEAD (Imputer)
# This takes a Patient vector and a Feature vector and predicts the edge value
class ImputerHead(tf.keras.layers.Layer):
    def __init__(self):
        super().__init__()
        self.mlp = tf.keras.Sequential([
            tf.keras.layers.Dense(16, activation='relu'),
            tf.keras.layers.Dense(1)
        ])

    def call(self, patient_embed, feature_embed):
        # Combine the two embeddings to predict the clinical value
        combined = tf.concat([patient_embed, feature_embed], axis=-1)
        return self.mlp(combined)

# 6. TRAINING AND INFERENCE
gnn = build_model(graph.spec)
head = ImputerHead()
optimizer = tf.keras.optimizers.Adam(0.01)

# Training Loop (Simplified)
for epoch in range(100):
    with tf.GradientTape() as tape:
        # Get embeddings from GNN
        updated_graph = gnn(graph)
        p_embeds = updated_graph.node_sets["patient"][tfgnn.HIDDEN_STATE]
        f_embeds = updated_graph.node_sets["feature"][tfgnn.HIDDEN_STATE]
        
        # Predict values for the EXISTING edges
        # We gather the embeddings for every source/target in our data
        p_gathered = tf.gather(p_embeds, p_indices)
        f_gathered = tf.gather(f_embeds, f_indices)
        preds = head(p_gathered, f_gathered)
        
        loss = tf.reduce_mean(tf.square(v_values - preds))
        
    grads = tape.gradient(loss, gnn.trainable_variables + head.trainable_variables)
    optimizer.apply_gradients(zip(grads, gnn.trainable_variables + head.trainable_variables))

# 7. PREDICT MISSING VALUE (Patient 1, Feature 1)
final_graph = gnn(graph)
p1_vector = final_graph.node_sets["patient"][tfgnn.HIDDEN_STATE][1:2]
f1_vector = final_graph.node_sets["feature"][tfgnn.HIDDEN_STATE][1:2]

imputed_bmi = head(p1_vector, f1_vector)
print(f"Imputed BMI for Patient 1: {imputed_bmi.numpy()[0][0]:.2f}")




