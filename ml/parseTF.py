import tensorflow as tf
import os

# 1. THE DATASET: A simple list of patient records
original_data = [
    {"name": "Patient_A", "age": 45, "marker": 1.5},
    {"name": "Patient_B", "age": 50, "marker": 1.8},
    {"name": "Patient_C", "age": 48, "marker": 1.6},
]

tfrecord_file = "clinical_data.tfrecord"

# 2. HELPER FUNCTIONS: Convert data types to tf.train.Feature
def _bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    if isinstance(value, type(tf.constant(0))):
        value = value.numpy() # BytesList won't unpack a string from an EagerTensor.
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value.encode()]))

def _int64_feature(value):
    """Returns an int64_list from a bool / enum / int / uint."""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def _float_feature(value):
    """Returns a float_list from a float / double."""
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

# 3. WRITING: Create the binary TFRecord file
print(f"--- Writing data to {tfrecord_file} ---")
with tf.io.TFRecordWriter(tfrecord_file) as writer:
    for record in original_data:
        # Create a feature dictionary
        feature = {
            'name': _bytes_feature(record['name']),
            'age': _int64_feature(record['age']),
            'marker': _float_feature(record['marker']),
        }
        # Wrap the features in an Example protocol buffer
        example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
        # Serialize to binary and write
        writer.write(example_proto.SerializeToString())
print("Writing complete.\n")

# 4. READING: Use tf.data.TFRecordDataset
raw_dataset = tf.data.TFRecordDataset(tfrecord_file)

# 5. PARSING: Convert the binary back to Tensors
# We must define the schema so TensorFlow knows how to "un-byte" the data
feature_description = {
    'name': tf.io.FixedLenFeature([], tf.string),
    'age': tf.io.FixedLenFeature([], tf.int64),
    'marker': tf.io.FixedLenFeature([], tf.float32),
}

def _parse_function(example_proto):
    # Parse the input tf.train.Example proto using the dictionary above
    return tf.io.parse_single_example(example_proto, feature_description)

parsed_dataset = raw_dataset.map(_parse_function)

# 6. VERIFICATION: Display the results
print("--- Reading and Parsing Data back from TFRecord ---")
for record in parsed_dataset:
    # We use .numpy() to see the actual values
    name = record['name'].numpy().decode('utf-8') # Decode binary string to text
    age = record['age'].numpy()
    marker = record['marker'].numpy()
    
    print(f"Decoded -> Name: {name}, Age: {age}, Marker: {marker:.2f}")

# Optional: Cleanup the file
if os.path.exists(tfrecord_file):
    os.remove(tfrecord_file)
    
