
-> ML framework

-> foloseste Tensori -> tenosorul este un multi dimensional array(ca pytorch?)
 - tipuri de tensori: 
	 -- .constant([1, 2, 3]) -> immutable
	 -- .variable([1, 2, 3]) -> Mutable, folositi la training
 * Distributii si proprietati:
	 -- z = tf.random.normal((3,3), mean=0, stddev=1) # Normal Distribution 
	 -- z = tf.random.uniform((3,3), minval=0, maxval=10) # Uniform Distribution
	 -- .shape
	 -- .dtyoe
	 -- .numpy() -> converteste tensorul in vecoor numpy


-> folosea pe vechi static graphs, iar mai apoi a introdus eager execution 
* Graph Execution:
	* cele mai bune pt performanta
	* folosite in productie: TF Serving, TF Lite
	* computational graphs are optimized before execution 
* Eager Execution:
	* Mai usor de dat debug
	* mai incete decat grafuri statice
	* folosite la crearea unui prototip rapid


->  tenserflow calculeaza gradientii automat folosind backpropagation

#### TenserFlow ofera tf.keras pt construirea usoara a unui model

-> Sequential model:
from tensorflow import keras
from tensorflow.keras import layers

model = keras.Sequential([
    layers.Dense(64, activation="relu"),
    layers.Dense(32, activation="relu"),
    layers.Dense(1)  # Output layer
])

-> Model Compilation & Training:
model.compile(optimizer="adam", loss="mse")
model.fit(X_train, y_train, epochs=10, batch_size=32)


#### TenserFlow ofera si un fel de dataframe, pt cand datasetul e prea mare

dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))
dataset = dataset.shuffle(100).batch(32).prefetch(tf.data.AUTOTUNE)

-> optimizat pt GPU si TPU si e tot paralel

#### TenserFlow permite deplyment ul prin TF Serving:
tensorflow_model_server --rest_api_port=8501 --model_name=my_model --model_base_path="/models/"


#### TensorFlow merge convertit la mobile si embedded devices:
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

with open("model.tflite", "wb") as f:
    f.write(tflite_model)



