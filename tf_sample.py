import tensorflow as tf

class SimpleModel(tf.keras.Model):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.dense = tf.keras.layers.Dense(10)

    def call(self, inputs):
        return self.dense(inputs)

model = SimpleModel()
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

@tf.function
def train_step(data, labels):
    with tf.GradientTape() as tape:
        predictions = model(data)
        loss = tf.reduce_mean(tf.keras.losses.MSE(labels, predictions))

    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    return loss

for epoch in range(10):
    for (data, labels) in dataset:
        loss = train_step(data, labels)
    print(f'Epoch {epoch}, Loss: {loss.numpy()}')
