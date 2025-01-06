import tensorflow as tf
import numpy as np
from tensorflow.keras.datasets import cifar10
from tensorflow.keras import layers
import tensorflow_compression as tfc
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input
from sklearn.model_selection import train_test_split

def real_awgn(x, stddev):
    """Implements the real additive white gaussian noise channel."""
    awgn = tf.random.normal(tf.shape(x), 0, stddev, dtype=tf.float32)
    y = x + awgn
    return y

def build_model(snrdb, blocksize):
    input_img = Input(shape=(32, 32, 3))
    num_filters = 16
    conv_depth = blocksize

    # Encoder layers
    encoded = tfc.SignalConv2D(
        num_filters, (9, 9), name="layer_0", corr=True, strides_down=2,
        padding="same_zeros", use_bias=True, activation=tfc.GDN(name="gdn_0")
    )(input_img)
    encoded = layers.PReLU(shared_axes=[1, 2])(encoded)
    
    encoded = tfc.SignalConv2D(
        num_filters, (5, 5), name="layer_1", corr=True, strides_down=2,
        padding="same_zeros", use_bias=True, activation=tfc.GDN(name="gdn_1")
    )(encoded)
    encoded = layers.PReLU(shared_axes=[1, 2])(encoded)
    
    encoded = tfc.SignalConv2D(
        num_filters, (5, 5), name="layer_2", corr=True, strides_down=1,
        padding="same_zeros", use_bias=True, activation=tfc.GDN(name="gdn_2")
    )(encoded)
    encoded = layers.PReLU(shared_axes=[1, 2])(encoded)
    
    encoded = tfc.SignalConv2D(
        num_filters, (5, 5), name="layer_3", corr=True, strides_down=1,
        padding="same_zeros", use_bias=True, activation=tfc.GDN(name="gdn_3")
    )(encoded)
    encoded = layers.PReLU(shared_axes=[1, 2])(encoded)
    
    encoded = tfc.SignalConv2D(
        conv_depth, (5, 5), name="layer_out", corr=True, strides_down=1,
        padding="same_zeros", use_bias=True, activation=None
    )(encoded)

    # Channel simulation
    inter_shape = tf.shape(encoded)
    z = layers.Flatten()(encoded)
    noise_stddev = np.sqrt(10 ** (-snrdb / 10))
    dim_z = tf.shape(z)[1]
    z_in = tf.sqrt(tf.cast(dim_z, dtype=tf.float32)) * tf.nn.l2_normalize(z, axis=1)
    z_out = real_awgn(z_in, noise_stddev)
    z_out = tf.reshape(z_out, inter_shape)

    # Decoder layers
    decoded = tfc.SignalConv2D(
        num_filters, (5, 5), corr=False, strides_up=1,
        padding="same_zeros", use_bias=True,
        activation=tfc.GDN(name="igdn_out", inverse=True)
    )(z_out)
    decoded = layers.PReLU(shared_axes=[1, 2])(decoded)
    
    decoded = tfc.SignalConv2D(
        num_filters, (5, 5), corr=False, strides_up=1,
        padding="same_zeros", use_bias=True,
        activation=tfc.GDN(name="igdn_0", inverse=True)
    )(decoded)
    decoded = layers.PReLU(shared_axes=[1, 2])(decoded)
    
    decoded = tfc.SignalConv2D(
        num_filters, (5, 5), corr=False, strides_up=1,
        padding="same_zeros", use_bias=True,
        activation=tfc.GDN(name="igdn_1", inverse=True)
    )(decoded)
    decoded = layers.PReLU(shared_axes=[1, 2])(decoded)
    
    decoded = tfc.SignalConv2D(
        num_filters, (5, 5), corr=False, strides_up=2,
        padding="same_zeros", use_bias=True,
        activation=tfc.GDN(name="igdn_2", inverse=True)
    )(decoded)
    decoded = layers.PReLU(shared_axes=[1, 2])(decoded)
    
    decoded = tfc.SignalConv2D(
        3, (9, 9), corr=False, strides_up=2,
        padding="same_zeros", use_bias=True,
        activation=tf.nn.sigmoid
    )(decoded)

    model = Model(inputs=input_img, outputs=decoded)
    return model

def train_model():
    print("Loading CIFAR-10 dataset...")
    # Load and preprocess data
    (x_train, _), (x_test, _) = cifar10.load_data()
    x_train = x_train.astype('float32') / 255.0
    x_test = x_test.astype('float32') / 255.0
    
    # Split training data into train and validation sets
    x_train, x_val = train_test_split(x_train, test_size=0.1, random_state=42)
    
    print("Building model...")
    # Build model with initial parameters
    snrdb = 10  # Training SNR
    block_size = 16
    model = build_model(snrdb, block_size)
    
    # Compile model
    model.compile(optimizer='adam', 
                 loss='mse',
                 metrics=[tf.keras.metrics.MeanSquaredError(),
                         lambda x, y: tf.image.psnr(x, y, max_val=1.0)])
    
    print("Training model...")
    # Train the model
    history = model.fit(
        x_train, x_train,
        batch_size=128,
        epochs=10,
        validation_data=(x_val, x_val),
        verbose=1
    )
    
    print("Evaluating model...")
    # Evaluate the model
    test_results = model.evaluate(x_test, x_test, verbose=1)
    print(f"Test loss: {test_results[0]:.4f}")
    print(f"Test PSNR: {test_results[2]:.2f} dB")
    
    print("Saving model weights...")
    # Save the model weights
    model.save_weights('classifier_model_weights_rec_train.h5')
    print("Training complete! You can now run the GUI to visualize the results.")

if __name__ == "__main__":
    # Set random seeds for reproducibility
    tf.random.set_seed(42)
    np.random.seed(42)
    
    train_model()