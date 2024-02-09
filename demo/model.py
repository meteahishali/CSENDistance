import tensorflow as tf
tf.random.set_seed(10)

def get_CL_CSEN(feature_size, imageSizeM, imageSizeN):
    input_shape = (imageSizeM, imageSizeN, 1)
    input = tf.keras.Input(shape = feature_size, name='input')
    x_0 = tf.keras.layers.Dense(imageSizeM * imageSizeN, activation = 'relu')(input)
    x_0 = tf.keras.layers.Reshape(input_shape)(x_0) # Size of reshaped proxy from CRC estimation.
    x_0 = tf.keras.layers.Conv2D(64, 5, padding = 'same', activation = 'relu')(x_0)
    x_0 = tf.keras.layers.MaxPooling2D(pool_size=(4, 5))(x_0) # Sparse code shapes.
    x_0 = tf.keras.layers.Conv2D(1, 5, padding = 'same', activation = 'relu')(x_0)
    
    y = tf.keras.layers.Flatten()(x_0)
    y = tf.keras.layers.Dense(1, activation = 'softplus')(y)
    
    model = tf.keras.models.Model(input, y, name='CL-CSEN')
    model.summary()

    return model

def get_CL_CSEN_1D(feature_size, imageSizeM, imageSizeN):
    
    input_shape = (imageSizeM * imageSizeN, 1)
    input = tf.keras.Input(shape = feature_size, name='input')
    x_0 = tf.keras.layers.Dense(imageSizeM * imageSizeN, activation = 'relu')(input)
    x_0 = tf.keras.layers.Reshape(input_shape)(x_0) # Size of reshaped proxy from CRC estimation.
    x_0 = tf.keras.layers.Conv1D(64, 25, padding = 'same', activation = 'relu')(x_0)
    x_0 = tf.keras.layers.MaxPooling1D(pool_size=(20))(x_0) # Sparse code shapes.
    x_0 = tf.keras.layers.Conv1D(1, 25, padding = 'same', activation = 'relu')(x_0)
    
    y = tf.keras.layers.Flatten()(x_0)
    y = tf.keras.layers.Dense(1, activation = 'softplus')(y)
    
    model = tf.keras.models.Model(input, y, name='CL-CSEN-1D')
    model.summary()

    return model

######