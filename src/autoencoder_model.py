"""
Autoencoder model implementation for anomaly detection
"""
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout
from tensorflow.keras.regularizers import l2

def build_autoencoder(input_dim):
    """
    Build a simple autoencoder model
    
    Args:
        input_dim: int - number of input features
        
    Returns:
        autoencoder, encoder models
    """
    # Encoder
    input_layer = Input(shape=(input_dim,))
    encoded = Dense(128, activation='relu', kernel_regularizer=l2(1e-4))(input_layer)
    encoded = Dropout(0.2)(encoded)
    encoded = Dense(64, activation='relu', kernel_regularizer=l2(1e-4))(encoded)
    encoded = Dropout(0.2)(encoded)
    encoded = Dense(32, activation='relu', kernel_regularizer=l2(1e-4))(encoded)
    
    # Decoder
    decoded = Dense(64, activation='relu', kernel_regularizer=l2(1e-4))(encoded)
    decoded = Dropout(0.2)(decoded)
    decoded = Dense(128, activation='relu', kernel_regularizer=l2(1e-4))(decoded)
    decoded = Dropout(0.2)(decoded)
    decoded = Dense(input_dim, activation='sigmoid')(decoded)
    
    # Models
    autoencoder = Model(input_layer, decoded)
    encoder = Model(input_layer, encoded)
    
    return autoencoder, encoder
