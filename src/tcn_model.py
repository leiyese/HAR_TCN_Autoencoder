"""
TCN model implementation for activity classification with proper dilated convolutions
"""
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Dense, Conv1D, BatchNormalization, Dropout, 
    Add, Activation, GlobalAveragePooling1D, LayerNormalization
)
from tensorflow.keras.regularizers import l2


def tcn_residual_block(x, filters, kernel_size, dilation_rate, dropout_rate=0.2):
    """
    TCN residual block med dilated convolutions och residual connections
    
    Args:
        x: Input tensor
        filters: Antal filter för konvolutionerna
        kernel_size: Storlek på konvolutionskärnan
        dilation_rate: Dilation rate för dilated convolution
        dropout_rate: Dropout rate för regularisering
        
    Returns:
        Output tensor efter residual block
    """
    # Första dilated convolution
    conv1 = Conv1D(
        filters=filters,
        kernel_size=kernel_size,
        dilation_rate=dilation_rate,
        padding='causal',  # Causal padding för tidsserier
        kernel_regularizer=l2(0.005)
    )(x)
    conv1 = BatchNormalization()(conv1)
    conv1 = Activation('relu')(conv1)
    conv1 = Dropout(dropout_rate)(conv1)
    
    # Andra dilated convolution
    conv2 = Conv1D(
        filters=filters,
        kernel_size=kernel_size,
        dilation_rate=dilation_rate,
        padding='causal',
        kernel_regularizer=l2(0.005)
    )(conv1)
    conv2 = BatchNormalization()(conv2)
    conv2 = Activation('relu')(conv2)
    conv2 = Dropout(dropout_rate)(conv2)
    
    # Residual connection - anpassa dimensioner om nödvändigt
    if x.shape[-1] != filters:
        # 1x1 convolution för att matcha dimensioner
        residual = Conv1D(filters, 1, padding='same')(x)
    else:
        residual = x
    
    # Lägg till residual connection
    output = Add()([conv2, residual])
    output = Activation('relu')(output)
    
    return output


def build_tcn(input_shape, num_classes, num_filters=64, kernel_size=3, dropout_rate=0.35):
    """
    Bygger en komplett TCN-modell för HAR-klassificering
    
    Args:
        input_shape: tuple - input shape (timesteps, features) för HAR-data
        num_classes: int - antal aktivitetsklasser att klassificera
        num_filters: int - antal filter i TCN-blocken
        kernel_size: int - storlek på konvolutionskärnan
        dropout_rate: float - dropout rate för regularisering
        
    Returns:
        TCN model optimerad för HAR-data
    """
    # Input layer - förväntar sig (timesteps, sensor_features)
    inputs = Input(shape=input_shape, name='sensor_input')
    
    # Initial convolution för att förbereda data
    x = Conv1D(
        filters=num_filters,
        kernel_size=1,
        padding='same',
        name='initial_conv'
    )(inputs)
    x = BatchNormalization(name='initial_bn')(x)
    x = Activation('relu', name='initial_activation')(x)
    
    # TCN-block med ökande dilation rates [1, 2, 4, 8, 16]
    # Detta ger modellen förmåga att se både korta och långa tidssammanhang
    dilation_rates = [1, 2, 4, 8, 16]
    
    for i, dilation_rate in enumerate(dilation_rates):
        x = tcn_residual_block(
            x,
            filters=num_filters,
            kernel_size=kernel_size,
            dilation_rate=dilation_rate,
            dropout_rate=dropout_rate
        )
        # Lägg till layer normalization för stabilitet
        x = LayerNormalization(name=f'layer_norm_{i}')(x)
    
    # Extra TCN-block med större filter för djupare representation
    x = tcn_residual_block(
        x,
        filters=num_filters * 2,
        kernel_size=kernel_size,
        dilation_rate=1,
        dropout_rate=dropout_rate
    )
    
    # Global average pooling för att aggregera tidssekvensen
    # Detta reducerar temporal dimension till en enda vektor
    x = GlobalAveragePooling1D(name='global_avg_pool')(x)
    
    # Dense layers för final klassificering
    x = Dense(128, activation='relu', name='dense_1')(x)
    x = BatchNormalization(name='dense_bn_1')(x)
    x = Dropout(dropout_rate, name='post_dense_dropout_1')(x)
    
    x = Dense(64, activation='relu', name='dense_2')(x)
    x = BatchNormalization(name='dense_bn_2')(x)
    x = Dropout(dropout_rate, name='post_dense_dropout_2')(x)
    
    # Output layer för aktivitetsklassificering
    outputs = Dense(
        num_classes, 
        activation='softmax', 
        name='activity_classification'
    )(x)
    
    # Skapa modell
    model = Model(inputs=inputs, outputs=outputs, name='TCN_HAR_Classifier')
    
    return model


def build_tcn_for_autoencoder_features(latent_dim, num_classes, num_filters=32):
    """
    Specialiserad TCN för komprimerade features från autoencoder
    
    Args:
        latent_dim: int - dimension av autoencoder latent space
        num_classes: int - antal aktivitetsklasser
        num_filters: int - antal filter (mindre än vanlig TCN)
        
    Returns:
        TCN model optimerad för autoencoder features
    """
    # Input för komprimerade features från autoencoder
    inputs = Input(shape=(1, latent_dim), name='autoencoder_features')
    
    # Enklare TCN eftersom data redan är komprimerad
    x = Conv1D(num_filters, 3, padding='same', activation='relu')(inputs)
    x = BatchNormalization()(x)
    x = Dropout(0.2)(x)
    
    # TCN-block med mindre dilation rates för komprimerad data
    for dilation_rate in [1, 2, 4]:
        x = tcn_residual_block(
            x,
            filters=num_filters,
            kernel_size=3,
            dilation_rate=dilation_rate,
            dropout_rate=0.2
        )
    
    # Global pooling och klassificering
    x = GlobalAveragePooling1D()(x)
    x = Dense(64, activation='relu')(x)
    x = Dropout(0.3)(x)
    outputs = Dense(num_classes, activation='softmax')(x)
    
    model = Model(inputs=inputs, outputs=outputs, name='TCN_Autoencoder_Classifier')
    
    return model
