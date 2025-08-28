from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Masking, Dropout
from create_dataset import create_dataset, MAIN_CLASS
from tensorflow.keras.callbacks import ModelCheckpoint
import matplotlib.pyplot as plt
from tensorflow.keras.optimizers import Adam


# =======================
# Parameters
# =======================
CHECKPOINT_PATH = 'analysis/checkpoints/best_model_{epoch:02d}-{val_loss:.2f}.keras'
EPOCHS = 100
BATCH_SIZE = 8
LEARNING_RATE = 0.0001
PREDICTION_THRESHOLD = 0.5
CLASS_LABELS = ["incorrect", MAIN_CLASS]


def split_data(X, y, val_split=0.2, random_state=42):
    """
    Splits the dataset into training and validation sets.
    
    Args:
    X: Feature data.
    y: Labels corresponding to the data.
    val_split (float): Proportion of data to be used for validation.
    random_state: Seed for random number generator to ensure reproducibility.
    
    Returns:
    Split data (X_train, X_test, y_train, y_test).
    """
    return train_test_split(
        X, y,
        test_size=val_split,
        stratify=y,
        random_state=random_state
    )

def create_lstm_model(input_shape, learning_rate=LEARNING_RATE):
    """
    Creates and compiles a Sequential LSTM model.
    
    Args:
    input_shape: Shape of the input data (number of frames, number of features).
    learning_rate.
    
    Returns:
    Compiled LSTM model.
    """
    model = Sequential([
        Masking(mask_value=0., input_shape=input_shape),
        LSTM(64, return_sequences=True),
        Dropout(0.5),
        LSTM(32),
        Dense(32, activation='relu'),
        Dropout(0.3),
        Dense(1, activation='sigmoid'),
    ])
    optimizer = Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

    return model


def get_checkpoint(filepath=CHECKPOINT_PATH):
    """
    Save the best model based on validation loss.
    
    Args:
    filepath (str): Path to save the best model.
    
    Returns:
    Checkpoint.
    """
    return ModelCheckpoint(
        filepath=filepath,
        monitor='val_loss',
        mode='min',
        save_best_only=True,
        verbose=1
    )

def print_graphics(history):
    """
    Debug function that prints plots training and validation accuracy and loss graphs.
    """

    # Accuracy
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()

    # Loss
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

def train_and_evaluate():
    """
    Trains the model and evaluates it.
    """
    X, y = create_dataset()
    X_train, X_test, y_train, y_test = split_data(X, y)

    # Create model
    input_shape = (X.shape[1], X.shape[2])
    model = create_lstm_model(input_shape)

    # Save best checkpoint
    checkpoint = get_checkpoint()

    # Training
    history = model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=[checkpoint]
    )

    print_graphics(history)


if __name__ == "__main__":
    train_and_evaluate()






