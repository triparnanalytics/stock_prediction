import numpy as np
import pandas as pd
import pickle
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, SimpleRNN, Conv1D, MaxPooling1D, Flatten
from tensorflow.keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
import tensorflow as tf


# Define the ETFs and lookback periods
etfs = ['XLB', 'XLE', 'XLF', 'XLI', 'XLK', 'XLP', 'XLU', 'XLV', 'XLY', 'SPY']
lookback_periods = [10, 30, 60, 180, 250]
model_types = ['lstm', 'rnn', 'cnn']

def create_sequences(data, seq_length):
    X, y = [], []
    for i in range(len(data) - seq_length - 1):
        X.append(data[i:(i + seq_length), 0])
        y.append(1 if data[i + seq_length + 1, 0] > data[i + seq_length, 0] else 0)
    return np.array(X), np.array(y)

def create_model(model_type, seq_length):
    if model_type == 'lstm':
        model = Sequential([
            LSTM(50, return_sequences=True, input_shape=(seq_length, 1)),
            LSTM(50, return_sequences=False),
            Dense(25),
            Dense(1, activation='sigmoid')
        ])
    elif model_type == 'rnn':
        model = Sequential([
            SimpleRNN(50, return_sequences=True, input_shape=(seq_length, 1)),
            SimpleRNN(50, return_sequences=False),
            Dense(25),
            Dense(1, activation='sigmoid')
        ])
    elif model_type == 'cnn':
        model = Sequential([
            Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=(seq_length, 1)),
            MaxPooling1D(pool_size=2),
            Flatten(),
            Dense(50, activation='relu'),
            Dense(1, activation='sigmoid')
        ])
    
    model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])
    return model


results = []

for ticker in etfs:
    # Load the data
    df = pd.read_csv(f'{ticker}.csv', header=[0, 1], index_col=0)
    df.index = pd.to_datetime(df.index)
    data = df['Close'].values.reshape(-1, 1)

    # Get unique years
    years = df.index.year.unique()

    for year in years:
        # Filter data to include only data up to and including this year (cumulative)
        year_mask = df.index.year <= year
        year_data = data[year_mask]
        
        for seq_length in lookback_periods:
            # Check if we have enough data for this sequence length
            if len(year_data) < seq_length + 2:
                continue
                
            # Create sequences from year-specific RAW data (before scaling)
            X, y = create_sequences(year_data, seq_length)
            
            if len(X) < 2:  # Skip if not enough data
                continue
            
            # Split data into train and test sets (chronologically)
            samples = X.shape[0]
            X_train, X_test = X[:int(samples * 0.5)], X[int(samples * 0.5):]
            y_train, y_test = y[:int(samples * 0.5)], y[int(samples * 0.5):]
            samples = X_test.shape[0]
            X_val, X_test = X_test[:int(samples * 0.5)], X_test[int(samples * 0.5):]
            y_val, y_test = y_test[:int(samples * 0.5)], y_test[int(samples * 0.5):]
            
            # Fit scaler ONLY on training data (to avoid look-ahead bias)
            scaler = MinMaxScaler(feature_range=(0, 1))
            # Reshape for scaling: flatten sequences, scale, then reshape back
            n_train, seq_len = X_train.shape[0], X_train.shape[1]
            X_train_reshaped = X_train.reshape(-1, 1)
            scaler.fit(X_train_reshaped)
            X_train_scaled = scaler.transform(X_train_reshaped).reshape(n_train, seq_len, 1)
            
            # Transform validation and test sets using training scaler parameters
            n_val = X_val.shape[0]
            X_val_reshaped = X_val.reshape(-1, 1)
            X_val_scaled = scaler.transform(X_val_reshaped).reshape(n_val, seq_len, 1)
            
            n_test = X_test.shape[0]
            X_test_reshaped = X_test.reshape(-1, 1)
            X_test_scaled = scaler.transform(X_test_reshaped).reshape(n_test, seq_len, 1)
            
            # Use scaled versions
            X_train = X_train_scaled
            X_test = X_test_scaled

            for model_type in model_types:
                model_checkpoint = ModelCheckpoint(
                    f'checkpoints/{ticker}_{year}_{seq_length}_{model_type}.keras',
                    monitor='val_loss',
                    save_best_only=True,
                    verbose=0
                )
                
                model = create_model(model_type, seq_length)
                history = model.fit(X_train, y_train, batch_size=32, epochs=5, 
                                    validation_data=(X_val, y_val), 
                                    callbacks=[model_checkpoint], verbose=0)
                
                train_loss = history.history['loss'][-1]
                val_loss = history.history['val_loss'][-1]
                model.load_weights(f'checkpoints/{ticker}_{year}_{seq_length}_{model_type}.keras')
                y_pred = (model.predict(X_test) > 0.5).astype(int)
                accuracy = accuracy_score(y_test, y_pred)

                results.append({
                    'ETF': ticker,
                    'Year': year,
                    'Lookback': seq_length,
                    'Model': model_type,
                    'Accuracy': accuracy,
                    'Train Loss': train_loss,
                    'Validation Loss': val_loss
                })
                # Create DataFrame and save to CSV
                results_df = pd.DataFrame(results)
                results_df.to_csv('etf_model_accuracies.csv', index=False)

                print(f"{ticker} - Year {year} - Lookback {seq_length} - {model_type.upper()} Model - Accuracy: {accuracy:.4f}")

# Create DataFrame and save to CSV
results_df = pd.DataFrame(results)
results_df.to_csv('etf_model_accuracies.csv', index=False)
#print("Results saved to etf_model_accuracies.csv")
