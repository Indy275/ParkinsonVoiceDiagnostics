from eval import evaluate_predictions
import configparser

config = configparser.ConfigParser()
config.read('settings.ini')
plot_fimp = config.getboolean('OUTPUT_SETTINGS', 'plot_fimp')
print_intermediate = config.getboolean('OUTPUT_SETTINGS', 'print_intermediate')


def create_dnn_model(input_shape, l2_lambda=0.1):
    from tensorflow import keras
    from keras import layers
    from keras import regularizers

    model = keras.Sequential()

    # Input layer
    model.add(layers.Input(shape=input_shape))

    # First hidden layer
    model.add(layers.Dense(128, activation='relu', kernel_regularizer=regularizers.l2(l2_lambda)))
    model.add(layers.Dropout(0.5))  # Dropout layer with 50% rate

    # Second hidden layer
    model.add(layers.Dense(128, activation='relu', kernel_regularizer=regularizers.l2(l2_lambda)))
    model.add(layers.Dropout(0.5))  # Dropout layer with 50% rate

    # Output layer
    model.add(layers.Dense(1, activation='sigmoid'))  # Sigmoid for binary classification

    return model


def run_dnn_model(X_train, X_test, y_train, y_test, df, test_indices):
    model = create_dnn_model((X_train.get_shape()[1],)) 
    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy', 'AUC'])

    verbose = 1 if print_intermediate else 0
    model.fit(X_train, y_train, epochs=15, batch_size=5, validation_split=0.4, verbose=verbose)
    predictions = (model.predict(X_test) >= 0.5).astype(int).reshape(-1)
    df.loc[test_indices, 'preds'] = predictions

    file_scores = evaluate_predictions('DeepNN' + 'Window', y_test, df.loc[test_indices, 'preds'].tolist())

    samples_preds = df.loc[test_indices, ['subject_id', 'preds']].groupby('subject_id').agg({'preds': lambda x: x.mode()[0]}).reset_index()
    samples_ytest = df.loc[test_indices, ['subject_id', 'y']].groupby('subject_id').agg({'y': lambda x: x.mode()[0]}).reset_index()

    subj_scores = evaluate_predictions('DeepNN'+'SubjID', samples_ytest['y'].tolist(), samples_preds['preds'].tolist())
    
    return file_scores, subj_scores