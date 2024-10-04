import numpy as np
import math
from sklearn.model_selection import GridSearchCV
from typing import List, Dict, Union, Tuple, Optional
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Input, SeparableConv1D, Dense, Flatten
from tensorflow.keras.layers import GlobalMaxPooling1D, GlobalAveragePooling1D, MaxPooling1D, AveragePooling1D
from tensorflow.keras.layers import Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from scikeras.wrappers import KerasClassifier

from data_util.data_util import load_data
from eval import evaluate_predictions

# Constants
N_EPOCHS = 150
VALIDATION_SPLIT = 0.2
BATCH_SIZE = 16
E_STOP_PATIENCE = 5
CORAL_LAMBDA = 0.1


def create_nn(
        input_shape: Optional[Tuple[int]] = None,
        n_classes: Optional[int] = 2,
        hidden_units: Optional[Union[int, List[int]]] = None,
        kernel_size: int = 3,
        dropout_rate: float = 0.1,
        pooling: bool = False,
        input_spatial_reduction: Optional[Union[int, float]] = None
):
    if hidden_units is None:
        hidden_units = []
    elif isinstance(hidden_units, int):
        hidden_units = [hidden_units]

    model = Sequential()
    if input_shape is not None:
        model.add(Input(shape=input_shape))
    if input_spatial_reduction is not None:
        if isinstance(input_spatial_reduction, float):
            input_spatial_reduction = int(math.ceil(input_spatial_reduction * input_shape[0]))
        model.add(AveragePooling1D(
            pool_size=2 * input_spatial_reduction, strides=input_spatial_reduction, padding='same')
        )
    for n_hidden in hidden_units:
        model.add(Dropout(dropout_rate))
        model.add(SeparableConv1D(n_hidden, kernel_size, padding='same', activation='relu'))
        if pooling:
            model.add(MaxPooling1D(pool_size=max(2, kernel_size - 1)))
        model.add(Flatten())
    if n_classes is not None:
        model.add(Dropout(dropout_rate))
        if n_classes == 2:
            model.add(Dense(1, activation='sigmoid'))
        else:
            model.add(Dense(n_classes, activation='softmax'))

    return model


def run_cnn_model(dataset, ifm_nifm):
    df, n_features = load_data(dataset, ifm_nifm)
    df['train_test'] = df['train_test'].astype(bool)
    X_train = df[df['train_test']].iloc[:, :n_features]
    X_test = df[~df['train_test']].iloc[:, :n_features]
    y_train = df[df['train_test']]['y']
    y_test = df[~df['train_test']]['y']

    X_src_train_tensor = np.array(X_train).reshape(1, -1, n_features)
    y_src_train_split = np.array(y_train).reshape(1, -1)

    X_src_test_tensor = np.array(X_test).reshape(1, -1, n_features)
    y_src_test_split = np.array(y_test).reshape(1, -1)

    print(X_src_train_tensor.shape, y_src_train_split.shape, X_src_test_tensor.shape, y_src_test_split.shape)
    cv = GridSearchCV(
        estimator=KerasClassifier(
            model=create_nn,
            loss="binary_crossentropy",
            optimizer="adam",
            metrics=['accuracy'],
            epochs=N_EPOCHS,
            batch_size=BATCH_SIZE,
            validation_split=VALIDATION_SPLIT,
            verbose=0,
            callbacks=[EarlyStopping(monitor='val_accuracy', patience=E_STOP_PATIENCE)]
        ),
        param_grid={
            'model__input_shape': [X_src_train_tensor.shape[1:]],
            'model__hidden_units': [
                [512], [512, 512], [512, 512, 512], [1024], [1024, 1024], [1024, 1024, 1024]
            ],
            'optimizer__learning_rate': [1e-3, 5e-4]
        },
        n_jobs=1
    )
    cv.fit(X_src_train_tensor, y_src_train_split)
    predictions = cv.predict(X_src_test_tensor)
    evaluate_predictions('DeepNN', y_src_test_split, predictions)
