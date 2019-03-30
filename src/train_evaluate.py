# File: train_evaluate.py
# Name: Sassan Hashemi
# Date: March 29, 2019

import tensorflow as tf
import chess
import chess.pgn
import numpy as np
import os
from utility import *


# Read and format training and testing data
train_games = get_games(num_games=NUM_TRAINING_GAMES)
test_games = get_games(num_games=NUM_TESTING_GAMES)
x_train, y_train = get_training_data_evaluate(train_games)
x_test, y_test = get_training_data_evaluate(test_games)



"""
evaluate = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(64, 12)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(1024, activation='relu'),
    tf.keras.layers.Dense(1024, activation='relu'),
    tf.keras.layers.Dense(3, activation='softmax')
])
"""

# Load neural network model
evaluate = tf.keras.models.load_model(
    filepath="../models/evaluate",
    custom_objects=None,
    compile=True
)


# Train network and test on data
evaluate.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
evaluate.fit(x_train, y_train, epochs=10)
evaluate.evaluate(x_test, y_test)


# Save network
tf.keras.models.save_model(
    model=evaluate,
    filepath="../models/evaluate",
    overwrite=True,
    include_optimizer=True
)
