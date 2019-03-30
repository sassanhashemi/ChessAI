# File: train_move.py
# Name: Sassan Hashemi
# Date: March 29, 2019

import tensorflow as tf
import chess
import chess.pgn
import numpy as np
import os
from utility import *


# Disable Tensorflow Warning
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


# Read and format training and testing data
train_games = get_games(num_games=NUM_TRAINING_GAMES)
test_games = get_games(num_games=NUM_TESTING_GAMES)
x_train, y_train_from, y_train_to = get_training_data_move(train_games)
x_test, y_test_from, y_test_to = get_training_data_move(test_games)


"""
move_from = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(64, 12)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(128, activation='relu'),
    #tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(1024, activation='relu'),
    tf.keras.layers.Dense(1024, activation='relu'),
    tf.keras.layers.Dense(64, activation='softmax')
])

move_to = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(64, 12)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(128, activation='relu'),
    #tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(1024, activation='relu'),
    tf.keras.layers.Dense(1024, activation='relu'),
    tf.keras.layers.Dense(64, activation='softmax')
])
"""

# Load neural network models
move_from = tf.keras.models.load_model(
    filepath="../models/move_from",
    custom_objects=None,
    compile=True
)

move_to = tf.keras.models.load_model(
    filepath="../models/move_to",
    custom_objects=None,
    compile=True
)


# Train networks and test on data
move_from.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
move_from.fit(x_train, y_train_from, epochs=10)
move_from.evaluate(x_test, y_test_from)

move_to.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
move_to.fit(x_train, y_train_to, epochs=10)
move_to.evaluate(x_test, y_test_to)


# Save networks
tf.keras.models.save_model(
    model=move_from,
    filepath="../models/move_from",
    overwrite=True,
    include_optimizer=True
)

tf.keras.models.save_model(
    model=move_to,
    filepath="../models/move_to",
    overwrite=True,
    include_optimizer=True
)
