import tensorflow as tf
import chess
import chess.pgn
import numpy as np
import os

# Just disables the warning, doesn't enable AVX/FMA
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


# Define Variables
NUM_TRAINING_GAMES = 5000
NUM_TESTING_GAMES = 100
DATA = "dataset/2016.pgn"




pieces = {"♔" : 0,
          "♕" : 1,
          "♖" : 2,
          "♗" : 3,
          "♘" : 4,
          "♙" : 5,
          "♚" : 6,
          "♛" : 7,
          "♜" : 8,
          "♝" : 9,
          "♞" : 10,
          "♟" : 11}


valid_pieces = ["♔", "♕", "♖", "♗", "♘", "♙",
                "♚", "♛", "♜", "♝", "♞", "♟"]


def get_x_data(unicode):
    x_data = np.zeros((64, 12))
    row = 0
    for char in unicode:
        if char in valid_pieces:
            x_data[row, pieces[char]] = 1

    return x_data

def get_result(result):
    if (result == "1-0"):
        return 0
    elif (result == "0-1"):
        return 1
    elif (result == "1/2-1/2"):
        return 2


def get_games(file=DATA, num_games=1):
    pgn = open(file)
    games = []
    for i in range(num_games):
        games.append(chess.pgn.read_game(pgn))
    return games

def get_training_data(games):
    x_train = []
    y_train = []
    for game in games:
        board = game.board()
        result = get_result(game.headers["Result"])
        for move in game.mainline_moves():
            board.push(move)
            x_train.append(get_x_data(board.unicode()))
            y_train.append(result)
    return np.array(x_train).reshape(-1, 64, 12), np.array(y_train).reshape(-1)


train_games = get_games(num_games=NUM_TRAINING_GAMES)
test_games = get_games(num_games=NUM_TESTING_GAMES)
x_train, y_train = get_training_data(train_games)
x_test, y_test = get_training_data(test_games)




model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(64, 12)),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    #tf.keras.layers.Dense(256, activation='relu'),
    #tf.keras.layers.Dense(128, activation='relu'),
    #tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(3, activation='softmax')
])
"""
model = tf.keras.models.load_model(
    filepath="model.txt",
    custom_objects=None,
    compile=True
)
"""


model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(x_train, y_train, epochs=10)
model.evaluate(x_test, y_test)


tf.keras.models.save_model(
    model=model,
    filepath="model.txt",
    overwrite=True,
    include_optimizer=True
)
