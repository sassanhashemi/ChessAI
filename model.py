import tensorflow as tf
import chess
import chess.pgn
import numpy as np
import os


# Just disables the warning, doesn't enable AVX/FMA
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


# Define Variables
NUM_TRAINING_GAMES = 300
NUM_TESTING_GAMES = 100
DATA = "dataset/2010.pgn"




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

valid_pieces_nn = ["♔", "♕", "♖", "♗", "♘", "♙",
                "♚", "♛", "♜", "♝", "♞", "♟", "·"]


def get_x_data(unicode):
    x_data = np.zeros((64, 12))
    row = 0
    for char in unicode:
        if char in valid_pieces:
            x_data[row, pieces[char]] = 1
        if char in valid_pieces_nn:
            row += 1

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
    y_train_from = []
    y_train_to = []
    for game in games:
        board = game.board()
        result = get_result(game.headers["Result"])
        for move in game.mainline_moves():
            x_train.append(get_x_data(board.unicode()))
            board.push(move)
            y_train_from.append(move.from_square)
            y_train_to.append(move.to_square)
    return np.array(x_train).reshape(-1, 64, 12), np.array(y_train_from).reshape(-1), np.array(y_train_to).reshape(-1)


train_games = get_games(num_games=NUM_TRAINING_GAMES)
test_games = get_games(num_games=NUM_TESTING_GAMES)
x_train, y_train_from, y_train_to = get_training_data(train_games)
x_test, y_test_from, y_test_to = get_training_data(test_games)


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

move_from = tf.keras.models.load_model(
    filepath="models/move_from",
    custom_objects=None,
    compile=True
)


move_to = tf.keras.models.load_model(
    filepath="models/move_to",
    custom_objects=None,
    compile=True
)



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




tf.keras.models.save_model(
    model=move_from,
    filepath="models/move_from",
    overwrite=True,
    include_optimizer=True
)

tf.keras.models.save_model(
    model=move_to,
    filepath="models/move_to",
    overwrite=True,
    include_optimizer=True
)
