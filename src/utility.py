# File: utility.py
# Name: Sassan Hashemi
# Date: March 29, 2019

import chess
import chess.pgn
import numpy as np
import tensorflow as tf
import os


# Disable Tensorflow Warning
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


# Variables
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

piece_values = {"♔" : 100000,
                "♕" : 900,
                "♖" : 500,
                "♗" : 310,
                "♘" : 300,
                "♙" : 100,
                "·" : 0,
                "♚" : -100000,
                "♛" : -900,
                "♜" : -500,
                "♝" : -310,
                "♞" : -300,
                "♟" : -100}

valid_pieces = ["♔", "♕", "♖", "♗", "♘", "♙",
                "♚", "♛", "♜", "♝", "♞", "♟"]

valid_pieces_nn = ["♔", "♕", "♖", "♗", "♘", "♙",
                   "♚", "♛", "♜", "♝", "♞", "♟", "·"]

# Define Variables
NUM_TRAINING_GAMES = 1000
NUM_TESTING_GAMES = 100
DATA = "../dataset/2017.pgn"
MAXVAL = 1000000
MAXDEPTH = 4

# Neural network models

evaluate = tf.keras.models.load_model(
    filepath="../models/evaluate",
    custom_objects=None,
    compile=True
)

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


# Training Functions
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


def get_training_data_evaluate(games):
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


def get_training_data_move(games):
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


def get_model_input(board):
    input = np.array(get_x_data(board.unicode())).flatten()
    return input


def material_value(board):
    value  = 0
    if (board.is_game_over()):
        if (board.result() == "1-0"):
            return MAXVAL
        elif (board.result() == "0-1"):
            return -MAXVAL
        else:
            return 0
    index = 0
    for piece in board.unicode():
        piece_value = piece_values.setdefault(piece, 0)
        value += piece_value

    return value


def neural_net_value(board):
    input = np.array([get_x_data(board.unicode())])
    output = evaluate.predict(input).flatten()
    value = output[0]-output[1]
    return value


def value(board):
    return neural_net_value(board)*100 + material_value(board)


# Playing Functions
def minimax(board, depth, a, b, last_move):
    if (depth >= MAXDEPTH or board.is_game_over()):
        return value(board), last_move

    result = 0
    turn = board.turn
    if (turn == chess.WHITE):
        result = -MAXVAL
    else:
        result = MAXVAL


    isort = []
    for move in board.legal_moves:
        board.push(move)
        isort.append((value(board), move))
        board.pop()
    moves = sorted(isort, key=lambda x: x[0], reverse=board.turn)

    #if (depth >= 3):
    moves = moves[:7]

    best_move = moves[0][1]
    for move in [x[1] for x in moves]:
        board.push(move)
        tval = minimax(board, depth+1, a, b, move)
        board.pop()

        if (turn == chess.WHITE):
            if (tval[0] > result):
                best_move = move
            result = max(result, tval[0])
            a = max(a, result)
            if (a >= b):
                break

        else:
            if (tval[0] < result):
                best_move = move
            result = min(result, tval[0])
            b = min(b, result)
            if (a >= b):
                break

    return result, best_move
