import chess
import chess.pgn
import sys
import tensorflow as tf
import numpy as np
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


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
                "♟" : -100,}

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
                "♚", "♛", "♜", "♝", "♞", "♟", "·"]

valid_pieces_nn = ["♔", "♕", "♖", "♗", "♘", "♙",
                "♚", "♛", "♜", "♝", "♞", "♟"]

MAXVAL = 100000
MAXDEPTH = 4


model = tf.keras.models.load_model(
    filepath="model.txt",
    custom_objects=None,
    compile=True
)

def get_x_data(unicode):
    x_data = np.zeros((64, 12))
    row = 0
    for char in unicode:
        if char in valid_pieces_nn:
            x_data[row, pieces[char]] = 1

    return x_data

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
        """
        if piece == valid_pieces[0]:
            piece_value += (index//8)
        elif piece == valid_pieces[4]:
            piece_value += ((63-index)//8)
        elif piece == valid_pieces[5]:
            piece_value += ((63-index)//8)
        elif piece == valid_pieces[6]:
            piece_value -= ((63-index)//8)
        elif piece == valid_pieces[10]:
            piece_value -= (index//8)
        elif piece == valid_pieces[11]:
            piece_value -= (index//8)

        if piece in valid_pieces:
            index += 1
        """
        """
        if (piece_value > 0 and piece_value < 1000):
            piece_value += (1 + ((63-index)//8))
            piece_value -= abs(4 - (index%8))
            index += 1
        elif (piece_value < 0 and piece_value > -1000):
            piece_value -= (1 + (index//8))
            piece_value += abs(4 - (index%8))
            index += 1
        elif (piece in valid_pieces):
            index += 1
            """
        value += piece_value

    return value

def neural_net_value(board):
    input = np.array([get_x_data(board.unicode())])
    output = model.predict(input).flatten()
    value = output[0]-output[1]
    return value

def value(board):
    return material_value(board)/100 + neural_net_value(board)


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



class Game:

    def __init__(self):
        self.board = chess.Board()


    def play(self):
        while (not self.board.is_game_over()):
            print(self.board.unicode())
            best_move = minimax(board=self.board, depth=0, a=-MAXVAL, b=MAXVAL, last_move=None)
            print("Engine move: %s" %(self.board.san(best_move[1])))
            print("Engine evaluation:", round(best_move[0], 3))

            valid_move = False
            while (not valid_move):
                valid_move = True
                move = input("Enter your move: ")
                try:
                    new_move = self.board.parse_san(move)
                except ValueError:
                    valid_move = False

            self.board.push(new_move)


"""
pgn = open("dataset/2017.pgn")
game = chess.pgn.read_game(pgn)
board = game.board()
for move in game.mainline_moves():
    board.push(move)
    print(board.unicode())
    print("Evaluation: ", neural_net_value(board))
"""

game = Game()
game.play()