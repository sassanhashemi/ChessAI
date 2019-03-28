import chess
import sys


piece_values = {"♔" : 100000,
                "♕" : 900,
                "♖" : 500,
                "♗" : 350,
                "♘" : 300,
                "♙" : 100,

                "♚" : -100000,
                "♛" : -900,
                "♜" : -500,
                "♝" : -350,
                "♞" : -300,
                "♟" : -100,}

MAXVAL = 10000000
MAXDEPTH = 4


def value(board):
    value  = 0
    if (board.is_game_over()):
        if (board.result() == "1-0"):
            return MAXVAL
        elif (board.result() == "0-1"):
            return -MAXVAL
        else:
            return 0

    for piece in board.unicode():
        value += piece_values.setdefault(piece, 0)
    return value



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

    if (depth >= 3):
        moves = moves[:10]

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

            valid_move = False
            while (not valid_move):
                valid_move = True
                move = input("Enter your move: ")
                try:
                    new_move = self.board.parse_san(move)
                except ValueError:
                    valid_move = False

            self.board.push(new_move)




game = Game()
game.play()
