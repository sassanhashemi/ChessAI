# File: game.py
# Name: Sassan Hashemi
# Date: March 29, 2019

import chess
import chess.pgn
import sys
import tensorflow as tf
import numpy as np
import os
from utility import *


# Disable Tensorflow warning
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


class Game:


    def __init__(self, evaluate=False):
        self.evaluate = evaluate
        self.board = chess.Board()


    def intro(self):
        msg = "Welcome to Sassan's command line chess game. I have provided an engine that reccomends moves and can evaluate the board position if chosen. Enjoy!"
        print(msg)


    def goodbye(self):
        if (self.board.result() == "1-0"):
            print("White wins")
        elif (self.board.result() == "0-1"):
            print("Black wins")
        elif (self.board.result() == "1/2-1/2"):
            print("It's a tie")

        print("Thanks for playing!")


    def get_best_move(self):
        input = np.array([get_x_data(self.board.unicode())])

        start_raw = np.array(move_from.__call__(input)).flatten()
        end_raw = np.array(move_to.__call__(input)).flatten()

        start = np.argsort(start_raw)
        end = np.argsort(end_raw)

        moves = []
        scores = []
        for i in range(64):
            for j in range(64):
                move = chess.Move(start[i], end[j])
                if move in self.board.legal_moves:
                    moves.append(move)
                    scores.append(i + j)
        best = np.argmax(np.array(scores))
        return moves[best]


    def game(self):
        while (not self.board.is_game_over()):
            print(self.board.unicode())

            if (self.board.turn == chess.WHITE):
                print("White to move")
            else:
                print("Black to move")

            if (self.evaluate):
                best_move = minimax(board=self.board, depth=0, a=-MAXVAL, b=MAXVAL, last_move=None)
                print("Engine evaluation:", round(best_move[0], 5))
                print("Engine move: %s" %(self.board.san(best_move[1])))

            else: # If next move
                best_move = self.get_best_move()
                print("Engine move: %s" %(self.board.san(best_move)))

            valid_move = False
            while (not valid_move):
                valid_move = True
                move = input("Enter your move: ")
                try:
                    new_move = self.board.parse_san(move)
                except ValueError:
                    valid_move = False

            self.board.push(new_move)


    def play(self):
        self.intro()
        self.game()
        self.goodbye()
        return None
