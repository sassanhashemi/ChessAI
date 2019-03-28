import chess

from flask import Flask, Response, request
app = Flask(__name__)

board = chess.Board()

@app.route("/")
def index():
    ret = open("index.html").read()
    return ret.replace('start', board.fen())




if __name__ == "__main__":
    app.run(debug=True)
