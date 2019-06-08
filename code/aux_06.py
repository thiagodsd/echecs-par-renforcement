import chess

DICT1 = {'p':-100,'r':-525,'n':-350,'b':-350,'q':-1000,'k':-10000,
         '.':0,
         'K':10000,'Q':1000,'B':350,'N':350,'R':525,'P':100}
DICT2 = {-100:(chess.PAWN, chess.BLACK),
         -525:(chess.ROOK, chess.BLACK), 
         -350:(chess.KNIGHT, chess.BLACK), 
         -350:(chess.BISHOP, chess.BLACK), 
         -1000:(chess.QUEEN, chess.BLACK), 
         -10000:(chess.KING, chess.BLACK),
          0:'.',
         10000:(chess.KING, chess.WHITE),
         1000:(chess.QUEEN, chess.WHITE),
         350:(chess.BISHOP, chess.WHITE),
         350:(chess.KNIGHT, chess.WHITE),
         525:(chess.ROOK, chess.WHITE),
         100:(chess.PAWN, chess.WHITE)}

def symbol2value(list):
    for i, v in enumerate(list):
        if v in DICT1:
            list[i] = DICT1[v]
    return list


def list2piece(list):
    for i, v in enumerate(list):
        if v in DICT2:
            list[i] = DICT2[v]
    return list


def ceil_to_tens(x):
    return int(ceil(x / 10.0)) * 10