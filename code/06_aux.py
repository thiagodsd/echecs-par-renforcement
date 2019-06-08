DICT1 = {'p':-60,'r':-50,'n':-40,'b':-30,'q':-20,'k':-10,'.':0,'K':10,'Q':20,'B':30,'N':40,'R':50,'P':60}
DICT2 = {-60:(chess.PAWN, chess.BLACK),
         -50:(chess.ROOK, chess.BLACK), 
         -40:(chess.KNIGHT, chess.BLACK), 
         -30:(chess.BISHOP, chess.BLACK), 
         -20:(chess.QUEEN, chess.BLACK), 
         -10:(chess.KING, chess.BLACK),
          0:'.',
         10:(chess.KING, chess.WHITE),
         20:(chess.QUEEN, chess.WHITE),
         30:(chess.BISHOP, chess.WHITE),
         40:(chess.KNIGHT, chess.WHITE),
         50:(chess.ROOK, chess.WHITE),
         60:(chess.PAWN, chess.WHITE)}