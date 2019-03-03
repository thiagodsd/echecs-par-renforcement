# Movimentos aleatórios

A função `engineTurn` foi adicionada à versão anterior do código.

```python
def engineTurn(self):
      move = random.choice( list(iter(self.board.legal_moves)) )
      self.board.push(move)
      self.node = self.node.add_variation(move)
      
      self.drawBoard()
      
      if self.board.is_game_over() == True:
        self.game.headers["Result"] = self.board.result()
        pgn = open('test_game.pgn', 'w', encoding='utf-8')
        exporter = chess.pgn.FileExporter(pgn)
        self.game.accept(exporter)
```

Não há grandes detalhes a serem explorados, a função apenas sorteia uma jogada dentre todos os movimentos legais da posição.

## Teste

<p align="center">

![Partida no Lichess: [https://lichess.org/study/Nd5XNecb/UvlZPjvq#0](https://lichess.org/study/Nd5XNecb/UvlZPjvq#0)](https://media.giphy.com/media/1fl3uOHfyOsPnrxwTV/giphy.gif)

</p>    

Código modificado: [bostjan-mejak-chess_0_random.py](code/bostjan-mejak-chess_0_random.py)

