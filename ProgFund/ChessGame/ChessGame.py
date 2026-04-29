"""
项目名称: 中国象棋游戏
创建日期: 2025-12-08

依赖库:
pygame>=2.6.1
"""

import pygame
import sys

from ChessBoard import ChessBoard


class ChessGame:
    def __init__(self):
        pygame.init()
        self.cell_size = pygame.display.Info().current_h // 16
        self.screen = pygame.display.set_mode((self.cell_size * 10, self.cell_size * 11))
        pygame.display.set_caption('Chinese_Chess_Game')

    def run(self):
        chessboard = ChessBoard()
        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    sys.exit()
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    pos = int(event.pos[0] / self.cell_size - 0.5), int(event.pos[1] / self.cell_size - 0.5)
                    if 0 <= pos[0] <= 8 and 0 <= pos[1] <= 9:
                        chessboard.select_piece(pos)
            chessboard.draw_board(self.cell_size, self.screen)
            pygame.display.update()


if __name__ == '__main__':
    game = ChessGame()
    game.run()