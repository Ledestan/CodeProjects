import pygame

from ChessPiece import PieceFactory


class ChessBoard:
    def __init__(self):
        self.board = [[None] * 9 for _ in range(10)]
        self.init_board()
        self.current_player = 'r'
        self.win = False
        self.selected_piece = None

    def init_board(self):
        board = [['b_c', 'b_m', 'b_x', 'b_s', 'b_j', 'b_s', 'b_x', 'b_m', 'b_c'],
                 [None, None, None, None, None, None, None, None, None],
                 [None, 'b_p', None, None, None, None, None, 'b_p', None],
                 ['b_z', None, 'b_z', None, 'b_z', None, 'b_z', None, 'b_z'],
                 [None, None, None, None, None, None, None, None, None],
                 [None, None, None, None, None, None, None, None, None],
                 ['r_z', None, 'r_z', None, 'r_z', None, 'r_z', None, 'r_z'],
                 [None, 'r_p', None, None, None, None, None, 'r_p', None],
                 [None, None, None, None, None, None, None, None, None],
                 ['r_c', 'r_m', 'r_x', 'r_s', 'r_j', 'r_s', 'r_x', 'r_m', 'r_c']]
        for y in range(10):
            for x in range(9):
                if board[y][x]:
                    self.board[y][x] = PieceFactory.create_piece(board[y][x][0], board[y][x][-1], (x, y))

    def draw_board(self, cell_size, screen):
        screen.fill((255, 127, 38))
        board = pygame.image.load('Assets/board.png')
        board = pygame.transform.scale(board, (cell_size * 9, cell_size * 10))
        screen.blit(board, (cell_size * 0.5, cell_size * 0.5))
        for y in range(10):
            for x in range(9):
                if self.board[y][x]:
                    self.board[y][x].draw_piece(cell_size, screen)
        if self.win:
            win = pygame.image.load(f'Assets/win.png')
            win = pygame.transform.scale(win, (cell_size * 2.5, cell_size))
            screen.blit(win, (cell_size * 3.75, cell_size * 5))
        else:
            if self.current_player == 'r':
                current_player = pygame.image.load(f'Assets/r_turn.png')
            else:
                current_player = pygame.image.load(f'Assets/b_turn.png')
            current_player = pygame.transform.scale(current_player, (cell_size * 2.5, cell_size))
            screen.blit(current_player, (cell_size * 3.75, cell_size * 5))

    def select_piece(self, pos):
        target_x, target_y = pos
        if not self.win:
            if self.selected_piece:
                if self.board[target_y][target_x] and self.current_player == self.board[target_y][target_x].color:
                    self.selected_piece.select = False
                    self.selected_piece = self.board[target_y][target_x]
                    self.selected_piece.select = True
                else:
                    movement = self.selected_piece.movement(pos, self.board, self.current_player)
                    if movement:
                        self.move_piece(pos)
                    else:
                        self.selected_piece.select = False
                        self.selected_piece = None
            else:
                if self.board[target_y][target_x] and self.current_player == self.board[target_y][target_x].color:
                    self.selected_piece = self.board[target_y][target_x]
                    self.selected_piece.select = True

    def move_piece(self, pos):
        target_x, target_y = pos
        current_x, current_y = self.selected_piece.pos
        if self.board[target_y][target_x] and self.board[target_y][target_x].piece_type == 'j':
            self.win = True
        self.board[target_y][target_x], self.board[current_y][current_x] = self.selected_piece, None
        self.selected_piece.pos = pos
        self.selected_piece.select = False
        self.selected_piece = None
        pygame.mixer.music.load('Assets/move.mp3')
        pygame.mixer.music.play()
        self.current_player = 'b' if self.current_player == 'r' else 'r'