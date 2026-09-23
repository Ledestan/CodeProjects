from abc import ABC, abstractmethod

import pygame


class MoveStrategy(ABC):
    @abstractmethod
    def movement(self, current_pos, target_pos, board, current_player):
        pass


class PieceFactory:
    @staticmethod
    def create_piece(color, piece_type, pos):
        strategies = {
            "j": General,
            "s": Advisor,
            "x": Elephant,
            "c": Chariot,
            "m": Horse,
            "p": Cannon,
            "z": Pawn,
        }
        return strategies[piece_type](color, piece_type, pos)


class ChessPiece:
    def __init__(self, color, piece_type, pos):
        self.color = color
        self.piece_type = piece_type
        self.pos = pos
        self.select = False
        self.move_strategy = None

    def draw_piece(self, cell_size, screen):
        pos = cell_size * (self.pos[0] + 0.5), cell_size * (self.pos[1] + 0.5)
        image = pygame.image.load(f"data/{self.color}_{self.piece_type}.png")
        image = pygame.transform.scale(image, (cell_size, cell_size))
        screen.blit(image, pos)
        if self.select:
            frame = pygame.image.load(f"data/{self.color}_selected.png")
            frame = pygame.transform.scale(frame, (cell_size, cell_size))
            screen.blit(frame, pos)

    def movement(self, target_pos, board, current_player):
        return self.move_strategy.movement(self.pos, target_pos, board, current_player)


class GeneralStrategy(MoveStrategy):
    def movement(self, current_pos, target_pos, board, current_player):
        current_x, current_y = current_pos
        target_x, target_y = target_pos
        if current_player == "r":
            return (3 <= target_x <= 5 and 7 <= target_y <= 9) and abs(
                current_x - target_x
            ) + abs(current_y - target_y) == 1
        else:
            return (3 <= target_x <= 5 and 0 <= target_y <= 2) and abs(
                current_x - target_x
            ) + abs(current_y - target_y) == 1


class AdvisorStrategy(MoveStrategy):
    def movement(self, current_pos, target_pos, board, current_player):
        current_x, current_y = current_pos
        target_x, target_y = target_pos
        if current_player == "r":
            return (
                (3 <= target_x <= 5 and 7 <= target_y <= 9)
                and abs(current_y - target_y) == 1
                and abs(current_x - target_x) == 1
            )
        else:
            return (
                (3 <= target_x <= 5 and 0 <= target_y <= 2)
                and abs(current_y - target_y) == 1
                and abs(current_x - target_x) == 1
            )


class ElephantStrategy(MoveStrategy):
    def movement(self, current_pos, target_pos, board, current_player):
        current_x, current_y = current_pos
        target_x, target_y = target_pos
        if current_player == "r":
            return (
                target_y >= 5
                and abs(current_x - target_x) == 2
                and abs(current_y - target_y) == 2
                and board[(current_y + target_y) // 2][(current_x + target_x) // 2]
                is None
            )
        else:
            return (
                target_y <= 4
                and abs(current_x - target_x) == 2
                and abs(current_y - target_y) == 2
                and board[(current_y + target_y) // 2][(current_x + target_x) // 2]
                is None
            )


class ChariotStrategy(MoveStrategy):
    def movement(self, current_pos, target_pos, board, current_player):
        current_x, current_y = current_pos
        target_x, target_y = target_pos
        if current_x == target_x:
            start, end = min(current_y, target_y) + 1, max(current_y, target_y)
            for y in range(start, end):
                if board[y][current_x] is not None:
                    return False
            return True
        elif current_y == target_y:
            start, end = min(current_x, target_x) + 1, max(current_x, target_x)
            for x in range(start, end):
                if board[current_y][x] is not None:
                    return False
            return True
        return False


class HorseStrategy(MoveStrategy):
    def movement(self, current_pos, target_pos, board, current_player):
        current_x, current_y = current_pos
        target_x, target_y = target_pos
        if abs(current_x - target_x) == 2 and abs(current_y - target_y) == 1:
            return board[current_y][(current_x + target_x) // 2] is None
        elif abs(current_y - target_y) == 2 and abs(current_x - target_x) == 1:
            return board[(current_y + target_y) // 2][current_x] is None
        return False


class CannonStrategy(MoveStrategy):
    def movement(self, current_pos, target_pos, board, current_player):
        current_x, current_y = current_pos
        target_x, target_y = target_pos
        if current_x == target_x:
            start, end = min(current_y, target_y) + 1, max(current_y, target_y)
            count = 0
            for y in range(start, end):
                if board[y][current_x] is not None:
                    count += 1
            if board[target_y][target_x] is None:
                return count == 0
            else:
                return count == 1
        elif current_y == target_y:
            start, end = min(current_x, target_x) + 1, max(current_x, target_x)
            count = 0
            for x in range(start, end):
                if board[current_y][x] is not None:
                    count += 1
            if board[target_y][target_x] is None:
                return count == 0
            else:
                return count == 1
        return False


class PawnStrategy(MoveStrategy):
    def movement(self, current_pos, target_pos, board, current_player):
        current_x, current_y = current_pos
        target_x, target_y = target_pos
        if current_player == "r":
            return (target_x == current_x and target_y == current_y - 1) or (
                abs(target_x - current_x) == 1
                and current_y == target_y
                and current_y <= 4
            )
        else:
            return (target_x == current_x and target_y == current_y + 1) or (
                abs(target_x - current_x) == 1
                and current_y == target_y
                and current_y >= 5
            )


class General(ChessPiece):
    def __init__(self, color, piece_type, pos):
        super().__init__(color, piece_type, pos)
        self.move_strategy = GeneralStrategy()


class Advisor(ChessPiece):
    def __init__(self, color, piece_type, pos):
        super().__init__(color, piece_type, pos)
        self.move_strategy = AdvisorStrategy()


class Elephant(ChessPiece):
    def __init__(self, color, piece_type, pos):
        super().__init__(color, piece_type, pos)
        self.move_strategy = ElephantStrategy()


class Chariot(ChessPiece):
    def __init__(self, color, piece_type, pos):
        super().__init__(color, piece_type, pos)
        self.move_strategy = ChariotStrategy()


class Horse(ChessPiece):
    def __init__(self, color, piece_type, pos):
        super().__init__(color, piece_type, pos)
        self.move_strategy = HorseStrategy()


class Cannon(ChessPiece):
    def __init__(self, color, piece_type, pos):
        super().__init__(color, piece_type, pos)
        self.move_strategy = CannonStrategy()


class Pawn(ChessPiece):
    def __init__(self, color, piece_type, pos):
        super().__init__(color, piece_type, pos)
        self.move_strategy = PawnStrategy()
