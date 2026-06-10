import heapq
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

Board = Tuple[int, ...]  # 棋盘状态


# 移动方向
class Move:
    UP = 0
    DOWN = 1
    LEFT = 2
    RIGHT = 3


GOAL: Board = (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 0)  # 目标状态

# 每个数字在目标棋盘中的行和列(0-based), 用于快速计算距离
TARGET_ROW = [-1] * 16
TARGET_COL = [-1] * 16
for i, val in enumerate(GOAL):
    if val != 0:
        TARGET_ROW[val] = i // 4
        TARGET_COL[val] = i % 4


# 后继状态: 一次移动产生的完整信息
@dataclass
class Successor:
    state: Board  # 新棋盘
    blank: int  # 新空格位置
    move: int  # 产生此状态的移动方向


# 搜索节点
@dataclass
class Node:
    state: Board  # 当前棋盘
    g: int  # 已走步数
    h: int  # 启发式估计值
    blank: int  # 空格位置 (0~15)
    parent: Optional["Node"]  # 父节点
    move_from_parent: int  # 从父节点如何移动得到当前状态

    # 估价总分
    def f(self) -> int:
        return self.g + self.h

    def __lt__(self, other):
        return self.f() < other.f()


# 打印棋盘
def print_board(b: Board):
    for i in range(16):
        if b[i] == 0:
            print("   ", end="")  # 空格用空白表示
        else:
            print(f"{b[i]:2d} ", end="")
        if i % 4 == 3:
            print()
    print()


# 根据当前空格位置, 生成所有合法移动后的后继棋盘
def get_successors(state: Board, blank: int) -> List[Successor]:
    result = []
    row = blank // 4
    col = blank % 4

    # 上移
    if row > 0:
        next_state = list(state)
        target = blank - 4
        next_state[blank], next_state[target] = next_state[target], next_state[blank]
        result.append(Successor(tuple(next_state), target, Move.UP))
    # 下移
    if row < 3:
        next_state = list(state)
        target = blank + 4
        next_state[blank], next_state[target] = next_state[target], next_state[blank]
        result.append(Successor(tuple(next_state), target, Move.DOWN))
    # 左移
    if col > 0:
        next_state = list(state)
        target = blank - 1
        next_state[blank], next_state[target] = next_state[target], next_state[blank]
        result.append(Successor(tuple(next_state), target, Move.LEFT))
    # 右移
    if col < 3:
        next_state = list(state)
        target = blank + 1
        next_state[blank], next_state[target] = next_state[target], next_state[blank]
        result.append(Successor(tuple(next_state), target, Move.RIGHT))
    return result


# 移动方向转字符串
def move_to_str(m: int) -> str:
    if m == Move.UP:
        return "上"
    elif m == Move.DOWN:
        return "下"
    elif m == Move.LEFT:
        return "左"
    elif m == Move.RIGHT:
        return "右"
    return "?"


# 可解性判断
def is_solvable(state: Board) -> bool:
    # 计算逆序数(忽略空格)
    inversions = 0
    for i in range(16):
        if state[i] == 0:
            continue
        for j in range(i + 1, 16):
            if state[j] == 0:
                continue
            if state[i] > state[j]:
                inversions += 1
    # 找到空格行号
    blank_idx = state.index(0)
    blank_row = blank_idx // 4

    # 逆序数奇偶性 != 空格行号奇偶性
    return (inversions % 2) != (blank_row % 2)  # ((inversions + blank_row) % 2 == 1)


# 曼哈顿距离启发式
def manhattan(state: Board) -> int:
    dist = 0
    for i, val in enumerate(state):
        if val == 0:
            continue  # 空格不参与
        cur_row = i // 4
        cur_col = i % 4
        goal_row = TARGET_ROW[val]
        goal_col = TARGET_COL[val]
        dist += abs(cur_row - goal_row) + abs(cur_col - goal_col)
    return dist


# 查找空格位置的辅助函数
def find_blank(state: Board) -> int:
    return state.index(0)


def solve(start_state: Board) -> List[int]:
    # 可解性检查
    if not is_solvable(start_state):
        return []  # 无解返回空序列

    # 初始化优先队列和关闭列表
    open_heap = []
    closed: Dict[Board, int] = {}  # 状态 -> 最小 g 值

    start_blank = find_blank(start_state)
    start_node = Node(
        state=start_state,
        g=0,
        h=manhattan(start_state),
        blank=start_blank,
        parent=None,
        move_from_parent=Move.UP,  # move_from_parent 对于起点无意义, 可以随意设置
    )

    heapq.heappush(open_heap, start_node)

    # A* 主循环
    while open_heap:
        current = heapq.heappop(open_heap)

        # 检查目标
        if current.state == GOAL:
            # 重建路径
            path = []
            node = current
            while node.parent is not None:
                path.append(node.move_from_parent)
                node = node.parent
            path.reverse()
            return path

        # 惰性删除: 如果当前 g 值大于closed中记录的最小 g 值, 则跳过
        prev_g = closed.get(current.state)
        if prev_g is not None and prev_g < current.g:
            continue
        # 否则, 更新closed表
        closed[current.state] = current.g

        # 扩展后继
        successors = get_successors(current.state, current.blank)
        for succ in successors:
            new_g = current.g + 1
            # 检查 closed 表
            best_g = closed.get(succ.state)
            if best_g is not None and best_g <= new_g:
                continue  # 已有更优路径到达此状态
            # 创建子节点
            child = Node(
                state=succ.state,
                g=new_g,
                h=manhattan(succ.state),
                blank=succ.blank,
                parent=current,
                move_from_parent=succ.move,
            )
            heapq.heappush(open_heap, child)

    return []  # 因为已预先检查可解性, 理论上不会到这里


def simulate(start: Board, path: List[int]):
    cur = start
    blank = find_blank(cur)
    print_board(cur)
    for i, move in enumerate(path):
        print(f"Move {i+1}: {move_to_str(move)}")
        succs = get_successors(cur, blank)
        for s in succs:
            if s.move == move:
                cur = s.state
                blank = s.blank
                break
        print_board(cur)
    if cur == GOAL:
        print("成功复原!")
    else:
        print("错误: 未到达目标.")


def main():
    board = []
    print("请输入棋盘, 按行优先顺序, 空格用0表示(16个数字, 空格分隔):")
    nums = list(map(int, input().split()))
    if len(nums) != 16:
        print("错误: 需要16个数字")
        return
    input_board = tuple(nums)

    if not is_solvable(input_board):
        print("这个棋盘无解.")
        return

    print("求解中...")
    path = solve(input_board)
    if not path:
        print("求解失败.")  # 理论上不应该发生
        return

    print(f"找到解, 共 {len(path)} 步:")
    for i, m in enumerate(path):
        print(f"{i+1}: {move_to_str(m)}")

    # 模拟验证
    print("\n模拟过程:")
    simulate(input_board, path)


if __name__ == "__main__":
    main()
