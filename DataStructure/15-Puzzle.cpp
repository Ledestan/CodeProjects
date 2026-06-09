#include <iostream>
#include <algorithm>
#include <array>
#include <vector>
#include <memory>
#include <iomanip>
#include <string>
#include <numeric>

using Board = std::array<int, 16>; // 棋盘状态

// 移动方向
enum class Move
{
    UP,
    DOWN,
    LEFT,
    RIGHT
};

constexpr Board GOAL = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 0}; // 目标状态

// 每个数字在目标棋盘中的行和列(0-based), 用于快速计算距离
constexpr std::array<int, 16> TARGET_ROW = {
    -1,         // 0 不用
    0, 0, 0, 0, // 1~4
    1, 1, 1, 1, // 5~8
    2, 2, 2, 2, // 9~12
    3, 3, 3     // 13~15
};
constexpr std::array<int, 16> TARGET_COL = {
    -1,         // 0 不用
    0, 1, 2, 3, // 1~4
    0, 1, 2, 3, // 5~8
    0, 1, 2, 3, // 9~12
    0, 1, 2     // 13~15
};

// 后继状态: 一次移动产生的完整信息
struct Successor
{
    Board state; // 新棋盘
    int blank;   // 新空格位置
    Move move;   // 产生此状态的移动方向
};

// 搜索节点
struct Node
{
    Board state;                  // 当前棋盘
    int g;                        // 已走步数
    int h;                        // 启发式估计值
    int blank;                    // 空格位置 (0~15)
    std::shared_ptr<Node> parent; // 父节点
    Move move_from_parent;        // 从父节点如何移动得到当前状态

    // 估价总分
    int f() const { return g + h; }
};

// 打印棋盘
void print_board(const Board &b)
{
    for (int i = 0; i < 16; ++i)
    {
        if (b[i] == 0)
            std::cout << "   "; // 空格用空白表示
        else
            std::cout << std::setw(2) << b[i] << ' ';
        if (i % 4 == 3)
            std::cout << "\n";
    }
    std::cout << "\n";
}

// 根据当前空格位置, 生成所有合法移动后的后继棋盘
std::vector<Successor> get_successors(const Board &state, int blank)
{
    std::vector<Successor> result;
    int row = blank / 4;
    int col = blank % 4;

    // 上移
    if (row > 0)
    {
        Board next = state;
        int target = blank - 4;
        std::swap(next[blank], next[target]);
        result.push_back({next, target, Move::UP});
    }
    // 下移
    if (row < 3)
    {
        Board next = state;
        int target = blank + 4;
        std::swap(next[blank], next[target]);
        result.push_back({next, target, Move::DOWN});
    }
    // 左移
    if (col > 0)
    {
        Board next = state;
        int target = blank - 1;
        std::swap(next[blank], next[target]);
        result.push_back({next, target, Move::LEFT});
    }
    // 右移
    if (col < 3)
    {
        Board next = state;
        int target = blank + 1;
        std::swap(next[blank], next[target]);
        result.push_back({next, target, Move::RIGHT});
    }
    return result;
}

// 移动方向转字符串
std::string move_to_str(Move m)
{
    switch (m)
    {
    case Move::UP:
        return "上";
    case Move::DOWN:
        return "下";
    case Move::LEFT:
        return "左";
    case Move::RIGHT:
        return "右";
    }
    return "?";
}

bool is_solvable(const Board& state) {
    // 计算逆序数(忽略空格)
    int inversions = 0;
    for (int i = 0; i < 16; ++i) {
        if (state[i] == 0) continue;
        for (int j = i + 1; j < 16; ++j) {
            if (state[j] == 0) continue;
            if (state[i] > state[j]) ++inversions;
        }
    }
    // 找到空格行号
    int blank_idx = static_cast<int>(std::distance(
        state.begin(), std::find(state.begin(), state.end(), 0)));
    int blank_row = blank_idx / 4;

    // 4x4（偶宽度）：逆序数奇偶性 ≠ 空格行号奇偶性
    return (inversions % 2) != (blank_row % 2);
}

int main()
{
    std::cout << "目标棋盘：\n";
    print_board(GOAL);

    int blank_pos = 15;
    auto succs = get_successors(GOAL, blank_pos);
    std::cout << "空格在 " << blank_pos << "，共有 " << succs.size() << " 个后继：\n\n";
    for (const auto &s : succs)
    {
        std::cout << "移动方向: " << move_to_str(s.move)
                  << "，新空格位置: " << s.blank << "\n";
        print_board(s.state);
    }
    return 0;
}