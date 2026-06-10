#include <iostream>
#include <algorithm>
#include <array>
#include <vector>
#include <memory>
#include <iomanip>
#include <string>
#include <numeric>
#include <queue>
#include <map>

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

// 可解性判断
bool is_solvable(const Board &state)
{
    // 计算逆序数(忽略空格)
    int inversions = 0;
    for (int i = 0; i < 16; ++i)
    {
        if (state[i] == 0)
            continue;
        for (int j = i + 1; j < 16; ++j)
        {
            if (state[j] == 0)
                continue;
            if (state[i] > state[j])
                ++inversions;
        }
    }
    // 找到空格行号
    int blank_idx = static_cast<int>(std::distance(
        state.begin(), std::find(state.begin(), state.end(), 0)));
    int blank_row = blank_idx / 4;

    // 逆序数奇偶性 != 空格行号奇偶性
    return (inversions % 2) != (blank_row % 2); // ((inversions + blank_row) % 2 == 1)
}

// 曼哈顿距离启发式
int manhattan(const Board &state)
{
    int dist = 0;
    for (int i = 0; i < 16; ++i)
    {
        int val = state[i];
        if (val == 0)
            continue; // 空格不参与
        int cur_row = i / 4;
        int cur_col = i % 4;
        int goal_row = TARGET_ROW[val];
        int goal_col = TARGET_COL[val];
        dist += std::abs(cur_row - goal_row) + std::abs(cur_col - goal_col);
    }
    return dist;
}

// 优先队列比较: f值小的优先(小根堆)
struct CompareNode
{
    bool operator()(const std::shared_ptr<Node> &a, const std::shared_ptr<Node> &b) const
    {
        return a->f() > b->f();
    }
};

// 查找空格位置的辅助函数
int find_blank(const Board &state)
{
    return static_cast<int>(std::distance(state.begin(), std::find(state.begin(), state.end(), 0)));
}

std::vector<Move> solve(const Board &start_state)
{
    // 可解性检查
    if (!is_solvable(start_state))
    {
        return {}; // 无解返回空序列
    }

    // 初始化优先队列和关闭列表
    std::priority_queue<std::shared_ptr<Node>,
                        std::vector<std::shared_ptr<Node>>,
                        CompareNode>
        open_set;
    std::map<Board, int> closed; // 状态 -> 最小 g 值

    int start_blank = find_blank(start_state);
    auto start_node = std::make_shared<Node>();
    start_node->state = start_state;
    start_node->g = 0;
    start_node->h = manhattan(start_state);
    start_node->blank = start_blank;
    start_node->parent = nullptr;
    // move_from_parent 对于起点无意义, 可以随意设置
    start_node->move_from_parent = Move::UP;

    open_set.push(start_node);

    // A* 主循环
    while (!open_set.empty())
    {
        auto current = open_set.top();
        open_set.pop();

        // 检查目标
        if (current->state == GOAL)
        {
            // 重建路径
            std::vector<Move> path;
            auto node = current;
            while (node->parent != nullptr)
            {
                path.push_back(node->move_from_parent);
                node = node->parent;
            }
            std::reverse(path.begin(), path.end());
            return path;
        }

        // 惰性删除: 如果当前 g 值大于closed中记录的最小 g 值, 则跳过
        auto it = closed.find(current->state);
        if (it != closed.end() && it->second < current->g)
        {
            continue;
        }
        // 否则, 更新closed表
        closed[current->state] = current->g;

        // 扩展后继
        auto successors = get_successors(current->state, current->blank);
        for (const auto &succ : successors)
        {
            int new_g = current->g + 1;
            // 检查 closed 表
            auto closed_it = closed.find(succ.state);
            if (closed_it != closed.end() && closed_it->second <= new_g)
            {
                continue; // 已有更优路径到达此状态
            }
            // 创建子节点
            auto child = std::make_shared<Node>();
            child->state = succ.state;
            child->g = new_g;
            child->h = manhattan(succ.state);
            child->blank = succ.blank;
            child->parent = current;
            child->move_from_parent = succ.move;
            open_set.push(child);
        }
    }

    return {}; // 因为已预先检查可解性, 理论上不会到这里
}

void simulate(const Board &start, const std::vector<Move> &path)
{
    Board cur = start;
    int blank = find_blank(cur);
    print_board(cur);
    for (size_t i = 0; i < path.size(); ++i)
    {
        std::cout << "Move " << i + 1 << ": " << move_to_str(path[i]) << "\n";
        auto succs = get_successors(cur, blank);
        for (auto &s : succs)
        {
            if (s.move == path[i])
            {
                cur = s.state;
                blank = s.blank;
                break;
            }
        }
        print_board(cur);
    }
    if (cur == GOAL)
    {
        std::cout << "成功复原!\n";
    }
    else
    {
        std::cout << "错误: 未到达目标.\n";
    }
}

int main()
{
    Board input;
    std::cout << "请输入棋盘, 按行优先顺序, 空格用0表示(16个数字, 空格分隔):\n";
    for (int i = 0; i < 16; ++i)
    {
        std::cin >> input[i];
    }

    if (!is_solvable(input))
    {
        std::cout << "这个棋盘无解.\n";
        return 0;
    }

    std::cout << "求解中...\n";
    auto path = solve(input);
    if (path.empty())
    {
        std::cout << "求解失败.\n"; // 理论上不应该发生
        return 1;
    }

    std::cout << "找到解, 共 " << path.size() << " 步:\n";
    for (size_t i = 0; i < path.size(); ++i)
    {
        std::cout << i + 1 << ": " << move_to_str(path[i]) << '\n';
    }

    // 模拟验证
    std::cout << "\n模拟过程:\n";
    simulate(input, path);

    return 0;
}