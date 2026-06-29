/*
项目名称: 校园导航
创建日期: 2026-06-12
需求文件: CampusMap.csv
*/

#include <iostream>
#include <vector>
#include <string>
#include <unordered_map>
#include <utility>
#include <sstream>
#include <fstream>
#include <queue>
#include <limits>
#include <algorithm>

using namespace std;

class Edge
{
public:
    int to;     // 目标节点索引
    int weight; // 距离, 单位米
};

class Graph
{
private:
    vector<vector<Edge>> adj;            // 邻接表
    unordered_map<string, int> nameToId; // 节点名称 -> 索引
    vector<string> idToName;             // 索引 -> 节点名称
    vector<int> popularity;              // 每个节点的热度值

public:
    // 添加节点, 返回索引
    int addNode(const string &name)
    {
        if (nameToId.count(name))
            return nameToId[name];
        int id = idToName.size();
        nameToId[name] = id;
        idToName.push_back(name);
        adj.push_back({});       // 为新节点开辟邻接表
        popularity.push_back(0); // 初始化热度为 0
        return id;
    }

    // 添加无向边(双向权重相同)
    void addEdge(const string &uName, const string &vName, int weight)
    {
        int u = addNode(uName);
        int v = addNode(vName);
        adj[u].push_back({v, weight});
        adj[v].push_back({u, weight});
    }

    // 获取节点数量
    int getNodeCount() const { return idToName.size(); }

    // 根据名称获取索引(用于算法调用)
    int getIndex(const string &name) const
    {
        auto it = nameToId.find(name);
        if (it != nameToId.end())
            return it->second;
        return -1;
    }

    // 根据索引获取名称
    string getName(int id) const { return idToName[id]; }

    // 获取某个节点的邻接边
    const vector<Edge> &getNeighbors(int id) const { return adj[id]; }

    // 增加某个节点的热度
    void increasePopularity(int id, int delta = 1)
    {
        if (id >= 0 && id < (int)popularity.size())
            popularity[id] += delta;
    }

    // 获取某个节点的热度
    int getPopularity(int id) const
    {
        if (id >= 0 && id < (int)popularity.size())
            return popularity[id];
        return 0;
    }

    // 获取所有节点的热度
    const vector<int> &getAllPopularity() const { return popularity; }
};

// 读取 csv 文件, 初始化地图
void initCampusMap(Graph &g, const string &filename)
{
    ifstream file(filename);
    if (!file.is_open())
    {
        cerr << "Error: Cannot open file " << filename << endl;
        return;
    }

    string line;
    bool firstLine = true; // 跳过表头

    while (getline(file, line))
    {
        if (line.empty())
            continue;

        // 跳过表头
        if (firstLine)
        {
            firstLine = false;
            continue;
        }

        stringstream ss(line);
        string u, v, wStr;

        // 读取三个字段, 以逗号分隔
        if (getline(ss, u, ',') &&
            getline(ss, v, ',') &&
            getline(ss, wStr, ','))
        {
            int weight = stoi(wStr);
            g.addEdge(u, v, weight);
        }
    }
    file.close();
}

void printGraph(const Graph &g)
{
    int n = g.getNodeCount();
    for (int i = 0; i < n; ++i)
    {
        cout << g.getName(i) << ":";
        for (const auto &edge : g.getNeighbors(i))
        {
            cout << " ->" << g.getName(edge.to) << "(" << edge.weight << "m)";
        }
        cout << endl;
    }
}

const int INF = numeric_limits<int>::max();

/*
 * 执行 Dijkstra 算法
 * @param graph     图对象
 * @param startName 起点名称
 * @param dist      输出数组: dist[i] = 起点到节点 i 的最短距离
 * @param prev      输出数组:prev[i] = 最短路径中节点 i 的前驱节点(-1 表示无前驱)
 * @return          是否成功(起点是否存在)
 */
bool dijkstra(const Graph &graph, const string &startName,
              vector<int> &dist, vector<int> &prev)
{
    int startId = graph.getIndex(startName);
    if (startId == -1)
    {
        cerr << "错误: 起点 '" << startName << "' 不存在!" << endl;
        return false;
    }

    int n = graph.getNodeCount();
    dist.assign(n, INF);
    prev.assign(n, -1);
    vector<bool> visited(n, false);

    dist[startId] = 0;
    // 最小堆: pair<距离, 节点ID>
    using P = pair<int, int>;
    priority_queue<P, vector<P>, greater<P>> pq;
    pq.push({0, startId});

    while (!pq.empty())
    {
        int d = pq.top().first;
        int u = pq.top().second;
        pq.pop();

        if (visited[u])
            continue;
        visited[u] = true;

        // 遍历邻接边
        for (const Edge &edge : graph.getNeighbors(u))
        {
            int v = edge.to;
            int w = edge.weight;
            if (!visited[v] && dist[u] + w < dist[v])
            {
                dist[v] = dist[u] + w;
                prev[v] = u;
                pq.push({dist[v], v});
            }
        }
    }
    return true;
}

/*
 * 根据 prev 数组回溯最短路径
 * @param prev    前驱数组
 * @param startId 起点ID
 * @param endId   终点ID
 * @return        路径上的节点ID序列(包含起点和终点)
 */
vector<int> getPath(const vector<int> &prev, int startId, int endId)
{
    vector<int> path;
    if (startId == endId)
    {
        path.push_back(startId);
        return path;
    }
    int cur = endId;
    while (cur != -1)
    {
        path.push_back(cur);
        if (cur == startId)
            break;
        cur = prev[cur];
    }
    if (path.back() != startId)
    {
        // 无法回溯到起点, 说明不连通
        path.clear();
        return path;
    }
    reverse(path.begin(), path.end());
    return path;
}

// 并查集
class DSU
{
private:
    vector<int> parent, rank;

public:
    DSU(int n)
    {
        parent.resize(n);
        rank.resize(n, 0);
        for (int i = 0; i < n; ++i)
            parent[i] = i;
    }

    int find(int x)
    {
        if (parent[x] != x)
            parent[x] = find(parent[x]); // 路径压缩
        return parent[x];
    }

    bool unite(int x, int y)
    {
        int rx = find(x), ry = find(y);
        if (rx == ry)
            return false;
        if (rank[rx] < rank[ry])
            parent[rx] = ry;
        else if (rank[rx] > rank[ry])
            parent[ry] = rx;
        else
        {
            parent[ry] = rx;
            rank[rx]++;
        }
        return true;
    }
};

// Kruskal 边
class KruskalEdge
{
public:
    int u, v, weight;
};

/*
 * 执行 Kruskal 算法, 计算最小生成树
 * @param graph    图对象
 * @param mstEdges 输出参数: 存放选中的边 (u, v, weight)
 * @return         最小生成树的总权重, 若不连通返回 -1
 */
int kruskal(const Graph &graph, vector<tuple<string, string, int>> &mstEdges)
{
    int n = graph.getNodeCount();
    if (n == 0)
        return 0;

    // 收集所有边(去重, 无向图只保留一条)
    vector<KruskalEdge> edges;
    for (int u = 0; u < n; ++u)
    {
        for (const Edge &e : graph.getNeighbors(u))
        {
            int v = e.to;
            int w = e.weight;
            if (u < v)
            { // 去重
                edges.push_back({u, v, w});
            }
        }
    }

    // 按权重升序排序
    sort(edges.begin(), edges.end(),
         [](const KruskalEdge &a, const KruskalEdge &b)
         {
             return a.weight < b.weight;
         });

    // 初始化并查集
    DSU dsu(n);

    // 贪心选择边
    int totalWeight = 0;
    int edgeCount = 0;
    mstEdges.clear();

    for (const auto &e : edges)
    {
        if (dsu.unite(e.u, e.v))
        {
            totalWeight += e.weight;
            mstEdges.push_back({graph.getName(e.u), graph.getName(e.v), e.weight});
            edgeCount++;
            if (edgeCount == n - 1)
                break; // 已生成完整树
        }
    }

    // 检查连通性
    if (edgeCount != n - 1)
    {
        return -1; // 图不连通, 无法生成最小生成树
    }
    return totalWeight;
}

void showPopularityRanking(const Graph &graph)
{
    int n = graph.getNodeCount();
    if (n == 0)
    {
        cout << "暂无地点数据." << endl;
        return;
    }

    // 收集 (热度, 节点ID, 名称)
    vector<tuple<int, int, string>> items;
    for (int i = 0; i < n; ++i)
    {
        items.emplace_back(graph.getPopularity(i), i, graph.getName(i));
    }

    // 排序: 热度降序, 热度相同时按名称升序
    sort(items.begin(), items.end(),
         [](const tuple<int, int, string> &a, const tuple<int, int, string> &b)
         {
             if (get<0>(a) != get<0>(b))
                 return get<0>(a) > get<0>(b);
             return get<2>(a) < get<2>(b);
         });

    cout << "地点热度排行榜" << endl;
    cout << "排名\t热度\t地点名称" << endl;
    int rank = 1;
    for (const auto &item : items)
    {
        int pop = get<0>(item);
        string name = get<2>(item);
        cout << rank++ << "\t" << pop << "\t" << name << endl;
    }
}

void showLocationDetail(const Graph &g, const string &name)
{
    int id = g.getIndex(name); // 哈希查找
    if (id == -1)
    {
        cout << "错误: 地点 '" << name << "' 不存在!" << endl;
        return;
    }

    cout << "\n地点: " << g.getName(id) << endl;
    cout << "热度: " << g.getPopularity(id) << endl;

    cout << "相邻地点：" << endl;
    const auto &neighbors = g.getNeighbors(id);
    if (neighbors.empty())
    {
        cout << "  (无连接道路)" << endl;
    }
    else
    {
        for (const Edge &e : neighbors)
        {
            cout << "  - " << g.getName(e.to) << " (" << e.weight << "米)" << endl;
        }
    }
}

int main()
{
    Graph campus;
    initCampusMap(campus, "data/CampusMap.csv");
    cout << "节点数: " << campus.getNodeCount() << endl;
    printGraph(campus);

    int choice;
    while (true)
    {
        cout << "\n校园导航系统" << endl;
        cout << "1: 最短路径导航" << endl;
        cout << "2: 最小生成树道路规划" << endl;
        cout << "3: 查看地点热度排行榜" << endl;
        cout << "4: 查询地点详细信息" << endl;
        cout << "0: 退出" << endl;
        cout << "请选择操作: ";
        cin >> choice;

        if (choice == 1)
        {
            string start, end;
            cout << "请输入起点名称: ";
            cin >> start;
            cout << "请输入终点名称: ";
            cin >> end;

            vector<int> dist, prev;
            if (!dijkstra(campus, start, dist, prev))
            {
                continue;
            }
            int startId = campus.getIndex(start);
            int endId = campus.getIndex(end);
            if (endId == -1)
            {
                cerr << "错误: 终点 '" << end << "' 不存在!" << endl;
                continue;
            }
            vector<int> path = getPath(prev, startId, endId);
            if (path.empty())
            {
                cout << "从 " << start << " 到 " << end << " 没有可达路径!" << endl;
            }
            else
            {
                cout << "\n最短路径(总距离 = " << dist[endId] << " 米):" << endl;
                for (size_t i = 0; i < path.size(); ++i)
                {
                    cout << campus.getName(path[i]);
                    if (i != path.size() - 1)
                        cout << " -> ";
                }
                campus.increasePopularity(endId);
                cout << " (已为终点增加1点热度)";
                cout << endl;
            }
        }
        else if (choice == 2)
        {
            vector<tuple<string, string, int>> mstEdges;
            int total = kruskal(campus, mstEdges);
            if (total == -1)
            {
                cout << "校园图不连通, 无法生成最小生成树! 请检查地图数据." << endl;
            }
            else
            {
                cout << "\n最小生成树总长度 = " << total << " 米" << endl;
                cout << "建议修建/优化的道路如下:" << endl;
                for (const auto &edge : mstEdges)
                {
                    cout << "  " << get<0>(edge) << " —— " << get<1>(edge)
                         << " : " << get<2>(edge) << " 米" << endl;
                }
            }
        }
        else if (choice == 3)
        {
            showPopularityRanking(campus);
        }
        else if (choice == 4)
        {
            string name;
            cout << "请输入地点名称：";
            cin >> name;
            showLocationDetail(campus, name);
        }
        else if (choice == 0)
        {
            cout << "感谢使用, 再见!" << endl;
            return 0;
        }
        else
        {
            cout << "无效选择, 请重新输入." << endl;
        }
    }

    return 0;
}