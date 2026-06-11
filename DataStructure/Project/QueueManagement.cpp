#include <iostream>
#include <queue>
#include <string>
#include <vector>

using namespace std;

class Ticket
{
public:
    int number;
    bool isVip;
};

class CompareTicket
{
public:
    bool operator()(const Ticket &a, const Ticket &b) const
    {
        if (a.isVip && !b.isVip) // A 是 VIP, B 不是
            return false;
        else if (!a.isVip && b.isVip) // A 不是 VIP, B 是
            return true;
        return a.number > b.number; // 号小优先
    }
};

priority_queue<Ticket, vector<Ticket>, CompareTicket> pq;

int number = 0;

int main()
{
    int choice;
    while (true)
    {
        cout << "\n银行取号系统" << endl;
        cout << "取普通号请输入 1" << endl;
        cout << "取 VIP 号请输入 2" << endl;
        cout << "叫号请输入 3" << endl;
        cout << "退出请输入 4" << endl;
        cout << "请输入选项: ";
        cin >> choice;

        switch (choice)
        {
        case 1:
        {
            number++;
            Ticket ticket;
            ticket.number = number;
            ticket.isVip = false;
            pq.push(ticket);
            cout << "您已取普通号, 号码: " << number << endl;
            break;
        }
        case 2:
        {
            number++;
            Ticket ticket;
            ticket.number = number;
            ticket.isVip = true;
            pq.push(ticket);
            cout << "您已取 VIP 号, 号码: " << number << endl;
            break;
        }
        case 3:
        {
            if (pq.empty())
                cout << "目前没有正在等待的客户." << endl;
            else
            {
                Ticket ticket = pq.top();
                cout << "请 " << ticket.number << " 号 (" << (ticket.isVip ? "VIP" : "普通") << ") 到窗口办理" << endl;
                pq.pop();
            }
            break;
        }
        case 4:
            cout << "感谢使用, 再见!";
            return 0;

        default:
            cout << "无效输入, 请重新输入." << endl;
        }
    }
    return 0;
}