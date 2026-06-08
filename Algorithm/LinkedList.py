class Node:
    """链表节点类"""

    def __init__(self, data):
        self.data = data
        self.next = None


class LinkedList:
    """单链表类, 支持常见序列操作"""

    # ---------- 构造函数及核心容器协议 ----------
    def __init__(self, iterable=None):
        """支持从可迭代对象创建链表"""
        self.head = None
        if iterable is not None:
            # 逐个追加元素，保持原顺序
            for item in iterable:
                self.append(item)

    def __len__(self):
        """返回链表长度, 支持 len()"""
        count = 0
        current = self.head
        while current:
            count += 1
            current = current.next
        return count

    def __iter__(self):
        """迭代器, 支持 for ... in"""
        current = self.head
        while current:
            yield current.data
            current = current.next

    def __contains__(self, value):
        """支持 in 操作符"""
        current = self.head
        while current:
            if current.data == value:
                return True
            current = current.next
        return False

    # ---------- 序列协议：索引访问与修改 ----------
    def __getitem__(self, idx):
        """支持正向索引读取, 负数索引抛出异常"""
        if idx < 0:
            raise IndexError("Negative index not supported")
        current = self.head
        # 移动指针到目标位置
        for _ in range(idx):
            if current is None:
                raise IndexError("Index out of range")
            current = current.next
        if current is None:
            raise IndexError("Index out of range")
        return current.data

    def __setitem__(self, index, value):
        """支持索引赋值修改节点值"""
        if index < 0:
            raise IndexError("Negative index not supported")
        current = self.head
        for _ in range(index):
            if current is None:
                raise IndexError("Index out of range")
            current = current.next
        if current is None:
            raise IndexError("Index out of range")
        current.data = value

    def __delitem__(self, index):
        """支持 del 删除指定位置节点"""
        if index < 0:
            raise IndexError("Negative index not supported")
        length = len(self)
        if index >= length:
            raise IndexError("Index out of range")

        if index == 0:
            # 删除头节点：将 head 指向下一个节点
            self.head = self.head.next
            return

        # 找到待删除节点的前驱
        prev = self.head
        for _ in range(index - 1):
            prev = prev.next
        # 跳过被删除节点
        prev.next = prev.next.next

    # ---------- 运算符重载 ----------
    def __add__(self, other):
        """支持链表相加, 返回新链表"""
        if not isinstance(other, LinkedList):
            return NotImplemented
        # 复制当前链表
        new_list = LinkedList(self)
        # 追加另一个链表的所有元素
        for item in other:
            new_list.append(item)
        return new_list

    def __eq__(self, other):
        """支持 == 比较两个链表是否相等"""
        if not isinstance(other, LinkedList):
            return False
        if len(self) != len(other):
            return False
        # 逐元素比较
        for a, b in zip(self, other):
            if a != b:
                return False
        return True

    def __bool__(self):
        """支持布尔测试, 空链表为 False"""
        return self.head is not None

    def __str__(self):
        """字符串表示, 用于 print"""
        result = []
        current = self.head
        while current:
            result.append(str(current.data))
            current = current.next
        return " -> ".join(result)

    # ---------- 链表特有方法 ----------
    def clear(self):
        """清空链表"""
        self.head = None

    def append(self, data):
        """在尾部添加节点"""
        new_node = Node(data)
        if not self.head:
            self.head = new_node
            return
        # 找到尾节点
        last_node = self.head
        while last_node.next:
            last_node = last_node.next
        last_node.next = new_node

    def insert(self, index, data):
        """在指定位置(0 based)插入节点, 位置范围 [0, len]"""
        if index < 0 or index > len(self):
            raise IndexError("index out of range")

        new_node = Node(data)

        if index == 0:
            # 头插法：新节点指向原头节点，更新 head
            new_node.next = self.head
            self.head = new_node
            return

        # 找到待插入位置的前驱节点
        prev = self.head
        for _ in range(index - 1):
            prev = prev.next
        # 将新节点插入到 prev 之后
        new_node.next = prev.next
        prev.next = new_node

    def pop(self, index=-1):
        """删除并返回指定位置的节点(默认末尾), 负数索引从末尾计数"""
        length = len(self)
        if length == 0:
            raise IndexError("pop from empty list")

        # 负数索引转换为正数
        if index < 0:
            index += length
        if index < 0 or index >= length:
            raise IndexError("pop index out of range")

        if index == 0:
            # 弹出头节点
            data = self.head.data
            self.head = self.head.next
            return data

        # 找到待弹出节点的前驱
        prev = self.head
        for _ in range(index - 1):
            prev = prev.next
        data = prev.next.data
        # 绕过被弹出节点
        prev.next = prev.next.next
        return data

    def remove(self, value):
        """删除链表中第一个值为 value 的节点, 若不存在则引发 ValueError"""
        if self.head is None:
            raise ValueError("value not found")

        # 头节点就是要删除的节点
        if self.head.data == value:
            self.head = self.head.next
            return

        # 从第二个节点开始查找
        prev = self.head
        while prev.next:
            if prev.next.data == value:
                prev.next = prev.next.next
                return
            prev = prev.next
        raise ValueError("value not found")

    def index(self, value, start=0, stop=None):
        """返回第一个值为 value 的索引, 支持搜索范围 [start, stop)"""
        length = len(self)

        # 处理负数索引，转换为正数
        if start < 0:
            start += length
        if stop is None:
            stop = length
        elif stop < 0:
            stop += length

        # 边界裁剪到合法范围
        start = max(start, 0)
        stop = min(stop, length)

        # 无效范围或 start 已超出链表长度, 值不可能存在
        if start >= stop or start >= length:
            raise ValueError("value not found")

        # 定位到 start 位置
        current = self.head
        pos = 0
        while pos < start and current:
            current = current.next
            pos += 1

        # 从 start 开始查找，直到 stop 或链表结束
        while pos < stop and current:
            if current.data == value:
                return pos
            current = current.next
            pos += 1

        raise ValueError("value not found")

    def count(self, value):
        """统计 value 出现的次数"""
        cnt = 0
        current = self.head
        while current:
            if current.data == value:
                cnt += 1
            current = current.next
        return cnt

    def sort(self):
        """对链表进行排序(不保留原顺序), 使用列表排序后重建"""
        if not self.head:
            return
        # 收集所有节点值到列表并排序
        values = list(self)
        values.sort()
        # 根据排序后的值重建链表
        self.head = None
        tail = None
        for val in values:
            new_node = Node(val)
            if self.head is None:
                self.head = new_node
                tail = new_node
            else:
                tail.next = new_node
                tail = new_node


if __name__ == "__main__":
    ll = LinkedList()
