## 题目

- 输入一个链表，反转链表后，输出新链表的表头。



## 思路

- 每次插入到新链表的表头
- n和p先指向第一个结点进行特殊处理
- p指向第二个结点
- n的next指向null
- 进入循环
- p存在的时候/就把他下一个位置让q指向
- 然后把p指向的结点放到n的头上
- p在指向q的结点一直到p不存在为止



## 代码

```cpp
/*
struct ListNode {
	int val;
	struct ListNode *next;
	ListNode(int x) :
			val(x), next(NULL) {
	}
};*/
class Solution {
public:
    ListNode* ReverseList(ListNode* pHead) {
        if(pHead == NULL)
            return NULL;
        ListNode* p = pHead;
        ListNode* q = NULL;
        ListNode* newhead = p;
        p = p->next;
        newhead->next = NULL;
        while(p != NULL)
        {
            q = p->next;
            p ->next = newhead;
            newhead = p;
            p = q;
        }
        return newhead;
    }
};
```

