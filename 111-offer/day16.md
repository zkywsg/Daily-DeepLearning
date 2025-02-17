## 题目

- 输入两个单调递增的链表，输出两个链表合成后的链表，当然我们需要合成后的链表满足单调不减规则。



## 思路

- 左边空的时候返回右边
- 右边空的时候返回左边
- 两个指针指向各自头结点
- 左边比右边小就把左边添加到新链表,指针后移
- 右边比左边小就把右边添加到新链表,指针后移
- 然后设立指针开始遍历
- 对最后剩余的部分进行直接拼接处理



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
    ListNode* Merge(ListNode* pHead1, ListNode* pHead2)
    {
        if(pHead1 == NULL)
            return pHead2;
        if(pHead2 == NULL)
            return pHead1;
        ListNode* head = NULL;
        if(pHead1->val < pHead2->val)
        {
            head = pHead1;
            pHead1 = pHead1->next;
        }
        else
        {
            head = pHead2;
            pHead2 = pHead2->next;
        }
        ListNode* curr = head;
        while(pHead1 != NULL && pHead2 != NULL)
        {
            if(pHead1->val < pHead2->val)
            {
                curr->next = pHead1;
                pHead1 = pHead1->next;
                curr = curr->next;
            }
            else
            {
                curr->next = pHead2;
                pHead2 = pHead2->next;
                curr = curr->next;
            }
        }
        if(pHead1 != NULL)
            curr->next = pHead1;
        if(pHead2 != NULL)
            curr->next = pHead2;
        return head;
    }
};
```

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
    ListNode* Merge(ListNode* pHead1, ListNode* pHead2)
    {
        if(pHead1 == NULL)
            return pHead2;
        if(pHead2 == NULL)
            return pHead1;
        ListNode* head;
        if(pHead1->val < pHead2->val)
        {
            head = pHead1;
            head->next = Merge(pHead1->next,pHead2);
        }
        else
        {
            head = pHead2;
            head->next = Merge(pHead1,pHead2->next);
        }
        return head;
    }
};
```

