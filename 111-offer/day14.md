## 题目

- 输入一个链表，输出该链表中倒数第k个结点。



## 思路

- 双指针法则
- right指针先走k-1步
- 然后left和right一起走
- 最后他们就相差k的位置/left指向的就是第k个结点



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
    ListNode* FindKthToTail(ListNode* pListHead, unsigned int k) {
        if(pListHead == NULL)
            return NULL;
        ListNode* right = pListHead;
        unsigned int i = 0;
        while(i < k-1 && right != NULL)
        {
            right = right->next;
            i++;
        }
        if(right == NULL)
            return NULL;
        ListNode* left = pListHead;
        while(right->next != NULL)
        {
            left = left->next;
            right = right->next;
        }
        return left;
    }
};
```

