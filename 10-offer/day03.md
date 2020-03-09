## 题目

- 输入一个链表，按链表从尾到头的顺序返回一个ArrayList。



## 思路

- 要实现反向输出链表
- 可以使用栈的特性把输出结构反转过来
- 递归的本质就是一个栈



## 算法流程

- 判断当前是否为NULL/不是的时候就递归
- 递归结束后/所有的元素就按照逆序进入了vector中



## 代码

```cpp
/**
*  struct ListNode {
*        int val;
*        struct ListNode *next;
*        ListNode(int x) :
*              val(x), next(NULL) {
*        }
*  };
*/
class Solution {
public:
    vector<int> res;
    vector<int> printListFromTailToHead(ListNode* head) {
        if(head != NULL)
        {
            printListFromTailToHead(head->next);
            res.push_back(head->val);
        }
        return res;
    }
};
```



