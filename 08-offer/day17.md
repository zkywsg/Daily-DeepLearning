## 题目

- 输入两棵二叉树A，B，判断B是不是A的子结构。（ps：我们约定空树不是任意一个树的子结构）



## 思路

- HasSubtree函数
- 要先判断是否为空 空就false
- 当ab当前的值相同的时候 进入DoesParHaveChild去判断子树是否相同
- 如果判断不是子树 就继续找a中是否有和b相同的值 进行递归



- DoesParHaveChild函数
- 子树递归到最后是NULL那么一定是子树了
- 如果a树先出现NULL,b树没有,就一定不是子树
- 然后值相等的时候递归进树

## 代码

```cpp
/*
struct TreeNode {
	int val;
	struct TreeNode *left;
	struct TreeNode *right;
	TreeNode(int x) :
			val(x), left(NULL), right(NULL) {
	}
};*/
class Solution {
public:
    bool HasSubtree(TreeNode* pRoot1, TreeNode* pRoot2)
    {
        if(pRoot1 == NULL || pRoot2 == NULL)
        {
            return false;
        }
        bool res;
        if(pRoot1->val == pRoot2->val)
        {
            res =  DoesParHaveChild(pRoot1,pRoot2);
        }
        if(res != true)
        {
            return HasSubtree(pRoot1->left,pRoot2) || HasSubtree(pRoot1->right,pRoot2);
        }
        else
        {
            return true;
        }
    }
    bool DoesParHaveChild(TreeNode* pRoot1, TreeNode* pRoot2)
    {
        if(pRoot2 == NULL)
        {
            return true;
        }
        else if(pRoot1 == NULL)
        {
            return false;
        }
        if(pRoot1->val != pRoot2->val)
        {
            return false;
        }
        else
        {
            return DoesParHaveChild(pRoot1->left,pRoot2->left) && DoesParHaveChild(pRoot1->right,pRoot2->right);
        }
    }
};
```

