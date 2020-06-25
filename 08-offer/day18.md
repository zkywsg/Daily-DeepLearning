## 题目

- 操作给定的二叉树，将其变换为源二叉树的镜像。



## 思路

- 递归方式太简单
- 非递归思路 以中序遍历为例
- 遍历的过程用栈代替了递归
- 走到最左边之后
- 拿出当前栈最上边的点和右边的进行交换
- 原本就是应该弹出该点这一层之后往右走
- 应为左右进行了交换所以继续往左走



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
    void Mirror(TreeNode *pRoot) {
        if(pRoot == NULL)
        {
            return;
        }
        swap(pRoot->left,pRoot->right);
        Mirror(pRoot->left);
        Mirror(pRoot->right);
    }
};
```

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
    void Mirror(TreeNode *pRoot) {
        if(pRoot == NULL)
        {
            return;
        }
        stack<TreeNode *> nstack;
        TreeNode *node = pRoot;
        while(node != NULL || nstack.empty() != true)
        {
            while(node != NULL)
            {
                nstack.push(node);
                node = node->left;
            }
            if(nstack.empty() != true)
            {
                node = nstack.top();
                if(node->left != NULL || node->right != NULL)
                {
                    swap(node->left,node->right);
                }
                nstack.pop();
                node = node->left;
            }
        }
    }
};
```

