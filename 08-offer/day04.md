## 题目

- 输入某二叉树的前序遍历和中序遍历的结果，请重建出该二叉树。假设输入的前序遍历和中序遍历的结果中都不含重复的数字。例如输入前序遍历序列{1,2,4,7,3,5,6,8}和中序遍历序列{4,7,2,1,5,3,8,6}，则重建二叉树并返回。



## 思路

- 前序遍历:根左右
- 中序遍历:左根右
- 通过递归的思想从小到大的去构建
- 我们知道一开始就可以通过前序遍历找到第一个确定的根的位置
- 然后看中序遍历的位置中,根的位置,他的左边就是左子树,右边就是右子树
- 然后对左右子树进行递归进行上述的两步



## 代码

```cpp
/**
 * Definition for binary tree
 * struct TreeNode {
 *     int val;
 *     TreeNode *left;
 *     TreeNode *right;
 *     TreeNode(int x) : val(x), left(NULL), right(NULL) {}
 * };
 */
class Solution {
public:
    TreeNode* reConstructBinaryTree(vector<int> pre,vector<int> vin) {
        // 必须是前序和中序长度相等
        if(pre.size() != vin.size())
            return NULL;
        // 记录下来长度
        int size = pre.size();
        //长度不可以为0
        if(size == 0)    return NULL;
        // 找到当前的根结点
        int value = pre[0];
        TreeNode *root = new TreeNode(value);
        
        //去索引 找中序遍历中的根结点在那里
        int rootIndex = 0;
        for(rootIndex = 0; rootIndex < size; rootIndex++)
        {
            if(vin[rootIndex] == value)
                break;
        }
        if(rootIndex >= size)
            return NULL;
        
        // 定义左右子树的长度
        int leftlength = rootIndex;
        int rightlength = size - 1 - rootIndex;
        vector<int> preLeft(leftlength), vinLeft(leftlength);
        vector<int> preRight(rightlength), vinRight(rightlength);
        // 把左右子树填写好
        for(int i = 0; i < size; i++)
        {
            if(i < rootIndex)
            {
                preLeft[i] = pre[i+1];
                vinLeft[i] = vin[i];
            }
            else if(i > rootIndex)
            {
                preRight[i - rootIndex - 1] = pre[i];
                vinRight[i - rootIndex - 1] = vin[i];
            }
        }
        // 递归进去
        root->left = reConstructBinaryTree(preLeft,vinLeft);
        root->right = reConstructBinaryTree(preRight,vinRight);
        // 一层一层出来之后 得到了整个完整的树
        return root;
    }
};
```

