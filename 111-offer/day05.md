## 题目

- 用两个栈来实现一个队列，完成队列的Push和Pop操作。 队列中的元素为int类型。



## 思路

- 栈的特点是后进先出
- 队列的特点是先进先出
- 把数据压进第一个栈里/出来的数据是反向的队列顺序
- 当把第一个栈的数据/压进到第二个栈的话/输出的数据则就是正向的队列顺序了
- 所以push操作就是正常进行
- pop操作需要把数据压到第二个栈中/再进行输出
- 注意到/第一个栈只有一个数字的时候是不用压到第二个栈中的/因为他就没有顺序



## 代码

```cpp
class Solution
{
public:
    // push就是正常操作就可以了
    void push(int node) {
        stack1.push(node);
    }

    int pop() {
        // node初始的时候是-1/因为刚开始没有值
        int node = -1;
        // 如果没有数据 就返回-1
        if(stack1.empty()==true && stack2.empty()==true)
            return -1;
        else 
        {
            if(stack2.empty()==true)
            {
                // 从栈1拿出数据/压到栈2
                while(stack1.empty() != true)
                {
                    node = stack1.top();
                    stack1.pop();
                    stack2.push(node);
                }
            }
        }
        // 弹出顶上的数据
        node = stack2.top();
        stack2.pop();
        return node;
    }

private:
    stack<int> stack1;
    stack<int> stack2;
};
```

