## 题目

- 大家都知道斐波那契数列，现在要求输入一个整数n，请你输出斐波那契数列的第n项（从0开始，第0项为0）。



## 思路

- 斐波那契数列用循环的方式比迭代要节省空间
- 当n = 0/结果也是0
- 当n = 1/结果也是1
- 接下来就是fib(n-1)+fib(n-2)
- 写成循环的形式即可



## 代码

```cpp
class Solution {
public:
    int Fibonacci(int n) {
        int res[2] = {0,1};
        if(n<2)
            return res[n];
        
        long fibone = 0;
        long fibtwo = 1;
        long fibN = 0;
        for(int i = 2;i<=n;i++)
        {
            fibN = fibone + fibtwo;
            fibone = fibtwo;
            fibtwo = fibN;
        }
        return fibN;
    }
};
```

