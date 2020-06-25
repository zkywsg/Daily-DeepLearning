## 题目

- 一只青蛙一次可以跳上1级台阶，也可以跳上2级。求该青蛙跳上一个n级的台阶总共有多少种跳法（先后次序不同算不同的结果）。



## 思路

- 先跳1级/剩下的是跳N-1个台阶
- 先跳2级/剩下的是跳N-2个台阶
- N=0/0种
- N=1/1种
- N=2/2中
- N=n/f(n-1)+f(n-2)



## 代码

```cpp
class Solution {
public:
    int jumpFloor(int number) {
        if(number <=2)
        {
            return number;
        }
        long one = 1;
        long two = 2;
        long res = 0;
        for(int i = 3; i <= number; i++)
        {
            res = one + two;
            one = two;
            two = res;
        }
        return res;
    }
};
```

