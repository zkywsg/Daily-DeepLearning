## 题目

- 一只青蛙一次可以跳上1级台阶，也可以跳上2级……它也可以跳上n级。求该青蛙跳上一个n级的台阶总共有多少种跳法。



## 思路

- n=1/一种
- n=2/一次跳一个或者两个 f(2) = f(1)+f(0)
- n=3/跳出一个之后剩下f(2)/跳出两个之后剩下f(1)/跳出是哪个之后剩下f(0)
- f(n-1) = f(0) + f(1) + f(2) + f(3) + ... + f(n-2)
- f(n) = f(0) + f(1) + f(2) + f(3) + ... + f(n-2) + f(n-1)
-  f(n) = 2*f(n-1)



## 代码

```cpp
class Solution {
public:
    int jumpFloorII(int number) {
        if(number <= 0)
        {
            return -1;
        }
        else
        {
            return pow(2,number-1);
        }
    }
};
```

