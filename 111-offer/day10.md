## 题目

- 我们可以用2x1的小矩形横着或者竖着去覆盖更大的矩形。请问用n个2x1的小矩形无重叠地覆盖一个2xn的大矩形，总共有多少种方法？



## 思路

- n=0/0种
- n=1/一种
- f(n) = f(n-1) + f(n-2)



## 代码

```cpp
class Solution {
public:
    int rectCover(int number) {
        if(number <=1)
        {
            return number;
        }
        long one = 1;
        long two = 1;
        long res = 0;
        for(int i = 2; i <= number; i++)
        {
            res = one + two;
            one = two;
            two = res;
        }
        return res;
    }
};
```

