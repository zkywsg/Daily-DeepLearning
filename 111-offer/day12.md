## 题目

- 给定一个double类型的浮点数base和int类型的整数exponent。求base的exponent次方。保证base和exponent不同时为0



## 思路

- 巧妙思路
- 如果当前的exponent是奇数/当前的值就是res x res x base
- 如果当前的exponent是偶数/当前的值就是res x res
- 通过右移表示除2
- 通过&表示当前是奇数还是偶数



## 代码

```cpp
class Solution {
    double power(double base, int exp) {
        if (exp == 1) 
            return base;
        if ((exp & 1) == 0) 
        {
            int tmp = power(base, exp >> 1);
            return tmp * tmp;
        } 
        else 
        {
            int tmp = power(base, exp >> 1);
            return tmp * tmp * base;
        }
    }
public:
    double Power(double base, int exp) {
        if (base == 0) {
            return 0;
        } 
        else 
        {
            if (exp > 0) 
                return power(base, exp);
            else if (exp == 0) 
                return 1;
            else 
                return 1 / power(base, -exp);
        }
    }
};
```

