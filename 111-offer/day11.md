## 题目

- 输入一个整数，输出该数二进制表示中1的个数。其中负数用补码表示。



## 思路1

- 从右向左去判断每一位的数字是否是1
- flag设置为1
- n&flag代表去判断了最低位开始/是否有1
- 然后把flag左移一位



## 代码1

```cpp
class Solution {
public:
     int  NumberOf1(int n) {
         int count = 0;
         unsigned int flag = 1;
         while(flag)
         {
             if(n&flag)
             {
                 count++;
             }
             flag <<=1;
         }
         return count;
     }
};
```



## 思路2

- 当前假设是1100/数值上-1的话就是1011
- 1100&1011 = 1000
- 实际上这是一个特殊的规律/利用这个规律/每次去记录count/然后把这个1去掉



## 代码2

```cpp
class Solution {
public:
     int  NumberOf1(int n) {
         int count = 0;
         while(n)
         {
             ++ count;
             n = (n-1) & n;
         }
         return count;
     }
};
```

