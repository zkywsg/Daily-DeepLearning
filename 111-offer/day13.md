## 题目

- 输入一个整数数组，实现一个函数来调整该数组中数字的顺序，使得所有的奇数位于数组的前半部分，所有的偶数位于数组的后半部分，并保证奇数和奇数，偶数和偶数之间的相对位置不变。



## 思路

- 开辟一个临时数组
- 把选出来的偶数放过去
- 遍历结束再放回来



## 代码

```cpp
class Solution {
public:
    void reOrderArray(vector<int> &array) {
        if(array.size() <= 1)
            return ;
        vector<int> array_temp;
        vector<int>::iterator ib1 = array.begin();
        
        while(ib1 != array.end())
        {
            if((*ib1 & 1) == 0)
            {
                array_temp.push_back(*ib1);
                ib1 = array.erase(ib1);
            }
            else
            {
                ib1++;
            }
        }
        vector<int>::iterator ib2 = array_temp.begin();
        while(ib2 != array_temp.end())
        {
            array.push_back(*ib2);
            ib2++;
        }
    }
};
```

