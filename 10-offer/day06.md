## 题目

- 把一个数组最开始的若干个元素搬到数组的末尾，我们称之为数组的旋转。
- 输入一个非递减排序的数组的一个旋转，输出旋转数组的最小元素。
- 例如数组{3,4,5,1,2}为{1,2,3,4,5}的一个旋转，该数组的最小值为1。
- NOTE：给出的所有元素都大于0，若数组大小为0，请返回0。



## 思路

- 简单粗暴的方法就是一次遍历过去就完事了
- 想要时间复杂度更理想的话/就要进行二分查找
- 一开始第一个指针指向首部/第二个指针指向尾部
- 当中间位置的元素>第一个指针/那么这个位置还在前半部分/第一个指针指向他
- 当中间位置的元素<第二个指针/那么这个位置已在后半部分/第二个指针指向他
- 第一个指针始终在前半部分
- 第二个指针始终在后半部分
- 所以最终指向最小的数字的是第二个指针指向的元素
- 特殊情况就是/当第一个指针=第二个指针=中间元素的时候
- 再暴力去解呗



## 代码

```cpp
class Solution {
public:
    int minNumberInRotateArray(vector<int> rotateArray) {
        // 根据题意返回0
        if(rotateArray.size() == 0)
            return 0;
        int mid = 0;
        int low = 0,high = rotateArray.size()-1;
        // 没有经过反转 就返回第一个呗
        if(rotateArray[low] < rotateArray[high])
            return rotateArray[low];
        // 进行二分了
        while(rotateArray[low] > rotateArray[high])
        {
            // 终止条件
            if(high - low == 1)
                return rotateArray[high];
            mid = (low+high)/2;
            // 特殊情况 进入暴力算法
            if(rotateArray[low] == rotateArray[mid] && rotateArray[mid] == rotateArray[high])
            {
                return MinOrder(rotateArray,low,high);
            }
            if(rotateArray[mid] >= rotateArray[low])
            {
                low = mid;
            }
            if(rotateArray[mid] <= rotateArray[high])
            {
                high = mid;
            }
        }
    }
private:
    // 暴力算法
    int MinOrder(vector<int> &num, int low,int high)
    {
        int res = num[low];
        for(int i = low+1;i < high;i++)
        {
            if(num[i] < res)
            {
                res = num[i];
            }
        }
        return res;
    }
};
```

