class Solution {
public:
	void replaceSpace(char *str,int length) {
        if(str == nullptr || length <= 0)
            return ;
        int originalLength = 0;
        int numberOfBlank = 0;
        int i = 0;
        while(str[i]!='\0')
        {
            ++ originalLength;
            if(str[i] == ' ')
                ++numberOfBlank;
            ++i;
        }
        int newLength = originalLength + numberOfBlank * 2;
        if(newLength > length)
            return;
        while(originalLength >= 0 && newLength > originalLength)
        {
            if(str[originalLength] == ' ')
            {
                str[newLength --] = '0';
                str[newLength --] = '2';
                str[newLength --] = '%';
            }
            else
            {
                str[newLength --] = str[originalLength];
            }
            -- originalLength;
        }
	}
};
