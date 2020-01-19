// read a sequence of ints from cin and
// store those values in a vector.

#include<iostream>
#include<vector>

int main()
{
	std::vector<int> vec;
	for(int i;std::cin >> i;vec.push_back(i));
	
	return 0;
 } 



