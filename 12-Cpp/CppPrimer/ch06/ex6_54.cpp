#include<iostream>
#include<string>
#include<vector>


using std::vector;
using std::cout;

inline int f(const int,const int);
typedef decltype(f) fp;

inline int NumAdd(const int n1, const int n2) 
{
	return n1 + n2;
}

inline int NumSub(const int n1, const int n2) 
{
	return n1 - n2;
}

inline int NumMul(const int n1, const int n2) 
{
	return n1 * n2;
}

inline int NumDiv(const int n1, const int n2) 
{
	return n1 / n2;
}

vector<fp*> v{ NumAdd,NumSub,NumMul,NumDiv};

int main()
{
	for(auto it = v.cbegin(); it != v.cend(); ++it)
		cout << (*it)(2,2) << std::endl;
		
	return 0;
 } 
