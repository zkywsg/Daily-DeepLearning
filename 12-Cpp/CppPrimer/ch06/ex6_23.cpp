#include<iostream>

using namespace std;

void print(const int* pi)
{
	if(pi)
		cout << *pi << endl;
}

void print(const char* p)
{
	if(p)
		while(*p) cout << *p++;
	cout << endl;
}

void print(const int* beg , const int* end)
{
	while(beg != end)
		cout << *beg++ << endl; 
}

void print(const int ia[], size_t size)
{
	for(size_t i = 0; i != size; ++i)
	{
		cout << ia[i] << endl;
	}
}

void print(int (&arr)[3])
{
	for(auto i : arr)
		cout << i << endl;
}

int main()
{
	int i = 0, j[3] = {0,1,2};
	char ch[5] = "abcd";
	
	print(ch);
	cout << "-------------------------" << endl; 
	print(begin(j),end(j));
	cout << "-------------------------" << endl;
	print(&i);
	cout << "-------------------------" << endl;
	print(j,end(j) - begin(j));
	cout << "-------------------------" << endl;
	print(j);
	cout << "-------------------------" << endl;
	
	return 0;
}
