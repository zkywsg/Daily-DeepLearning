#include<iostream>

int main()
{
	int a = 0 ,b = 1;
	int *p1 = &a , *p2 = p1;
	
	std :: cout << p1 << "\n";
	std :: cout << *p2 << "\n";
	
	//change the value of pointer
	p1 = &b;
	std :: cout << p1 << "\n";
	//change the value  to which  the pointer points
	*p2 = b;
	std :: cout << *p2;
	
	return 0;
}
