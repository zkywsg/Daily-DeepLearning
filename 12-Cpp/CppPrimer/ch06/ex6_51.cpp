#include<iostream> 
using std::cout;
using std::endl;

void f()
{
	cout << "f()" << endl;
}

void f(int)
{
	cout << "f(int)" << endl;
}

void f(int,int)
{
	cout << "f(int,int)" << endl;
}

void f(double,double)
{
	cout << "f(double,double)" << endl;
}

int main()
{
	f();
	f(42);
	f(42,0);
	f(2.11,23.33);
	
	return 0;
}
