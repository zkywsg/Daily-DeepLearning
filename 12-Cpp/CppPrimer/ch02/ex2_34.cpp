#include<iostream>

int main()
{
	int i = 0, &r = i;  // r is an alias for i ,which type is int
	auto a = r;  // a is an int 
	
	const int ci = i, &cr = ci;
	auto b = ci; //b  is an int  ,top-level const in ci is dropped
	auto c = cr; //c is an int ,cr is an alias for ci whose const is top level
	auto d = &i; //d is an int*, & of an int object
	auto e = &ci; //e is const int*,& of a const object is low level const
	
	const auto f = ci; //deduced type of ci is int ,f has type const int
	auto &g = ci; //g is a const int& that is bound to ci
	
	a = 42; b=42; c = 42; *d = 42; e = &c;
	
	return 0;
}
