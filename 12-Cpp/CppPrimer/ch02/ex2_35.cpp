#include <iostream>
#include <typeinfo>

int main()
{
	const int  i = 42;
	auto j = i;
	const  auto &k = i;
	auto *p = &i;
	const auto j2 = i, &k2 = i;

	std::cout   << "j is " << typeid(j).name()
				<< "\nk is " << typeid(k).name()
				<< "\np is " << typeid(p).name()
				<< "\nj2 is " << typeid(j2).name()
				<< "\nk2 is " << typeid(k2).name()
				<< std::endl;
	system("pause");
	return 0;
}