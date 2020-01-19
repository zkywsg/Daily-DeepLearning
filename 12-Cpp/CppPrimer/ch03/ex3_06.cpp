//Use a range for to change all the characters in a string to X.

#include <iostream>
#include <string>

using std::string;
using std::cout;
using std::endl;

int main()
{
	string str("a simple string");
	cout << str << endl;
	for(auto &c : str) c = 'x';
	cout << str << endl;
	
	return 0;
}
