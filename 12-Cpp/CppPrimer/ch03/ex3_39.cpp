#include<iostream>
#include<string>
#include<cstring>

using std::cout;
using std::endl;
using std::string;

int main()
{
	//use string
	string s1("Lauzanhing"),s2("Ezreal");
	if(s1 == s2)
		cout << "same string." << endl;
	else if(s1 > s2)
		cout << "Lauzanhing > Ezreal" << endl;
	else
		cout << "Lauzanhing < Ezreal" << endl;
	
	cout << "=================" << endl;
	
	//use C-style character strings
	const char* cs1 = "Lauzanhing";
	const char* cs2 = "Ezreal";
	auto result = strcmp(cs1,cs2);
	if(result == 0)
		cout << "same thing." << endl;
	else if(result < 0)
		cout << "Lauzanhing < Ezreal" << endl;
	else
		cout << "Lauzanhing > Ezreal" << endl;
		
	return 0; 
}
