#include<iostream>
#include<string>

using std::string;
using std::cout;
using std::endl;

inline bool is_shorter(const string& lft, const string& rht)
{
	return lft.size() < rht.size();
}

int main()
{
//	string str1 = "pezy";
//	string str2 = "Lauzanhing";
//	cout << is_shorter(str1,str2) << endl;
	cout << is_shorter("pezy","Lauzanhing") << endl;
	
	return 0;
}
