#include "ex7_02.h"
#include<iostream>
using std::cin;
using std::endl;
using std::cout;

int main()
{
	sales_data total;
	if(cin >> total.bookNo >> total.units_sold >> total.revenue)
	{
		sales_data trans;
		while(cin >> trans.bookNo >> trans.units_sold >> trans.revenue)
		{
			if(total.isbn() == trans.isbn())
				total.combine(trans);
			else
			{
				cout << total.bookNo << " " << total.units_sold << " " << total.revenue << endl;
				total = trans;
			}
		}
		cout << total.bookNo << " " << total.units_sold << " "  << total.revenue << endl;
	}
	else
	{
		std::cerr << " No data?!" << std::endl;
		return -1;
	}
	return 0;
}
