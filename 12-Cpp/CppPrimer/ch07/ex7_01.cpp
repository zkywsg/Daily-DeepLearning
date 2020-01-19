#include<iostream>
#include<string>

using std::cin;
using std::cout;
using std::endl;
using std::string;

struct sales_data
{
	string bookNo;
	unsigned units_sold;
	double revenue = 0.0;
};

int main()
{
	sales_data total;
	if(cin >> total.bookNo >> total.units_sold >> total.revenue)
	{
		sales_data trans;
		while(cin >> trans.bookNo >> trans.units_sold >> trans.revenue)
		{
			if(total.bookNo == trans.bookNo)
			{
				total.units_sold += trans.units_sold;
				total.revenue += trans.revenue;
			}
			else
			{
				cout << total.bookNo << " " << total.units_sold << " " << total.revenue << endl;
				total = trans;
			}
		}
		cout << total.bookNo << " " << total.units_sold << " " << total.revenue << endl;
	}
	else
	{
		std::cerr << "No data?!" << std::endl;
		return -1;
	}
	return 0;
}
