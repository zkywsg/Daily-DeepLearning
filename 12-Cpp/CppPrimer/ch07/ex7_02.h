#ifndef ex7_02_h
#define ex7_02_h

#include<string>

struct sales_data
{
	std::string isbn() const { return bookNo;};
	sales_data& combine (const sales_data&);
	
	std::string bookNo;
	unsigned units_sold = 0;
	double revenue = 0.0;
};

sales_data& sales_data::combine(const sales_data& rhs)
{
	units_sold += rhs.units_sold;
	revenue += rhs.revenue;
	return *this;
}

#endif
