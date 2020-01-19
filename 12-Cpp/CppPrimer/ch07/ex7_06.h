#ifndef ex7_06_h
#define ex7_06_h

#include<iostream>
#include<string>

struct sales_data
{
	std::string const& isbn() const {return bookNo;};
	sales_data& combine(const sales_data&);

	std::string bookNo;
	unsigned units_sold = 0;
	double revenue = 0.0;
};

//member functions.
sales_data& sales_data::combine(const sales_data &rhs)
{
	units_sold += rhs.units_sold;
	revenue += rhs.revenue;
	return *this; 
}

//nonmember functions.
std::istream &read(std::istream &is, sales_data &item)
{
	double price = 0;
	is >> item.bookNo >> item.units_sold >> price;
	item.revenue = price * item.units_sold;
	return is;
}

std::ostream &print(std::ostream &os, const sales_data &item)
{
	os << item.isbn() << " " << item.units_sold << " " << item.revenue;
	return os;
}

sales_data add(const sales_data &lhs,const sales_data &rhs)
{
	sales_data sum = lhs;
	sum.combine(rhs);
	return sum;
}

#endif
