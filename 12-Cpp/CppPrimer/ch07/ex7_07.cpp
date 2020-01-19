#include "ex7_06.h"

int main()
{
	sales_data total;
	if(read(std::cin,total))
	{
		sales_data trans;
		while(read(std::cin,trans))
		{
			if(total.isbn() == trans.isbn())
				total.combine(trans);
			else
			{
				print(std::cout,total) << std::endl;
				total = trans;
			}
		}
		print(std::cout,total) << std::endl;
	}
	else
	{
		std::cerr << "No data?!" << std::endl;
		return -1;
	}
	
	return 0;
}
