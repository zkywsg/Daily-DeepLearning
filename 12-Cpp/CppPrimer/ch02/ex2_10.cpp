What are the initial values, if any, of each of the following variables?
std::string global_str;
int global_int;
int main()
{
    int local_int;
    std::string local_str;
}
global_str is global variable, so the value is empty string. global_int is global variable,
so the value is zero. local_int is a local variable which is uninitialized, so it has a undefined value.
local_str is also a local variable which is uninitialized, but it has a value that is defined by the class.
So it is empty string. PS: please read P44 in the English version, P40 in Chinese version to get more.
The note: Uninitialized objects of built-in type defined inside a function body have a undefined value.
Objects of class type that we do not explicitly inititalize have a value that is defined by class.
