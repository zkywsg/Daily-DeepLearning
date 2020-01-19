#include <iostream>
using std::cout;

int larger_one(const int i, const int *const p)
{
    return (i > *p) ? i : *p;
}

int main()
{
    int i = 6;
    cout << larger_one(7, &i);

    return 0;
}
