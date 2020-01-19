//
// Created by Lauzanhing on 2018/3/20.
//

#include <iostream>
#include <string>
#include <vector>
#include <list>

int main() {
    std::list<const char *> l{"Mooophy", "Pezy", "Queeuqueg"};
    std::vector<std::string> v;
    v.assign(l.cbegin(), l.cend());
    for (auto ptr:v)
        std::cout << ptr << std::endl;

    return 0;
}