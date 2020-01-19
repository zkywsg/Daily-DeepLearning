//
// Created by Lauzanhing on 2018/3/20.
//

#include <iostream>
#include <string>
#include <deque>

using namespace std;

int main() {
    deque<string> input;
    for (string str; cin >> str; input.push_back(str));
    for (auto iter = input.cbegin(); iter != input.cend(); ++iter)
        cout << *iter << endl;

    return 0;
}