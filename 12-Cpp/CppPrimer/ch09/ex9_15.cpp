//
// Created by Lauzanhing on 2018/3/20.
//

#include <iostream>
#include <vector>

using namespace std;

int main() {
    vector<int> vec1{1, 2, 3, 4, 5};
    vector<int> vec2{1, 2, 3, 4, 5};
    vector<int> vec3{1,2,3,4};

    cout<<(vec1==vec2?"true":"false")<<endl;
    cout<<(vec1==vec3?"true":"false")<<endl;

    return 0;
}