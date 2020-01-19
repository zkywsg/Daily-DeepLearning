#include <fstream>
#include <string>
#include <vector>
#include <iostream>

using std::vector;
using std::string;
using std::ifstream;
using std::cout;
using std::endl;

void ReadFileToVec(/* arguments */const string& fileName, vector<string>& vec)
{
    /* code */
    ifstream ifs(fileName);
    if(ifs)
    {
        string buf;
        while (std::getline(ifs,buf))
        {
            /* code */
            vec.push_back(buf);
        }
    }
}



int main(int argc, char const *argv[]) {
    vector<string> vec;
    ReadFileToVec("../data/book.txt",vec);
    for(const auto &str : vec)
        cout << str << endl;
    return 0;
}
