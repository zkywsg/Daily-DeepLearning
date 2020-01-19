int i = 0, &r = i;
// same
auto a = i;
decltype(i) b = i;
// different "c" will be int "d" will be int&
auto c = r;
decltype(r) d = r;
