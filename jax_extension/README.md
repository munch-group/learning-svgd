    c++ -O3 -fPIC -shared -std=c++17 -rdynamic -o libdph_param.so dph_pmf_param.cc

or alternatively on mac
    c++ -O3 -fPIC -std=c++17 -shared -Wl,-exported_symbol,_dph_pmf_param -o libdph_param.so dph_pmf_param.cc

with ptd lib:

    g++ -O3 -std=c++17 -fPIC -I/path/to/ptdalgorithms/include -L/path/to/ptdalgorithms/lib -lptdalgorithms -shared -o libdph_param.so dph_param_kernel.cc


Ensure the symbol is exported correctly:

    nm -D libdph_param.so | grep dph_pmf_param

You should see:

    000000000000xxxx T dph_pmf_param


