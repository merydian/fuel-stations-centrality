cd ..
c++ -O3 -Wall -shared -std=c++17 -fPIC \
   $(python3 -m pybind11 --includes) \
   centrality_core.cpp -o centrality_core$(python3-config --extension-suffix) \
   -fopenmp
