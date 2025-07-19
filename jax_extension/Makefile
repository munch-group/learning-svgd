CXX = g++
# CXXFLAGS = -std=c++17 -I./ -I./cereal -Wall
CXXFLAGS = -O3 -fPIC -shared -std=c++17 -rdynamic -I./ -I../.pixi/envs/default/include -L../.pixi/envs/default/lib -Wall
LIBS = -lhdf5_cpp -lhdf5 -lz -lssl -lcrypto

main: dph_pmf_param.cpp
	$(CXX) $(CXXFLAGS) dph_pmf_param.cpp -o libdph_param.so $(LIBS)

clean:
	rm -f dph_pmf_param model.bin.z

