CXX = g++
# CXXFLAGS = -std=c++17 -I./ -I./cereal -Wall
CXXFLAGS = -O3 -fPIC -shared -std=c++17 -rdynamic -I./ -I../.pixi/envs/default/include -L../.pixi/envs/default/lib -Wall
LIBS = -lhdf5_cpp -lhdf5 -lz -lssl -lcrypto

main: ptdalgorithms_jax.cpp
	$(CXX) $(CXXFLAGS) ptdalgorithms_jax.cpp -o libdph_param.so $(LIBS)

clean:
	rm -f ptdalgorithms_jax model.bin.z

