### Building 

To build the application: make a build directory (eg. `build/`), run `cmake` in that dir, and use `make` to build the target.

#### Example

###### on Windows using CMake and GCC (mingw-w64)

``` bash
mkdir build ; cd build
cmake -G "MinGW Makefiles" ../src  # option to specify the appropriate generator, argument is the location of CMakeLists.txt 
mingw32-make.exe                   # path to make executable
```

###### on Linux

``` bash
mkdir build && cd build
cmake ../src                       # argument is the location of CMakeLists.txt
make
