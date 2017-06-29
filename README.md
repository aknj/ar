### Building 

To build the application: make a build directory (eg. `build/`), run `cmake` in that dir, and then use `make` to build the target.

#### Example

###### on Windows using CMake and GCC

``` bash
mkdir build && cd build
cmake ../src -G "MinGW Makefiles"  # arguments are: location of CMakeLists.txt, build system generator
mingw32-make.exe                   # path to make executable
```

###### on Linux

``` bash
mkdir build && cd build
cmake ../src                       # argument is location of CMakeLists.txt
make
