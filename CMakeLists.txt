cmake_minimum_required(VERSION 2.8)
project("ar")


## file globbing ##############################################################

# these instructions search the directory tree when cmake is invoked
# and put all files that match the pattern in the variables
# 'sources', 'sources_test', etc.
file(GLOB   sources   src/*.cpp src/*.hpp)
# file(GLOB_RECURSE sources_test src/test/*.cpp)
# file(GLOB_RECURSE data resources/*)
# you can use set(sources src/main.cpp) etc if you don't want to
# use globing to find files automatically

add_executable(main ${sources})


## dependencies ###############################################################

# this defines the variables OpenCV_LIBRARIES that contain all library names
# that we need to link to
find_package(OpenCV REQUIRED)

target_link_libraries(main
  ${OpenCV_LIBS}
)

include_directories(
  ${OpenCV_INCLUDES}
)


## set compile flags safely ###################################################

include(CheckCXXCompilerFlag)

function(enable_cxx_compiler_flag_if_supported flag)
  string(FIND "${CMAKE_CXX_FLAGS}" "${flag}" flag_already_set)
  if(flag_already_set EQUAL -1)
    check_cxx_compiler_flag("${flag}" flag_supported)
    if(flag_supported)
      set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${flag}" PARENT_SCOPE)
    endif()
  endif()
endfunction()

# here use the function ^
enable_cxx_compiler_flag_if_supported("-Wall")
enable_cxx_compiler_flag_if_supported("-Wextra")
enable_cxx_compiler_flag_if_supported("-pedantic")
# enable_cxx_compiler_flag_if_supported("-ansi")
enable_cxx_compiler_flag_if_supported("-std=c++11")
enable_cxx_compiler_flag_if_supported("-g")
enable_cxx_compiler_flag_if_supported("-Wformat=0")

# enable_cxx_compiler_flag_if_supported("-DSTEPS")
