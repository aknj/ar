cmake_minimum_required(VERSION 2.8)

project( "OpenCV" )
find_package( OpenCV REQUIRED )

ADD_EXECUTABLE( main main.cpp )

INCLUDE_DIRECTORIES( 
    c:/OpenCV/opencv-2.4.13/build/include 
    c:/mingw64/mingw64/x86_64-w64-mingw32/include
    c:/mingw64/mingw64/x86_64-w64-mingw32/include/sys )
LINK_DIRECTORIES( c:/OpenCV/opencv-2.4.13/release/lib )


TARGET_LINK_LIBRARIES( main opencv_core opencv_imgproc opencv_highgui )


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

# example usage
enable_cxx_compiler_flag_if_supported("-Wall")
enable_cxx_compiler_flag_if_supported("-Wextra")
enable_cxx_compiler_flag_if_supported("-pedantic")
# enable_cxx_compiler_flag_if_supported("-ansi")
enable_cxx_compiler_flag_if_supported("-std=c++11")
enable_cxx_compiler_flag_if_supported("-g")
enable_cxx_compiler_flag_if_supported("-Wformat=0")

#enable_cxx_compiler_flag_if_supported("-DSTEPS")
