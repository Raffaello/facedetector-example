# CMakeList.txt : CMake project for face-detector-example, include source and define
# project specific logic here.
#
cmake_minimum_required (VERSION 3.8)

 find_package(OpenCV CONFIG REQUIRED)
 
# Note: 10 target(s) were omitted.


# Add source to this project's executable.
add_executable (face-detector-example "face-detector-example.cpp" "face-detector-example.h")

# TODO: Add tests and install targets if needed.
target_link_libraries(face-detector-example PRIVATE ${OpenCV_LIBS} )

add_custom_command(TARGET face-detector-example POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_directory ${CMAKE_SOURCE_DIR}/data/haarcascades/ $<TARGET_FILE_DIR:face-detector-example>/data/haarcascades
)

