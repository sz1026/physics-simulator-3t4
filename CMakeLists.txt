cmake_minimum_required(VERSION 3.10)
project(OpenGL_Project)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find packages
find_package(OpenGL REQUIRED)

# Add external libraries
add_subdirectory(GLEW)
add_subdirectory(GLFW)
add_subdirectory(GLM)

# Include directories
include_directories(
        ${OPENGL_INCLUDE_DIR}
        ${GLEW_INCLUDE_DIRS}
        ${GLFW_INCLUDE_DIRS}
        ${GLM_INCLUDE_DIRS}
)

# Source files
set(SOURCES
        src/main.cpp
        src/shader.cpp
        src/renderer.cpp
)

# Create executable
add_executable(OpenGL_Project ${SOURCES})

# Link libraries
target_link_libraries(OpenGL_Project
        ${OPENGL_LIBRARIES}
        GLEW
        GLFW
        GLM
)