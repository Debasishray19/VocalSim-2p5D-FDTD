cmake_minimum_required(VERSION 3.18 FATAL_ERROR)

# 1) Project name & languages
project(VocalSim LANGUAGES CXX CUDA)

# 2) C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 3) Default build type
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Debug CACHE STRING "Build type" FORCE)
endif()

# 4) CUDA compile flags
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -lineinfo")
set(CMAKE_CUDA_FLAGS_DEBUG "${CMAKE_CUDA_FLAGS_DEBUG} -G -lineinfo")
set(CMAKE_CUDA_FLAGS_RELEASE "${CMAKE_CUDA_FLAGS_RELEASE} -O2")

# 5) GPU architectures
set(CMAKE_CUDA_ARCHITECTURES 75 80 86)

# 6) Pull in CUDA
find_package(CUDAToolkit REQUIRED)

# 7) List sources & headers
set(SOURCES
  src/main.cu
  src/SimulationParams.cu
  src/SolverParams.cu
  src/FDTDSolver.cu
  src/GlottalExcitations.cu
  src/DeviceSymbols.cu
  src/TubeGeometry.cu
  src/TubeDepth.cu
  src/FDTDEngineCPU.cpp
  src/FDTDEngineEigen.cpp
  src/ShapeGenerator.cpp
  src/ShapeAlignment.cu
  src/Utils.cpp
  src/TwoMassModel.cpp
)

# To check the constant memory usage
#set_source_files_properties(
#  ${CMAKE_CURRENT_SOURCE_DIR}/src/FDTDEngine.cu
#  PROPERTIES
#    COMPILE_FLAGS "$<$<CONFIG:Release>:-Xptxas=-v>"
#)

set(HEADERS
  include/SimulationUnits.h
  include/SimulationParams.h
  include/FDTDSolver.h
  include/SolverParams.h
  include/GlottalExcitations.h
  include/DeviceSymbols.h
  include/TubeGeometry.h
  include/TubeDepth.h
  include/FDTDEngineCPU.h
  include/FDTDEngineEigen.h
  include/ShapeGenerator.h
  include/ShapeAlignment.h
  include/Utils.h
  include/TwoMassModel.h
  include/TestTwoMassModel.h
)

# 8) Create the executable
add_executable(${PROJECT_NAME}
  ${SOURCES}
  ${HEADERS}    # this just makes them show up in IDEs
)

# Suppress warnings coming from Eigen headers
target_compile_options(VocalSim
  PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:
      --expt-relaxed-constexpr
      -Xcudafe=--diag_suppress=20012
      -Xcudafe=--diag_suppress=20013
      -Xcudafe=--diag_suppress=20015
      -Xcompiler=/wd4068
    >
    $<$<COMPILE_LANGUAGE:CXX>:/wd4068> 
)

# 9) Include path
target_include_directories(${PROJECT_NAME}
  PRIVATE
    # Add the path to Eigen library
    "C:/Users/LENOVO/vcpkg/installed/x64-windows/include"
    "C:/ProgramData/chocolatey/lib/eigen/include/eigen3"
    ${CMAKE_SOURCE_DIR}/include
)

# 10) Group headers in Visual Studio
if(MSVC)
  source_group("Header Files" FILES ${HEADERS})
endif()

# 11) Suppress LIBCMT warnings on Windows
if (MSVC)
  # this goes only to the linker, not to cl during compilation or CUDA host‐compile
  set_property(TARGET ${PROJECT_NAME} PROPERTY LINK_FLAGS "/NODEFAULTLIB:LIBCMT")
endif()

# 12) Enable separable device code
set_target_properties(${PROJECT_NAME} PROPERTIES
  CUDA_SEPARABLE_COMPILATION ON
)

# 13) Link against the CUDA runtime (CMake will also pull in cudadevrt for you)
target_link_libraries(${PROJECT_NAME}
  PRIVATE
    CUDA::cudart
)