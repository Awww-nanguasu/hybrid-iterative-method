cmake_minimum_required(VERSION 3.10)
project(HIM_Solver)

# 1. 设置 C++ 标准
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 2. 查找 LibTorch (这个必须用 find_package)
find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

# ---------------- 改动开始 ----------------
# 3. 这里的改动最关键：
# 我们删掉了 find_package(Eigen3 REQUIRED)
# 直接用 include_directories 暴力引入路径，跳过版本检查
include_directories("C:/eigen-5.0.0") 
# ---------------- 改动结束 ----------------

# 4. 查找源文件
file(GLOB SOURCES "*.cpp")

# 5. 生成可执行文件
add_executable(main ${SOURCES})

# 6. 链接库
# 注意：删掉了 Eigen3::Eigen，因为上面 include 过了，这里不需要再链接
target_link_libraries(main "${TORCH_LIBRARIES}")