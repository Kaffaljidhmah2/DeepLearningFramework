cmake_minimum_required(VERSION 3.2)

PROJECT(ts)

add_compile_options(-std=c++11)

INCLUDE_DIRECTORIES(
include
)

AUX_SOURCE_DIRECTORY(src DIR_SRCS)

SET(TMP
${DIR_SRCS}
)

ADD_EXECUTABLE(${PROJECT_NAME} ${TMP})