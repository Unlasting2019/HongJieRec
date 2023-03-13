CMAKE_MINIMUM_REQUIRED(VERSION 3.16)
PROJECT(rank_tfrecord)

INCLUDE_DIRECTORIES(
    ${CPP_SRC_DIR}/
    ${CPP_PKG_DIR}/
    ${CPP_PKG_DIR}/tensorflow_protobuf/include/
    ${CPP_PKG_DIR}/fmt/include/
    ${CPP_PKG_DIR}/numeric_ranges/include/
)

AUX_SOURCE_DIRECTORY(${CPP_PKG_DIR}/CharConv pkg_src)
AUX_SOURCE_DIRECTORY(${CPP_PKG_DIR}/fast_float pkg_src)
AUX_SOURCE_DIRECTORY(${CPP_PKG_DIR}/cpp_reflect pkg_src)
AUX_SOURCE_DIRECTORY(${CPP_PKG_DIR}/cpp_tfrecord pkg_src)
AUX_SOURCE_DIRECTORY(${CPP_PKG_DIR}/cpp_frame pkg_src)

ADD_EXECUTABLE(${PROJECT_NAME} main.cpp ${pkg_src})

TARGET_LINK_LIBRARIES(
    ${PROJECT_NAME}
    /usr/lib/x86_64-linux-gnu/libdl.so
    ${CPP_PKG_DIR}/fmt/lib/libfmt.a
    ${CPP_PKG_DIR}/tensorflow_protobuf/lib/libprotobuf.so
)
