add_executable(waferedge-info waferedge_info.cpp)
target_link_libraries(waferedge-info PRIVATE waferedge waferedge_options)

add_executable(waferedge-maps waferedge_maps.cpp)
target_link_libraries(waferedge-maps PRIVATE waferedge waferedge_options)
