add_executable(waferedge-info waferedge_info.cpp)
target_link_libraries(waferedge-info PRIVATE waferedge waferedge_options)

add_executable(waferedge-maps waferedge_maps.cpp)
target_link_libraries(waferedge-maps PRIVATE waferedge waferedge_options)

add_executable(waferedge-classify waferedge_classify.cpp)
target_link_libraries(waferedge-classify PRIVATE waferedge waferedge_options)

add_executable(waferedge-cnn waferedge_cnn.cpp)
target_link_libraries(waferedge-cnn PRIVATE waferedge waferedge_options)

add_executable(tool-emulator tool_emulator.cpp)
target_link_libraries(tool-emulator PRIVATE waferedge waferedge_options)
