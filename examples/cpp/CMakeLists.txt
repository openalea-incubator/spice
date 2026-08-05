add_executable(main "main.cpp")
target_link_libraries(main PUBLIC pm image triangle primitive photon_map)
target_include_directories(main PUBLIC "../../src/cpp/include")