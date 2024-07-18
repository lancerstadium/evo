include(CTest)

set(INC_DIR "${PROJECT_SOURCE_DIR}/include")
set(GEN_DIR "${CMAKE_CURRENT_BINARY_DIR}/generated")
set(FBS_DIR "${CMAKE_CURRENT_SOURCE_DIR}")

include_directories("${GEN_DIR}" "${INC_DIR}")

add_custom_target(gen_etm_fbs ALL) 
add_custom_command (
    TARGET gen_etm_fbs
    COMMAND ${CMAKE_COMMAND} -E make_directory "${GEN_DIR}"
    COMMAND flatcc_cli -a -o "${GEN_DIR}" "${FBS_DIR}/etm.fbs"
    DEPENDS flatcc_cli "${FBS_DIR}/etm.fbs" 
)
add_executable(etm etm.c)
add_dependencies(etm gen_etm_fbs)
target_link_libraries(etm flatccrt)

if (FLATCC_TEST)
    add_test(etm etm${CMAKE_EXECUTABLE_SUFFIX})
endif()
