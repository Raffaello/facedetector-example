# GTest
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    enable_testing()

    find_package(GTest CONFIG REQUIRED)
	add_executable(runUnitTests "face-detector-example-test.cpp")
    target_link_libraries(runUnitTests PRIVATE GTest::gtest GTest::gtest_main GTest::gmock GTest::gmock_main)
    
    add_test(AllTestsInMain runUnitTests)
    #file(COPY GTest::gtest_main DESTINATION ${PROJECT_BINARY_DIR})
    #add_custom_command(TARGET runUnitTests POST_BUILD
    #COMMAND ${CMAKE_COMMAND} -E copy_if_different $<TARGET_FILE:GTest::gtest_main> $<TARGET_FILE_DIR:runUnitTests>
    #COMMAND ${CMAKE_COMMAND} -E copy_if_different $<TARGET_FILE:GTest::gtest> $<TARGET_FILE_DIR:runUnitTests>
    #COMMAND ${CMAKE_COMMAND} -E copy_if_different $<TARGET_FILE:GTest::gmock> $<TARGET_FILE_DIR:runUnitTests>
    )
    
endif()