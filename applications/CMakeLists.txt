FILE(GLOB SOURCE_FILES "*.cc")

FOREACH ( source_file ${SOURCE_FILES} )

    GET_FILENAME_COMPONENT(file_name ${source_file} NAME)

    STRING( REPLACE ".cc" "" exec ${file_name} )
    ADD_EXECUTABLE( ${exec} ${source_file})
    DEAL_II_SETUP_TARGET(${exec})
    TARGET_LINK_LIBRARIES(${exec} "ibm")

    find_library(ADAFLO adaflo HINTS ${ADAFLO_LIB})
    if(ADAFLO)
        message("-- Adaflo found: ${ADAFLO}")
        target_compile_definitions(${exec} PUBLIC IBM_WITH_ADAFLO)
        target_include_directories(${exec} PUBLIC ${ADAFLO_INCLUDE})
        target_link_libraries(${exec} ${ADAFLO})
        set(DEAL_II_WITH_ADAFLO_SUPPORT ON)
    endif()
ENDFOREACH ( source_file ${SOURCE_FILES} )
