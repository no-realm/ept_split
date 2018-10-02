set(CMAKE_BUILD_TYPE Release)

list(APPEND EXTENSION
    ${CMAKE_CURRENT_LIST_DIR}/../extended_apis
    ${CMAKE_CURRENT_LIST_DIR}/../ept_split
)

set(OVERRIDE_VMM ept_split_vmm)
set(OVERRIDE_VMM_TARGET ept_split)

if(ENABLE_BUILD_EFI)
    set(BUILD_SHARED_LIBS OFF)
    set(BUILD_STATIC_LIBS ON)
else()
    set(BUILD_SHARED_LIBS ON)
    set(BUILD_STATIC_LIBS OFF)
endif()

set(ENABLE_BUILD_VMM ON)
set(ENABLE_BUILD_USERSPACE ON)
set(ENABLE_BUILD_TEST OFF)
set(ENABLE_EXTENDED_APIS ON)

# ------------------------------------------------------------------------------
# Override VMM
# ------------------------------------------------------------------------------

if(OVERRIDE_VMM)
    if(OVERRIDE_VMM_TARGET)
        set_bfm_vmm(${OVERRIDE_VMM} TARGET ${OVERRIDE_VMM_TARGET})
    else()
        set_bfm_vmm(${OVERRIDE_VMM})
    endif()
endif()
