userspace_extension(
    ept_monitor
    SOURCE_DIR ${CMAKE_CURRENT_LIST_DIR}/userspace
    DEPENDS bfintrinsics
)

eapis_vmm_extension(
    ept_split
    SOURCE_DIR ${CMAKE_CURRENT_LIST_DIR}/vmm
)

# ------------------------------------------------------------------------------
# Custom Target
# ------------------------------------------------------------------------------

add_custom_target_category("EPT Monitor")

add_custom_target(monitor
    COMMAND ${USERSPACE_PREFIX_PATH}/bin/ept_monitor
    USES_TERMINAL
)

add_custom_target_info(
    TARGET monitor
    COMMENT "Run the EPT monitor application"
)
