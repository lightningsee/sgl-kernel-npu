# set custom compile options
if ("${CMAKE_BUILD_TYPE}x" STREQUAL "Debugx")
    npu_op_kernel_options(ALL OPTIONS -g -O0 )
endif()

message(STATUS "ASCEND include path: ${ASCEND_CANN_PACKAGE_PATH}/opp/built-in/op_impl/ai_core/tbe/impl/ascendc/moe_distribute_dispatch")

set(_DISPATCH_FFN_INC_OPTS)
if (EXISTS ${CMAKE_SOURCE_DIR}/third_party/catlass/include)
    list(APPEND _DISPATCH_FFN_INC_OPTS -I${CMAKE_SOURCE_DIR}/third_party/catlass/include)
else()
    message(FATAL_ERROR "dependency catlass is missing")
endif()
message(WARNING "_DISPATCH_FFN_INC_OPTS: ${_DISPATCH_FFN_INC_OPTS}")

# FusedDeepMoe A5 on Ascend 950 reuses the same Catlass dependency fetched by
# deepep/ops/build.sh for A3 kernels. These options must be appended to the
# final global option set because OPC's per-op compiler inherits that set.
set(_A5_FUSED_DEEP_MOE_INC_OPTS)
if ("${ASCEND_COMPUTE_UNIT}" MATCHES "^ascend950")
    set(_A5_SYSTEM_MOE_BASE_COMMON_DIR "${ASCEND_CANN_PACKAGE_PATH}/opp/built-in/op_impl/ai_core/tbe/impl/ops_transformer/ascendc/common")
    set(_A5_SYSTEM_MOE_BASE_OP_KERNEL "${_A5_SYSTEM_MOE_BASE_COMMON_DIR}/op_kernel/moe_distribute_base.h")
    set(_A5_SYSTEM_MOE_BASE_INC_KERNEL "${_A5_SYSTEM_MOE_BASE_COMMON_DIR}/inc/kernel/moe_distribute_base.h")
    set(_A5_SYSTEM_MOE_BASE_COMMON "${_A5_SYSTEM_MOE_BASE_COMMON_DIR}/moe_distribute_base.h")

    if (EXISTS "${_A5_SYSTEM_MOE_BASE_OP_KERNEL}")
        list(APPEND _A5_FUSED_DEEP_MOE_INC_OPTS
            -DDEEPEP_A5_SYSTEM_MOE_BASE_USE_OP_KERNEL=1)
    elseif(EXISTS "${_A5_SYSTEM_MOE_BASE_INC_KERNEL}")
        list(APPEND _A5_FUSED_DEEP_MOE_INC_OPTS
            -DDEEPEP_A5_SYSTEM_MOE_BASE_USE_INC_KERNEL=1)
    elseif(EXISTS "${_A5_SYSTEM_MOE_BASE_COMMON}")
        list(APPEND _A5_FUSED_DEEP_MOE_INC_OPTS
            -DDEEPEP_A5_SYSTEM_MOE_BASE_USE_COMMON=1)
    else()
        message(FATAL_ERROR
            "Cannot find system moe_distribute_base.h for A5 fused. "
            "Checked: ${_A5_SYSTEM_MOE_BASE_OP_KERNEL}, ${_A5_SYSTEM_MOE_BASE_INC_KERNEL} and ${_A5_SYSTEM_MOE_BASE_COMMON}")
    endif()

    list(APPEND _A5_FUSED_DEEP_MOE_INC_OPTS
        -I$ENV{ASCEND_TOOLKIT_HOME}/opp/built-in/op_impl/ai_core/tbe/impl/ops_transformer/ascendc/common/
        -DCATLASS_ARCH=3510
        -DDEEPEP_CATLASS_A5_COMPAT=1
        -DDEEPEP_SKIP_DISPATCH_FFN_COMBINE=1)

    # 1C1V 串行模拟 / 调试打印开关：由外部环境变量控制并透传为编译期宏，默认关闭（A5 原行为不变）。
    # 用法: FUSED_MOE_SIM_1C1V=1 FUSED_MOE_DEBUG_PRINT=1 bash build.sh -a deepep Ascend950
    if (DEFINED ENV{FUSED_MOE_SIM_1C1V})
        list(APPEND _A5_FUSED_DEEP_MOE_INC_OPTS -DFUSED_MOE_SIM_1C1V)
        message(STATUS "FUSED_MOE_SIM_1C1V: 1C1V serial simulation enabled")
    endif()
    if (DEFINED ENV{FUSED_MOE_DEBUG_PRINT})
        list(APPEND _A5_FUSED_DEEP_MOE_INC_OPTS -DFUSED_MOE_DEBUG_PRINT)
        message(STATUS "FUSED_MOE_DEBUG_PRINT: kernel printf enabled")
    endif()
endif()

npu_op_kernel_sources(ascendc_kernels
    KERNEL_DIR ./
)

npu_op_kernel_library(ascendc_kernels
    SRC_BASE ${CMAKE_CURRENT_SOURCE_DIR}/
    TILING_LIBRARY cust_optiling
)

npu_op_package_add(${package_name}
    LIBRARY ascendc_kernels
)

# printf 调试输出依赖 ASCENDC_DUMP：开 FUSED_MOE_DEBUG_PRINT 时同步放开，平时保持 0 不变
if (DEFINED ENV{FUSED_MOE_DEBUG_PRINT})
    set(_ASCENDC_DUMP_FLAG "-DASCENDC_DUMP=1")
else()
    set(_ASCENDC_DUMP_FLAG "-DASCENDC_DUMP=0")
endif()

npu_op_kernel_options("" ALL OPTIONS
    ${_ASCENDC_DUMP_FLAG}
    --cce-auto-sync=off
    ${_DISPATCH_FFN_INC_OPTS}
    ${_A5_FUSED_DEEP_MOE_INC_OPTS})
