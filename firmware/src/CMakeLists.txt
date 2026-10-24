idf_component_register(SRCS "main.cpp" "controller.cpp"
                       INCLUDE_DIRS ".")
# No fused multiply-add, so float results match the Python simulation bit for bit.
target_compile_options(${COMPONENT_LIB} PRIVATE -ffp-contract=off)
