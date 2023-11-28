idf_component_register(SRCS "nn.c" "train.c" "net.c"
                       INCLUDE_DIRS ".")

add_definitions("-Wno-restrict")
