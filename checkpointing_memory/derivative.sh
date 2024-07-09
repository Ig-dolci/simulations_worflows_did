#!/bin/bash

# This script is used to test the memory usage of the program
# It will run the program with different input sizes and check the memory usage
# The memory usage is checked by using the valgrind tool

# The program to test

# # mprof run -o no_annotation_interpolate_recompute python memory_leak.py --function "interpolate" --annotate "False"
# mprof run -o no_annotation_burger_recompute python memory_leak.py --function "burger" --annotate "False"
# mprof run -o no_annotation_assign_recompute python memory_leak.py --function "assign" --annotate "False"
# # mprof run -o no_revolve_interpolate_recompute python memory_leak.py --function "interpolate" --checkpointing "False" --annotate "True"
# mprof run -o no_revolve_burger_recompute python memory_leak.py --function "burger" --checkpointing "False" --annotate "True"
# mprof run -o no_revolve_assign_recompute python memory_leak.py --function "assign" --checkpointing "False" --annotate "True"
# # mprof run -o revolve_s100_interpolate_recompute python memory_leak.py --function "interpolate" --checkpointing "True" --annotate "True" --n_checkpoints 100
# mprof run -o revolve_s100_burger_recompute python memory_leak.py --function "burger" --checkpointing "True" --annotate "True" --n_checkpoints 100
# mprof run -o revolve_s100_assign_recompute python memory_leak.py --function "assign" --checkpointing "True" --annotate "True" --n_checkpoints 100


#!/bin/bash

# This script is used to test the memory usage of the program
# It will run the program with different input sizes and check the memory usage
# The memory usage is checked by using the valgrind tool

# The program to test
mprof run -o no_revolve_interpolate_derivative python memory_leak.py --function "interpolate" --checkpoint "False" --annotate "True"
mprof run -o no_revolve_burger_derivative python memory_leak.py --function "burger" --checkpoint "False" --annotate "True"
mprof run -o no_revolve_assign_derivative python memory_leak.py --function "assign" --checkpoint "False" --annotate "True"
mprof run -o revolve_s10_interpolate_derivative python memory_leak.py --function "interpolate" --checkpoint "True" --annotate "True" --n_checkpoints 10
mprof run -o revolve_s10_burger_derivative python memory_leak.py --function "burger" --checkpoint "True" --annotate "True" --n_checkpoints 10
mprof run -o revolve_s10_assign_derivative python memory_leak.py --function "assign" --checkpoint "True" --annotate "True" --n_checkpoints 10
mprof run -o revolve_s100_interpolate_derivative python memory_leak.py --function "interpolate" --checkpoint "True" --annotate "True" --n_checkpoints 100
mprof run -o revolve_s100_burger_derivative python memory_leak.py --function "burger" --checkpoint "True" --annotate "True" --n_checkpoints 100
mprof run -o revolve_s100_assign_derivative python memory_leak.py --function "assign" --checkpoint "True" --annotate "True" --n_checkpoints 100
mprof run -o revolve_s200_interpolate_derivative python memory_leak.py --function "interpolate" --checkpoint "True" --annotate "True" --n_checkpoints 200
mprof run -o revolve_s200_burger_derivative python memory_leak.py --function "burger" --checkpoint "True" --annotate "True" --n_checkpoints 200
mprof run -o revolve_s200_assign_derivative python memory_leak.py --function "assign" --checkpoint "True" --annotate "True" --n_checkpoints 200
mprof run -o revolve_s400_interpolate_derivative python memory_leak.py --function "interpolate" --checkpoint "True" --annotate "True" --n_checkpoints 400
mprof run -o revolve_s400_burger_derivative python memory_leak.py --function "burger" --checkpoint "True" --annotate "True" --n_checkpoints 400
mprof run -o revolve_s400_assign_derivative python memory_leak.py --function "assign" --checkpoint "True" --annotate "True" --n_checkpoints 400
# mprof run -o revolve_s600_interpolate python memory_leak.py --function "interpolate" --checkpoint "True" --annotate "True" --n_checkpoints 600
# mprof run -o revolve_s600_burger python memory_leak.py --function "burger" --checkpoint "True" --annotate "True" --n_checkpoints 600
# mprof run -o revolve_s600_assign python memory_leak.py --function "assign" --checkpoint "True" --annotate "True" --n_checkpoints 600


# mprof run -o singlechk_s10_interpolate python memory_leak.py --function "interpolate" --checkpointing "True" --annotate "True" --schedule "single"
# mprof run -o singlechk_s10_burger python memory_leak.py --function "burger" --checkpointing "True" --annotate "True" --schedule "single" 
# mprof run -o singlechk_s10_assign python memory_leak.py --function "assign" --checkpointing "True" --annotate "True" --schedule "single" 
# mprof run -o singlechk_s100_interpolate python memory_leak.py --function "interpolate" --checkpointing "True" --annotate "True" --schedule "single"
# mprof run -o singlechk_s100_burger python memory_leak.py --function "burger" --checkpointing "True" --annotate "True" --schedule "single"
# mprof run -o singlechk_s100_assign python memory_leak.py --function "assign" --checkpointing "True" --annotate "True" --schedule "single"
# mprof run -o singlechk_s200_interpolate python memory_leak.py --function "interpolate" --checkpointing "True" --annotate "True" --schedule "single"
# mprof run -o singlechk_s200_burger python memory_leak.py --function "burger" --checkpointing "True" --annotate "True" --schedule "single"
# mprof run -o singlechk_s200_assign python memory_leak.py --function "assign" --checkpointing "True" --annotate "True" --schedule "single"
# mprof run -o singlechk_s400_interpolate python memory_leak.py --function "interpolate" --checkpointing "True" --annotate "True" --schedule "single"
# mprof run -o singlechk_s400_burger python memory_leak.py --function "burger" --checkpointing "True" --annotate "True" --schedule "single"
# mprof run -o singlechk_s400_assign python memory_leak.py --function "assign" --checkpointing "True" --annotate "True" --schedule "single"
# mprof run -o singlechk_s600_interpolate python memory_leak.py --function "interpolate" --checkpointing "True" --annotate "True" --schedule "single"
# mprof run -o singlechk_s600_burger python memory_leak.py --function "burger" --checkpointing "True" --annotate "True" --schedule "single"
# mprof run -o singlechk_s600_assign python memory_leak.py --function "assign" --checkpointing "True" --annotate "True" --schedule "single"
