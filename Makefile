# ---- device-oob-detector Makefile ----
#
# Build the shared engine, both front-ends, the device patch cubin, and sample.
#
#   make                 # default arch sm_80
#   make SM=sm_90        # target Hopper
#   make clean
#
# Requires: nvcc in PATH, Compute Sanitizer headers + libsanitizer-public.so.

CUDA_HOME     ?= /home/Cuda-13
SANITIZER_DIR ?= $(CUDA_HOME)/compute-sanitizer
NVCC          ?= $(CUDA_HOME)/bin/nvcc
CXX           ?= g++

SM            ?= sm_80

BUILD    := build
INC      := -Iinclude -I$(SANITIZER_DIR)/include -I$(CUDA_HOME)/include
SAN_LIB  := -L$(SANITIZER_DIR) -lsanitizer-public
CUDA_LIB := -L$(CUDA_HOME)/lib64/stubs -L$(CUDA_HOME)/lib64 -L$(CUDA_HOME)/targets/sbsa-linux/lib/stubs -lcuda

CXXFLAGS := -O2 -fPIC -std=c++17 -Wall $(INC)

ENGINE_SRC   := src/sanitizer_engine.cpp
EMBED_SRC    := src/oob_detector.cpp
INJECT_SRC   := src/external_inject.cpp
PATCH_SRC    := device/oob_patch.cu
SAMPLE_SRC   := examples/vec_add_oob.cu

.PHONY: all clean dirs sample
all: dirs $(BUILD)/oob_patch.cubin \
         $(BUILD)/liboob_detector.so \
         $(BUILD)/liboob_inject.so \
         sample

dirs:
	@mkdir -p $(BUILD)

# ---- device patch (special --compile-as-tools-patch) ----
$(BUILD)/oob_patch.cubin: $(PATCH_SRC) | dirs
	$(NVCC) --cubin --compile-as-tools-patch $< -o $@ \
		-arch=$(SM) -I$(SANITIZER_DIR)/include -I$(CUDA_HOME)/include

# ---- shared engine object ----
$(BUILD)/sanitizer_engine.o: $(ENGINE_SRC) include/oob_detector.h | dirs
	$(CXX) $(CXXFLAGS) -c $< -o $@

# ---- embedded library ----
$(BUILD)/oob_detector.o: $(EMBED_SRC) include/oob_detector.h | dirs
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(BUILD)/liboob_detector.so: $(BUILD)/oob_detector.o $(BUILD)/sanitizer_engine.o
	$(CXX) -shared -o $@ $^ $(SAN_LIB) $(CUDA_LIB)

# ---- external injection library ----
$(BUILD)/external_inject.o: $(INJECT_SRC) | dirs
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(BUILD)/liboob_inject.so: $(BUILD)/external_inject.o $(BUILD)/sanitizer_engine.o
	$(CXX) -shared -o $@ $^ $(SAN_LIB) $(CUDA_LIB) -ldl

# ---- sample (both variants) ----
sample: $(BUILD)/vec_add_oob $(BUILD)/vec_add_oob_embedded

$(BUILD)/vec_add_oob: $(SAMPLE_SRC) | dirs
	$(NVCC) -arch=$(SM) $< -o $@

$(BUILD)/vec_add_oob_embedded: $(SAMPLE_SRC) $(BUILD)/liboob_detector.so | dirs
	$(NVCC) -arch=$(SM) -DOOB_EMBEDDED $< -o $@ \
		-Iinclude -I$(SANITIZER_DIR)/include \
		-L$(BUILD) -loob_detector -Xlinker -rpath -Xlinker $(SANITIZER_DIR)

clean:
	rm -rf $(BUILD)
