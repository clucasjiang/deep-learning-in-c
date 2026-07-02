CC := clang
CFLAGS ?= -std=c11 -Wall -Wextra -Wpedantic -O2
LDLIBS ?= -lm

MODEL ?= mnist_mlp
INDEX ?= 0
TRAIN_ARG ?= train

# Add future model directories here, then define <model>_SRCS below.
# Example:
# MODELS := mnist_mlp mnist_cnn mnist_lstm
MODELS := mnist_mlp

mnist_mlp_SRCS := \
	mnist_mlp/main.c \
	mnist_mlp/mlp.c \
	mnist_mlp/read_data.c \
	mnist_mlp/mlp_weights.c \
	mnist_mlp/stored_weights.c

.PHONY: all help models run train clean $(MODELS) $(addprefix run-,$(MODELS)) $(addprefix train-,$(MODELS)) $(addprefix clean-,$(MODELS))

all: $(MODELS)

help:
	@printf "Targets:\n"
	@printf "  make                  Build all models\n"
	@printf "  make models           List available models\n"
	@printf "  make <model>          Build one model, e.g. make mnist_mlp\n"
	@printf "  make run              Run MODEL=%s with INDEX=%s\n" "$(MODEL)" "$(INDEX)"
	@printf "  make run-<model>      Run one model directly, e.g. make run-mnist_mlp INDEX=100\n"
	@printf "  make train            Train MODEL=%s\n" "$(MODEL)"
	@printf "  make train-<model>    Train one model directly, e.g. make train-mnist_mlp\n"
	@printf "  make clean            Remove built binaries\n"

models:
	@printf "Available models:\n"
	@printf "  %s\n" $(MODELS)

run: run-$(MODEL)

train: train-$(MODEL)

clean: $(addprefix clean-,$(MODELS))
	@rmdir bin 2>/dev/null || true

define MODEL_RULES
$(1): bin/$(1)

bin/$(1): $$($(1)_SRCS) | bin
	$$(CC) $$(CFLAGS) $$($(1)_SRCS) $$(LDLIBS) -o $$@

run-$(1): bin/$(1)
	./bin/$(1) $$(INDEX)

train-$(1): bin/$(1)
	./bin/$(1) $$(TRAIN_ARG)

clean-$(1):
	rm -f bin/$(1)
endef

$(foreach model,$(MODELS),$(eval $(call MODEL_RULES,$(model))))

bin:
	mkdir -p bin
