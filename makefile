INCLUDE := -I. -I./math -I./nn
FLAGS := -Wall -Wextra -g -Wno-parentheses -Wno-unused-parameter -Wno-override-init -fsanitize=address
LIBS := -lm

CC := gcc $(INCLUDE) $(FLAGS) $(LIBS)
CCO := $(CC) -c
CLC := ar rcs

all: libml.a example_lstm

libml.a: vec.o mat.o noise_distr.o layer.o defs.o network.o network_lstm.o training_set.o
	$(CLC) $@ $^
example_lstm: example_lstm.c libml.a
	$(CC) $< -o $@ -L. -lml

defs.o: defs.c defs.h
	$(CCO) $< -o $@

vec.o: math/vec.c math/vec.h
	$(CCO) $< -o $@
mat.o: math/mat.c math/mat.h
	$(CCO) $< -o $@
noise_distr.o: math/noise_distr.c math/noise_distr.h
	$(CCO) $< -o $@

layer.o: nn/layer.c nn/layer.h
	$(CCO) $< -o $@
network.o: nn/network.c nn/network.h nn/training_set.h nn/layer.h
	$(CCO) $< -o $@
network_lstm.o: nn/network_lstm.c nn/network_lstm.h nn/network.h
	$(CCO) $< -o $@
training_set.o: nn/training_set.c nn/training_set.h
	$(CCO) $< -o $@
