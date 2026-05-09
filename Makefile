CC = gcc
CFLAGS = -Wall -Wextra -O2 -Iinclude
LDFLAGS = -lm -lws2_32
SRC = main.c src/tensor.c src/json_utils.c src/mcp_server.c src/http_server.c src/layers/layer.c src/layers/dense.c src/layers/activation.c src/layers/dropout.c src/layers/batchnorm.c src/layers/conv.c src/layers/pooling.c src/layers/lstm.c src/layers/gru.c src/layers/flatten.c src/layers/attention.c src/layers/transformer.c src/model/model.c src/model/trainer.c src/model/save_load.c src/optimizer.c src/data_augment.c src/data_loader.c src/metrics.c src/loss.c
OBJ = $(SRC:.c=.o)
TARGET = zeki

all: $(TARGET)

$(TARGET): $(OBJ)
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	rm -f $(OBJ) $(TARGET)

.PHONY: all clean
