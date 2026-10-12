# Simple Makefile for 2048 AI
# Adjust LIBTORCH_PATH to your LibTorch installation

LIBTORCH_PATH ?= /usr/local/libtorch

CXX = g++
CXXFLAGS = -std=c++17 -O3 -Wall -Wextra
CXXFLAGS += -I$(LIBTORCH_PATH)/include
CXXFLAGS += -I$(LIBTORCH_PATH)/include/torch/csrc/api/include
CXXFLAGS += $(shell pkg-config --cflags sdl2 SDL2_ttf 2>/dev/null || echo "-I/usr/include/SDL2")

LDFLAGS = -L$(LIBTORCH_PATH)/lib
LDFLAGS += -Wl,-rpath,$(LIBTORCH_PATH)/lib
LDFLAGS += $(shell pkg-config --libs sdl2 SDL2_ttf 2>/dev/null || echo "-lSDL2 -lSDL2_ttf")
LDFLAGS += -ltorch -ltorch_cpu -lc10 -lpthread

TARGET = 2048_ai
SRC = 2048_ai.cpp
COMPONENTS = game_engine.cpp training.cpp training_visualization.cpp

.PHONY: all clean run

all: $(TARGET)

$(TARGET): $(SRC) $(COMPONENTS)
	$(CXX) $(CXXFLAGS) -o $@ $(SRC) $(LDFLAGS)

clean:
	rm -f $(TARGET)

run: $(TARGET)
	./$(TARGET)

# Debug build
debug: CXXFLAGS += -g -O0 -DDEBUG
debug: $(TARGET)
