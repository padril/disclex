.SUFFIXES:
.DELETE_ON_ERROR:
.SECONDARY:
SHELL := /bin/bash
.SHELLFLAGS := -O globstar -c

# we want to use the C++2X std for this project,
# but openfst is c++17 only (uses features deprecated in c++20)
# so much for backward compatibility!
CXX := g++ -std=c++17
# -O2 becaues Eigen is very slow without any optimization
CXXFLAGS := -Wall -Wextra -fmax-errors=5
LDFLAGS := -L/home/padril/libs/lib/ -lfst -lngram

BUILD := build
OBJ_DIR := $(BUILD)/objects
APP_DIR := $(BUILD)/bin

SRC_DIR := src
BIN_DIR := bin
INCLUDE_DIR := include

SRC := $(sort $(wildcard $(SRC_DIR)/*.cpp))
BIN := $(sort $(wildcard $(BIN_DIR)/*.cpp))
APPS := $(BIN:$(BIN_DIR)/%.cpp=$(APP_DIR)/%)
INCLUDE := -I$(INCLUDE_DIR) -isystem /home/padril/libs/include/
OBJECTS := $(SRC:%.cpp=$(OBJ_DIR)/%.o)
DEPENDS := $(OBJECTS:%.o=%.d)
HEADERS := $(sort $(wildcard include/*.hpp))

.PHONY: debug
debug: CXXFLAGS += -DDEBUG -g -Wpedantic
debug: build $(APPS)

.PHONY: release
release: CXXFLAGS += -O3
release: build $(APPS)

.PHONY: build
build:
	mkdir -p $(BUILD)
	mkdir -p $(OBJ_DIR)
	mkdir -p $(APP_DIR)

$(APP_DIR)/%: $(OBJ_DIR)/$(BIN_DIR)/%.o $(OBJECTS)
	mkdir -p $(@D)
	$(CXX) $(CXXFLAGS) $(INCLUDE) -o $@ $^ $(LDFLAGS)

-include $(DEPENDS)

$(OBJ_DIR)/%.o: %.cpp
	mkdir -p $(@D)
	$(CXX) $(CXXFLAGS) $(INCLUDE) -c $< -MMD -o $@

.PHONY: clean
clean:
	rm $(OBJ_DIR)/**/*.o
	rm $(OBJ_DIR)/**/*.d
	rm $(APPS)
	find $(BUILD) -type d -empty -delete

