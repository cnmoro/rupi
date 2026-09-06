# Makefile for rupi

# Project metadata
NAME := rupi
VERSION := $(shell sed -n 's/^version = "\(.*\)"/\1/p' Cargo.toml | head -n 1)
BINARY := target/release/rupi

# Directories
BUILD_DIR := target/release
DIST_DIR := dist

# Ensure build directory exists
$(BINARY):
	cargo build --locked --release

# Build the project
build:
	cargo build --locked --release

# Run tests
test:
	cargo test --locked --all-targets

# Install to system directories
install: build
	install -m 0755 $(BINARY) /usr/local/bin/$(NAME)

# Uninstall from system directories
uninstall:
	rm -f /usr/local/bin/$(NAME)

# Create a distribution package
dist: build
	rm -rf $(DIST_DIR)
	mkdir -p $(DIST_DIR)
	cp $(BINARY) $(DIST_DIR)/$(NAME)
	tar -czvf $(DIST_DIR)/$(NAME)-v$(VERSION)-x86_64-unknown-linux-gnu.tar.gz -C $(DIST_DIR) $(NAME)
	@echo "Distribution package created: $(DIST_DIR)/$(NAME)-v$(VERSION)-x86_64-unknown-linux-gnu.tar.gz"

# Clean build artifacts
clean:
	rm -rf $(BUILD_DIR) $(DIST_DIR)

.PHONY: build test install uninstall dist clean
