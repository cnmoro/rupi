# Makefile for rupi

# Project metadata
NAME := rupi
VERSION := 0.1.0
BINARY := target/release/rupi

# Directories
BUILD_DIR := target/release
DIST_DIR := dist

# Ensure build directory exists
$(BINARY):
	cargo build --release

# Build the project
build:
	cargo build --release

# Run tests
test:
	cargo test

# Install to system directories
install:
	install -m 0755 $(BINARY) /usr/local/bin/$(NAME)

# Uninstall from system directories
uninstall:
	rm -f /usr/local/bin/$(NAME)

# Create a distribution package
dist: clean
	mkdir -p $(DIST_DIR)
	cp $(BINARY) $(DIST_DIR)/$(NAME)
	tar -czvf $(DIST_DIR)/$(NAME)-v$(VERSION)-x86_64-unknown-linux-gnu.tar.gz -C $(DIST_DIR) $(NAME)
	@echo "Distribution package created: $(DIST_DIR)/$(NAME)-v$(VERSION)-x86_64-unknown-linux-gnu.tar.gz"

# Clean build artifacts
clean:
	rm -rf $(BUILD_DIR) $(DIST_DIR)

.PHONY: build test install uninstall dist clean
