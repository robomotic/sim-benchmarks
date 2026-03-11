# Makefile for MJX macOS Benchmarks
# Manages virtual environment and dependencies for MuJoCo MJX with JAX Metal support

SHELL := /bin/bash
VENV_DIR := venv
PYTHON := $(VENV_DIR)/bin/python
PIP := $(VENV_DIR)/bin/pip
PYTHON_VERSION := python3
LOGS_DIR := logs

# Color output
GREEN := \033[0;32m
YELLOW := \033[0;33m
BLUE := \033[0;34m
NC := \033[0m # No Color

.PHONY: help all venv install versions activate clean clean-all run_speed run_benchmark

# Default target
all: venv install versions
	@echo ""
	@echo "$(GREEN)✓ Setup complete!$(NC)"
	@echo "$(YELLOW)To activate the environment, run:$(NC)"
	@echo "  source $(VENV_DIR)/bin/activate"

help:
	@echo "$(BLUE)MJX macOS Benchmark Environment$(NC)"
	@echo ""
	@echo "Available targets:"
	@echo "  $(GREEN)make all$(NC)           - Create venv, install packages, and show versions (default)"
	@echo "  $(GREEN)make venv$(NC)          - Create Python virtual environment"
	@echo "  $(GREEN)make install$(NC)       - Install mujoco-mjx, jax, jaxlib, and jax-metal"
	@echo "  $(GREEN)make versions$(NC)      - Show installed package versions"
	@echo "  $(GREEN)make run_speed$(NC)     - Run speed benchmark (slow_jax.py) and save to logs/"
	@echo "  $(GREEN)make run_benchmark$(NC) - Run official MJX benchmark (testspeed.py) and save to logs/"
	@echo "  $(GREEN)make activate$(NC)      - Show activation command"
	@echo "  $(GREEN)make clean$(NC)         - Remove virtual environment"
	@echo "  $(GREEN)make clean-all$(NC)     - Remove venv and all cache directories"
	@echo ""
	@echo "$(YELLOW)After setup, activate the environment with:$(NC)"
	@echo "  source $(VENV_DIR)/bin/activate"

venv:
	@if [ -d "$(VENV_DIR)" ]; then \
		echo "$(YELLOW)Virtual environment already exists at $(VENV_DIR)$(NC)"; \
	else \
		echo "$(BLUE)Creating virtual environment...$(NC)"; \
		$(PYTHON_VERSION) -m venv $(VENV_DIR); \
		echo "$(GREEN)✓ Virtual environment created$(NC)"; \
		echo "$(BLUE)Upgrading pip...$(NC)"; \
		$(PIP) install --upgrade pip setuptools wheel; \
		echo "$(GREEN)✓ pip upgraded$(NC)"; \
	fi

install: venv
	@echo "$(BLUE)Installing JAX with Metal support...$(NC)"
	@$(PIP) install --upgrade jax jaxlib
	@echo ""
	@echo "$(BLUE)Installing JAX Metal plugin...$(NC)"
	@$(PIP) install --upgrade jax-metal
	@echo ""
	@echo "$(BLUE)Installing MuJoCo MJX...$(NC)"
	@$(PIP) install --upgrade mujoco-mjx
	@echo "$(GREEN)✓ All packages installed$(NC)"

versions: venv
	@echo ""
	@echo "$(BLUE)========================================$(NC)"
	@echo "$(BLUE)  Installed Package Versions$(NC)"
	@echo "$(BLUE)========================================$(NC)"
	@echo ""
	@echo "$(YELLOW)Python:$(NC)"
	@$(PYTHON) --version
	@echo ""
	@echo "$(YELLOW)Core Packages:$(NC)"
	@$(PIP) show mujoco-mjx 2>/dev/null | grep -E "^(Name|Version):" || echo "mujoco-mjx: Not installed"
	@echo ""
	@$(PIP) show jax 2>/dev/null | grep -E "^(Name|Version):" || echo "jax: Not installed"
	@echo ""
	@$(PIP) show jaxlib 2>/dev/null | grep -E "^(Name|Version):" || echo "jaxlib: Not installed"
	@echo ""
	@$(PIP) show jax-metal 2>/dev/null | grep -E "^(Name|Version):" || echo "jax-metal: Not installed"
	@echo ""
	@echo "$(BLUE)========================================$(NC)"
	@echo ""
	@echo "$(YELLOW)JAX Device Information:$(NC)"
	@$(PYTHON) -c "import jax; print(f'JAX version: {jax.__version__}'); print(f'Available devices: {jax.devices()}'); print(f'Default backend: {jax.default_backend()}')" 2>/dev/null || echo "$(YELLOW)Unable to query JAX (not installed or import error)$(NC)"
	@echo ""

activate:
	@echo "$(YELLOW)To activate the virtual environment, run:$(NC)"
	@echo "  source $(VENV_DIR)/bin/activate"
	@echo ""
	@echo "$(YELLOW)To deactivate later, run:$(NC)"
	@echo "  deactivate"

run_speed: venv
	@echo "$(BLUE)Running speed benchmark...$(NC)"
	@mkdir -p $(LOGS_DIR)
	@export XLA_FLAGS="--xla_backend_extra_options=xla_cpu_disable_new_fusion_emitters=true"; \
	export MUJOCO_MJX_DISABLE_WARP=1; \
	TIMESTAMP=$$(date +%Y%m%d_%H%M%S); \
	LOG_FILE="$(LOGS_DIR)/speed_benchmark_$$TIMESTAMP.log"; \
	echo "$(YELLOW)Output will be saved to: $$LOG_FILE$(NC)"; \
	echo "$(BLUE)Setting XLA_FLAGS for JAX compatibility$(NC)"; \
	echo "$(BLUE)Disabling WARP backend$(NC)"; \
	echo "========================================" | tee $$LOG_FILE; \
	echo "MJX Speed Benchmark" | tee -a $$LOG_FILE; \
	echo "Started: $$(date '+%Y-%m-%d %H:%M:%S')" | tee -a $$LOG_FILE; \
	echo "XLA_FLAGS: $$XLA_FLAGS" | tee -a $$LOG_FILE; \
	echo "MUJOCO_MJX_DISABLE_WARP: $$MUJOCO_MJX_DISABLE_WARP" | tee -a $$LOG_FILE; \
	echo "========================================" | tee -a $$LOG_FILE; \
	echo "" | tee -a $$LOG_FILE; \
	$(PYTHON) slow_jax.py 2>&1 | tee -a $$LOG_FILE; \
	EXIT_CODE=$$?; \
	echo "" | tee -a $$LOG_FILE; \
	echo "========================================" | tee -a $$LOG_FILE; \
	echo "Finished: $$(date '+%Y-%m-%d %H:%M:%S')" | tee -a $$LOG_FILE; \
	echo "Exit code: $$EXIT_CODE" | tee -a $$LOG_FILE; \
	echo "========================================" | tee -a $$LOG_FILE; \
	if [ $$EXIT_CODE -eq 0 ]; then \
		echo "$(GREEN)✓ Benchmark completed successfully$(NC)"; \
		echo "$(YELLOW)Results saved to: $$LOG_FILE$(NC)"; \
	else \
		echo "$(YELLOW)⚠ Benchmark exited with code $$EXIT_CODE$(NC)"; \
		echo "$(YELLOW)Check log file for details: $$LOG_FILE$(NC)"; \
	fi

run_benchmark: venv
	@echo "$(BLUE)Running MJX official benchmark...$(NC)"
	@mkdir -p $(LOGS_DIR)
	@export XLA_FLAGS="--xla_backend_extra_options=xla_cpu_disable_new_fusion_emitters=true"; \
	export MUJOCO_MJX_DISABLE_WARP=1; \
	TIMESTAMP=$$(date +%Y%m%d_%H%M%S); \
	MODEL=$${MODEL:-humanoid/humanoid.xml}; \
	NSTEP=$${NSTEP:-1000}; \
	BATCH_SIZE=$${BATCH_SIZE:-1024}; \
	SOLVER=$${SOLVER:-cg}; \
	LOG_FILE="$(LOGS_DIR)/mjx_benchmark_$$TIMESTAMP.log"; \
	echo "$(YELLOW)Output will be saved to: $$LOG_FILE$(NC)"; \
	echo "$(BLUE)Configuration:$(NC)"; \
	echo "  Model: $$MODEL"; \
	echo "  Steps: $$NSTEP"; \
	echo "  Batch size: $$BATCH_SIZE"; \
	echo "  Solver: $$SOLVER"; \
	echo "$(BLUE)Setting XLA_FLAGS for JAX compatibility$(NC)"; \
	echo "$(BLUE)Disabling WARP backend$(NC)"; \
	echo "========================================" | tee $$LOG_FILE; \
	echo "MJX Official Benchmark" | tee -a $$LOG_FILE; \
	echo "Started: $$(date '+%Y-%m-%d %H:%M:%S')" | tee -a $$LOG_FILE; \
	echo "Model: $$MODEL" | tee -a $$LOG_FILE; \
	echo "Steps: $$NSTEP" | tee -a $$LOG_FILE; \
	echo "Batch size: $$BATCH_SIZE" | tee -a $$LOG_FILE; \
	echo "Solver: $$SOLVER" | tee -a $$LOG_FILE; \
	echo "XLA_FLAGS: $$XLA_FLAGS" | tee -a $$LOG_FILE; \
	echo "MUJOCO_MJX_DISABLE_WARP: $$MUJOCO_MJX_DISABLE_WARP" | tee -a $$LOG_FILE; \
	echo "========================================" | tee -a $$LOG_FILE; \
	echo "" | tee -a $$LOG_FILE; \
	$(PYTHON) testspeed.py --mjcf=$$MODEL --nstep=$$NSTEP --batch_size=$$BATCH_SIZE --solver=$$SOLVER 2>&1 | tee -a $$LOG_FILE; \
	EXIT_CODE=$$?; \
	echo "" | tee -a $$LOG_FILE; \
	echo "========================================" | tee -a $$LOG_FILE; \
	echo "Finished: $$(date '+%Y-%m-%d %H:%M:%S')" | tee -a $$LOG_FILE; \
	echo "Exit code: $$EXIT_CODE" | tee -a $$LOG_FILE; \
	echo "========================================" | tee -a $$LOG_FILE; \
	if [ $$EXIT_CODE -eq 0 ]; then \
		echo "$(GREEN)✓ Benchmark completed successfully$(NC)"; \
		echo "$(YELLOW)Results saved to: $$LOG_FILE$(NC)"; \
	else \
		echo "$(YELLOW)⚠ Benchmark exited with code $$EXIT_CODE$(NC)"; \
		echo "$(YELLOW)Check log file for details: $$LOG_FILE$(NC)"; \
	fi

clean:
	@echo "$(YELLOW)Removing virtual environment...$(NC)"
	@rm -rf $(VENV_DIR)
	@echo "$(GREEN)✓ Virtual environment removed$(NC)"

clean-all: clean
	@echo "$(YELLOW)Removing Python cache directories...$(NC)"
	@find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	@find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	@find . -type f -name "*.pyc" -delete 2>/dev/null || true
	@find . -type f -name "*.pyo" -delete 2>/dev/null || true
	@echo "$(GREEN)✓ All cleanup complete$(NC)"

# Check if environment is activated
check-venv:
	@if [ -z "$$VIRTUAL_ENV" ]; then \
		echo "$(YELLOW)⚠ Virtual environment is not activated$(NC)"; \
		echo "$(YELLOW)Run: source $(VENV_DIR)/bin/activate$(NC)"; \
		exit 1; \
	fi
