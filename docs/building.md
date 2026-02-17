# Building from Source

This guide covers building AlphaFast from source for development or non-containerized deployment.

## Prerequisites

> **Note:** Building from source is only supported on **Linux** (x86_64 or aarch64). macOS and Windows are not supported.

- **Python** >= 3.12
- **CUDA** 12.6 (toolkit and compatible driver >= 560.28.03)
- **C++20 compiler**: gcc/g++ >= 10 (for pybind11 C++ extensions)
- **CMake** >= 3.28
- **git**
- **UV** package manager

### Installing UV

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

## Build Steps

### 1. Clone the Repository

```bash
git clone https://github.com/RomeroLab/alphafast.git
cd alphafast
```

### 2. Install Dependencies

UV reads `pyproject.toml` and `uv.lock` to install all Python dependencies and compile the C++ extensions:

```bash
uv sync --frozen --all-groups
```

This step:

- Creates a virtual environment
- Installs all Python dependencies (JAX, NumPy, RDKit, etc.)
- Compiles the C++ pybind11 extension via scikit-build-core and CMake

### 3. Build Chemical Components Database

The chemical components database is a set of pickled data files required before running inference:

```bash
uv run build_data
```

This command is provided as a script entry point defined in `pyproject.toml` and uses the `cifpp` and `dssp` C++ libraries built during the previous step.

### 4. Install MMseqs2-GPU and Foldseek

Download pre-built binaries:

```bash
# MMseqs2 with GPU support
wget https://mmseqs.com/latest/mmseqs-linux-gpu.tar.gz
tar xzf mmseqs-linux-gpu.tar.gz
cp mmseqs/bin/mmseqs ~/.local/bin/

# Foldseek with GPU support (optional, for foldseek template modes)
wget https://mmseqs.com/foldseek/foldseek-linux-gpu.tar.gz
tar xzf foldseek-linux-gpu.tar.gz
cp foldseek/bin/foldseek ~/.local/bin/
```

Ensure `~/.local/bin` is in your `PATH`, or pass the binary paths explicitly with `--mmseqs_binary_path` and `--foldseek_binary_path`.

## Build System Details

AlphaFast uses **scikit-build-core** as its build backend, which invokes CMake to compile C++ pybind11 extensions. The relevant files:

- **`pyproject.toml`**: Declares build dependencies (`scikit_build_core`, `pybind11`, `cmake`, `ninja`, `numpy`) and project metadata.
- **`CMakeLists.txt`**: Defines the C++ build. Uses `FetchContent` to download and build abseil-cpp, pybind11, pybind11_abseil, libcifpp, and dssp. Compiles all `.cc` files under `src/alphafold3/` into a single `cpp` pybind11 module.
- **`src/alphafold3/cpp.cc`**: The pybind11 entry point that exposes C++ functions to Python.

The C++20 standard is required (`CMAKE_CXX_STANDARD 20`).

## Docker Build

To build the Docker image locally:

```bash
docker build -f docker/Dockerfile -t romerolabduke/alphafast .
```

The Dockerfile:

1. Starts from `nvidia/cuda:12.6.3-base-ubuntu24.04`
2. Installs system dependencies (Python 3.12, gcc, g++, git)
3. Installs UV and creates a virtual environment
4. Downloads MMseqs2-GPU and Foldseek binaries
5. Installs Python dependencies via `uv sync`
6. Builds the chemical components database via `uv run build_data`
7. Sets `XLA_FLAGS` and `XLA_CLIENT_MEM_FRACTION` environment variables

## Running Tests

```bash
uv run pytest
```

## Model Weights

Model weights (`af3.bin.zst`) must be obtained separately from Google DeepMind. Request access via [this form](https://forms.gle/svvpY4u2jsHEwWYS6). Place the file in your `--weights_dir` / `--model_dir` directory. The weights are subject to the [AlphaFold 3 Model Parameters Terms of Use](https://github.com/google-deepmind/alphafold3/blob/main/WEIGHTS_TERMS_OF_USE.md).
