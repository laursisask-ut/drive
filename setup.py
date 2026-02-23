# Debug command:
#    DEBUG=1 python setup.py build_ext --inplace --force
#    CUDA_VISIBLE_DEVICES=None LD_PRELOAD=$(gcc -print-file-name=libasan.so) python3.12 -m pufferlib.clean_pufferl eval --train.device cpu

from setuptools import find_packages, find_namespace_packages, setup, Extension
import numpy
import os
import glob
import json
import urllib.request
import zipfile
import tarfile
import platform
import shutil
import shlex
import sys
from pathlib import Path

from setuptools.command.build_ext import build_ext
from torch.utils import cpp_extension
from torch.utils.cpp_extension import (
    CppExtension,
    CUDAExtension,
)

# Build with DEBUG=1 to enable debug symbols
DEBUG = os.getenv("DEBUG", "0") == "1"
NO_OCEAN = os.getenv("NO_OCEAN", "0") == "1"
NO_TRAIN = os.getenv("NO_TRAIN", "0") == "1"

# Build raylib for your platform
RAYLIB_URL = "https://github.com/raysan5/raylib/releases/download/5.5/"
RAYLIB_NAME = "raylib-5.5_macos" if platform.system() == "Darwin" else "raylib-5.5_linux_amd64"
RLIGHTS_URL = "https://raw.githubusercontent.com/raysan5/raylib/refs/heads/master/examples/shaders/rlights.h"

# Fetch inih library
INIH_URL = "https://github.com/benhoyt/inih/archive/refs/tags/{tag}.{ext}"


def download_raylib(platform, ext):
    if not os.path.exists(platform):
        print(f"Downloading Raylib {platform}")
        urllib.request.urlretrieve(RAYLIB_URL + platform + ext, platform + ext)
        if ext == ".zip":
            with zipfile.ZipFile(platform + ext, "r") as zip_ref:
                zip_ref.extractall()
        else:
            with tarfile.open(platform + ext, "r") as tar_ref:
                if sys.version_info >= (3, 12):  # Use secure call when python version >= 3.12
                    tar_ref.extractall(filter="data")
                else:
                    tar_ref.extractall()

        os.remove(platform + ext)
        urllib.request.urlretrieve(RLIGHTS_URL, platform + "/include/rlights.h")


def download_library(url: str, name: str, tag: str, ext: str = "tar.gz", files_to_extract: list = None):
    library_folder = name + "-" + tag
    archive_file = library_folder + "." + ext
    if not os.path.exists(library_folder):
        filled_url = url.format(tag=tag, ext=ext)
        print(f"Downloading {name}-{tag}")
        urllib.request.urlretrieve(filled_url, archive_file)
        if ext == "zip":
            with zipfile.ZipFile(archive_file, "r") as zip_ref:
                if files_to_extract:
                    members = [
                        member_info.filename
                        for member_info in zip_ref.infolist()
                        if os.path.basename(member_info.filename) in files_to_extract
                    ]
                    zip_ref.extractall(members=members)
                else:
                    zip_ref.extractall()
        else:
            with tarfile.open(archive_file, "r") as tar_ref:
                kwargs = {}
                if sys.version_info >= (3, 12):
                    kwargs["filter"] = "data"
                if files_to_extract:
                    kwargs["members"] = [
                        member for member in tar_ref.getmembers() if os.path.basename(member.name) in files_to_extract
                    ]
                tar_ref.extractall(**kwargs)
        os.remove(archive_file)


if not NO_OCEAN:
    download_library(INIH_URL, "inih", "r62", files_to_extract=["ini.c", "ini.h"])
    download_raylib("raylib-5.5_webassembly", ".zip")

BOX2D_URL = "https://github.com/capnspacehook/box2d/releases/latest/download/"
BOX2D_NAME = "box2d-macos-arm64" if platform.system() == "Darwin" else "box2d-linux-amd64"


def download_box2d(platform):
    if not os.path.exists(platform):
        ext = ".tar.gz"

        print(f"Downloading Box2D {platform}")
        urllib.request.urlretrieve(BOX2D_URL + platform + ext, platform + ext)
        with tarfile.open(platform + ext, "r") as tar_ref:
            if sys.version_info >= (3, 12):
                tar_ref.extractall(filter="data")
            else:
                tar_ref.extractall()

        os.remove(platform + ext)


if not NO_OCEAN:
    download_box2d("box2d-web")

COMPILE_COMMANDS = []
COMPILE_SOURCES = (".c", ".cc", ".cpp", ".cxx", ".cu", ".cuh")


def record_compile_command(cmd):
    source = next((part for part in cmd if part.endswith(COMPILE_SOURCES)), None)
    if source is None:
        return

    COMPILE_COMMANDS.append(
        {
            "directory": os.getcwd(),
            "command": " ".join(shlex.quote(part) for part in cmd),
            "file": os.path.abspath(source),
        }
    )


def write_compile_commands(path):
    if not COMPILE_COMMANDS:
        print("No compile commands captured; skipping compile_commands.json")
        return

    entries = []
    seen = set()
    for entry in COMPILE_COMMANDS:
        key = (entry["file"], entry["command"])
        if key in seen:
            continue
        seen.add(key)
        entries.append(entry)

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(entries, handle, indent=2)
    print(f"Wrote {len(entries)} commands to {output_path}")

# Shared compile args for all platforms
extra_compile_args = [
    "-DNPY_NO_DEPRECATED_API=NPY_1_7_API_VERSION",
    "-DPLATFORM_DESKTOP",
]
extra_link_args = ["-fwrapv"]
cxx_args = [
    "-fdiagnostics-color=always",
]
nvcc_args = []

if DEBUG:
    extra_compile_args += [
        "-O0",
        "-g",
        "-fsanitize=address,undefined,bounds,pointer-overflow,leak",
        "-fno-omit-frame-pointer",
    ]
    extra_link_args += [
        "-g",
        "-fsanitize=address,undefined,bounds,pointer-overflow,leak",
    ]
    cxx_args += [
        "-O0",
        "-g",
    ]
    nvcc_args += [
        "-O0",
        "-g",
    ]
else:
    extra_compile_args += [
        "-O2",
        "-flto",
    ]
    extra_link_args += [
        "-O2",
    ]
    cxx_args += [
        "-O3",
    ]
    nvcc_args += [
        "-O3",
    ]

system = platform.system()
if system == "Linux":
    extra_compile_args += [
        "-Wno-alloc-size-larger-than",
        "-Wno-implicit-function-declaration",
        "-fmax-errors=3",
    ]
    extra_link_args += [
        "-Bsymbolic-functions",
    ]
    if not NO_OCEAN:
        download_raylib("raylib-5.5_linux_amd64", ".tar.gz")
elif system == "Darwin":
    extra_compile_args += [
        "-Wno-error=int-conversion",
        "-Wno-error=incompatible-function-pointer-types",
        "-Wno-error=implicit-function-declaration",
    ]
    extra_link_args += [
        "-framework",
        "Cocoa",
        "-framework",
        "OpenGL",
        "-framework",
        "IOKit",
    ]
    if not NO_OCEAN:
        download_raylib("raylib-5.5_macos", ".tar.gz")
else:
    raise ValueError(f"Unsupported system: {system}")

if not NO_OCEAN:
    download_box2d(BOX2D_NAME)

# Default Gym/Gymnasium/PettingZoo versions
# Gym:
# - 0.26 still has deprecation warnings and is the last version of the package
# - 0.25 adds a breaking API change to reset, step, and render_modes
# - 0.24 is broken
# - 0.22-0.23 triggers deprecation warnings by calling its own functions
# - 0.21 is the most stable version
# - <= 0.20 is missing dict methods for gym.spaces.Dict
# - 0.18-0.21 require setuptools<=65.5.0


# Extensions
class CompileCommandsMixin:
    compile_commands_user_options = [
        ("generate-compile-commands", None, "Generate compile_commands.json from the build"),
        ("compile-commands-path=", None, "Output path for compile_commands.json (default: compile_commands.json)"),
    ]
    compile_commands_boolean_options = ["generate-compile-commands"]

    def initialize_options(self):
        super().initialize_options()
        self.generate_compile_commands = False
        self.compile_commands_path = None

    def finalize_options(self):
        super().finalize_options()
        if self.compile_commands_path is None:
            self.compile_commands_path = os.path.abspath("compile_commands.json")

    def _enable_compile_commands_capture(self):
        if not getattr(self, "generate_compile_commands", False):
            return

        compiler = getattr(self, "compiler", None)
        if compiler is None or getattr(compiler, "_compile_commands_wrapped", False):
            return

        original_spawn = compiler.spawn

        def capture(cmd, **kwargs):
            record_compile_command(cmd)
            return original_spawn(cmd, **kwargs)

        compiler.spawn = capture
        compiler._compile_commands_wrapped = True


class BuildExt(CompileCommandsMixin, build_ext):
    user_options = build_ext.user_options + CompileCommandsMixin.compile_commands_user_options
    boolean_options = build_ext.boolean_options + CompileCommandsMixin.compile_commands_boolean_options

    def run(self):
        if self.generate_compile_commands:
            COMPILE_COMMANDS.clear()

        # Propagate any build_ext options (e.g., --inplace, --force) to subcommands
        build_ext_opts = self.distribution.command_options.get("build_ext", {})
        if build_ext_opts:
            # Copy flags so build_torch and build_c respect inplace/force
            self.distribution.command_options["build_torch"] = build_ext_opts.copy()
            self.distribution.command_options["build_c"] = build_ext_opts.copy()

        # Run the torch and C builds (which will handle copying when inplace is set)
        self.run_command("build_torch")
        self.run_command("build_c")

        if self.generate_compile_commands:
            write_compile_commands(self.compile_commands_path)


class CBuildExt(CompileCommandsMixin, build_ext):
    user_options = build_ext.user_options + CompileCommandsMixin.compile_commands_user_options
    boolean_options = build_ext.boolean_options + CompileCommandsMixin.compile_commands_boolean_options

    def run(self, *args, **kwargs):
        self.extensions = [e for e in self.extensions if e.name != "pufferlib._C"]
        super().run(*args, **kwargs)

    def build_extensions(self):
        self._enable_compile_commands_capture()
        super().build_extensions()


class TorchBuildExt(CompileCommandsMixin, cpp_extension.BuildExtension):
    user_options = cpp_extension.BuildExtension.user_options + CompileCommandsMixin.compile_commands_user_options
    boolean_options = cpp_extension.BuildExtension.boolean_options + CompileCommandsMixin.compile_commands_boolean_options

    def run(self):
        self.extensions = [e for e in self.extensions if e.name == "pufferlib._C"]
        super().run()

    def build_extensions(self):
        self._enable_compile_commands_capture()
        super().build_extensions()


RAYLIB_A = f"{RAYLIB_NAME}/lib/libraylib.a"
INCLUDE = [numpy.get_include(), "raylib/include", f"{BOX2D_NAME}/include", f"{BOX2D_NAME}/src"]
extension_kwargs = dict(
    include_dirs=INCLUDE,
    extra_compile_args=extra_compile_args,
    extra_link_args=extra_link_args,
    extra_objects=[RAYLIB_A],
    libraries=["lz4"],
)

# Find C extensions
c_extensions = []
if not NO_OCEAN:
    c_extension_paths = glob.glob("pufferlib/ocean/**/binding.c", recursive=True)
    c_extensions = [
        Extension(
            path.rstrip(".c").replace("/", "."),
            sources=[path],
            **extension_kwargs,
        )
        for path in c_extension_paths
        if "matsci" not in path
    ]
    c_extension_paths = [os.path.join(*path.split("/")[:-1]) for path in c_extension_paths]

    for c_ext in c_extensions:
        if "drive" in c_ext.name:
            c_ext.sources.append("inih-r62/ini.c")
            c_ext.extra_compile_args.extend(
                [
                    '-DINI_START_COMMENT_PREFIXES="#"',
                    '-DINI_INLINE_COMMENT_PREFIXES="#"',
                ]
            )

        if "impulse_wars" in c_ext.name:
            print(f"Adding {c_ext.name} to extra objects")
            c_ext.extra_objects.append(f"{BOX2D_NAME}/libbox2d.a")

        if "matsci" in c_ext.name:
            c_ext.include_dirs.append("/usr/local/include")
            c_ext.extra_link_args.extend(["-L/usr/local/lib", "-llammps"])

# Check if CUDA compiler is available. You need cuda dev, not just runtime.
torch_extensions = []
if not NO_TRAIN:
    torch_sources = [
        "pufferlib/extensions/pufferlib.cpp",
    ]
    if shutil.which("nvcc"):
        extension = CUDAExtension
        torch_sources.append("pufferlib/extensions/cuda/pufferlib.cu")
    else:
        extension = CppExtension

    torch_extensions = [
        extension(
            "pufferlib._C",
            torch_sources,
            extra_compile_args={
                "cxx": cxx_args,
                "nvcc": nvcc_args,
            },
        ),
    ]

# Prevent Conda from injecting garbage compile flags
from distutils.sysconfig import get_config_vars

cfg_vars = get_config_vars()
for key in ("CC", "CXX", "LDSHARED"):
    if cfg_vars[key]:
        cfg_vars[key] = cfg_vars[key].replace("-B /root/anaconda3/compiler_compat", "")
        cfg_vars[key] = cfg_vars[key].replace("-pthread", "")
        cfg_vars[key] = cfg_vars[key].replace("-fno-strict-overflow", "")

for key, value in cfg_vars.items():
    if value and "-fno-strict-overflow" in str(value):
        cfg_vars[key] = value.replace("-fno-strict-overflow", "")

install_requires = [
    "setuptools",
    "numpy<2.0",
    "shimmy[gym-v21]",
    "gym==0.23",
    "gymnasium==0.29.1",
    "pettingzoo==1.24.1",
    "lanelet2==1.2.2",
    "matplotlib",
]

if not NO_TRAIN:
    install_requires += [
        "torch",
        "psutil",
        "nvidia-ml-py",
        "rich",
        "rich_argparse",
        "imageio",
        "pyro-ppl",
        "heavyball<2.0.0",
        "neptune",
        "wandb",
    ]

setup(
    version="3.0.0",
    packages=find_namespace_packages() + find_packages() + c_extension_paths + ["pufferlib/extensions"],
    package_data={"pufferlib": [RAYLIB_NAME + "/lib/libraylib.a"]},
    include_package_data=True,
    install_requires=install_requires,
    ext_modules=c_extensions + torch_extensions,
    cmdclass={
        "build_ext": BuildExt,
        "build_torch": TorchBuildExt,
        "build_c": CBuildExt,
    },
    include_dirs=[numpy.get_include(), RAYLIB_NAME + "/include"],
)
