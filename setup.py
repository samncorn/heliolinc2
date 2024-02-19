from pybind11.setup_helpers import Pybind11Extension, build_ext
from setuptools import setup
import os

os.environ['CC'] = 'g++-13'
os.environ["CXX"] = "g++-13"

__version__ = "0.0.1"

ext_modules = [
    Pybind11Extension(
        "heliohypy.heliohypy",
        ["heliohypy/heliohypy.cpp", "src/solarsyst_dyn_geo01.cpp"],
        define_macros=[("VERSION_INFO", __version__)],
        # extra_compile_args=["-O3", "-Wall", "-shared", ],
        extra_link_args=["-undefined", "dynamic_lookup", "-fopenmp"],
    )
]

setup(
    name="heliohypy",
    version=__version__,
    # author=""
    description="Python bindings for heliolinc",
    ext_modules=ext_modules,
    packages=['src', 'heliohypy'],
    # extras_require
    # cmd_class={"build_ext": build_ext},
    cxx_std=17,
    zip_safe=False,
    python_requires=">=3.11",
)