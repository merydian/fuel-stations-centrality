from setuptools import setup, Extension
from pybind11.setup_helpers import Pybind11Extension, build_ext
from pybind11 import get_cmake_dir
import pybind11

ext_modules = [
    Pybind11Extension(
        "stats_cpp",
        [
            "stats_pybind.cpp",
            "stats.cpp",
            # Add other required source files:
            # "graph.cpp",
            # "config.cpp", 
            # "centrality.cpp",
        ],
        include_dirs=[
            # Path to pybind11 headers
            pybind11.get_include(),
        ],
        language='c++'
    ),
]

setup(
    name="fuel_stations_stats",
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
    zip_safe=False,
    python_requires=">=3.6",
)
