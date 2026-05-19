"""Build script for ai-sdk-cpp Python bindings."""

import os
import sys
from setuptools import setup, Extension, find_packages
from setuptools.command.build_ext import build_ext

class CMakeExtension(Extension):
    def __init__(self, name, sourcedir=""):
        Extension.__init__(self, name, sources=[])
        self.sourcedir = os.path.abspath(sourcedir)

class CMakeBuild(build_ext):
    def build_extension(self, ext):
        import subprocess

        extdir = os.path.abspath(os.path.dirname(self.get_ext_fullpath(ext.name)))
        cfg = "Release"

        cmake_args = [
            f"-DCMAKE_LIBRARY_OUTPUT_DIRECTORY={extdir}",
            f"-DPYTHON_EXECUTABLE={sys.executable}",
            f"-DCMAKE_BUILD_TYPE={cfg}",
            "-DAI_SDK_BUILD_PYTHON=ON",
            "-DAI_SDK_BUILD_TESTS=OFF",
            "-DAI_SDK_BUILD_EXAMPLES=OFF",
            "-DAI_SDK_BUILD_CLI=OFF",
        ]

        build_dir = os.path.join(self.build_temp, ext.name)
        os.makedirs(build_dir, exist_ok=True)

        subprocess.check_call(["cmake", ext.sourcedir] + cmake_args, cwd=build_dir)
        subprocess.check_call(["cmake", "--build", ".", "--config", cfg, "-j"], cwd=build_dir)

setup(
    name="ai-sdk-cpp",
    version="0.1.0",
    author="AI SDK Contributors",
    description="Native C++ AI agent framework with Python bindings",
    long_description=open("README.md").read() if os.path.exists("README.md") else "",
    long_description_content_type="text/markdown",
    packages=find_packages(),
    ext_modules=[CMakeExtension("ai_sdk._native", sourcedir="../..")],
    cmdclass={"build_ext": CMakeBuild},
    python_requires=">=3.9",
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: C++",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
    ],
)
