from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name='BabyTorch',
    version='0.3.0',
    author="Amjad Yousef Majid",
    author_email="amjad.y.majid@gmail.com",
    description=("A tiny, readable deep learning framework for education. "
                 "BabyTorch mirrors the PyTorch API and runs on CPU (NumPy) "
                 "or GPU (CuPy), scaling all the way up to a small GPT."),
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/amjadmajid/BabyTorch",
    packages=find_packages(),
    # NumPy is the only hard requirement -- BabyTorch runs on any machine.
    install_requires=[
        'numpy',
    ],
    # Optional extras:
    #   pip install -e ".[gpu]"   -> GPU acceleration via CuPy (CUDA 12.x)
    #   pip install -e ".[viz]"   -> loss curves and computation-graph drawing
    #   pip install -e ".[dev]"   -> everything plus the test runner
    extras_require={
        'gpu': ['cupy-cuda12x'],
        'viz': ['matplotlib', 'graphviz'],
        'dev': ['pytest', 'matplotlib', 'graphviz'],
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Intended Audience :: Education",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Topic :: Education",
        "Development Status :: 4 - Beta",
    ],
    python_requires='>=3.8',
    keywords='education deep-learning pytorch neural-networks transformer llm gpt',
)
