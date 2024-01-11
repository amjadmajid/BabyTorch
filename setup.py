from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name='BabyTorch',
    version='0.2.0',
    author="Amjad Yousef Majid",
    author_email="amjad.y.majid@gmail.com",
    description="A simple deep learning framework for educational purposes. BabyTorch exposes similar APIs to PyTorch to enable zero-effort transition to PyTorch.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/amjadmajid/BabyTorch",
    packages=find_packages(),
    install_requires=[
        'numpy',
        'graphviz',
        'matplotlib',
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Intended Audience :: Education",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Topic :: Education",
        "Development Status :: 3 - Alpha",
    ],
    python_requires='>=3.6',
    keywords='education deep-learning pytorch neural-networks',
)
