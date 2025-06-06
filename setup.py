from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="blackwall",
    version="0.1.0",
    author="Brandon Petersen",
    author_email="petersen.brandon@sudomail.com",
    description="Blackwall - A GenAI detection tool for text, images, and videos",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/bdpetersen13/blackwall",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Anyone",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.10",
    install_requires=[
        # List from requirements.txt
    ],
    entry_points={
        "console_scripts": [
            "blackwall=blackwall.__main__:main",
        ],
    },
)