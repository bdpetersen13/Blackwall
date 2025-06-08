from setuptools import setup, find_packages

# Read requirements from requirements.txt
with open("requirements.txt") as f:
    requirements = [line.strip() for line in f if line.strip() and not line.startswith("#")]

# Read long description from README if it exists
long_description = ""
try:
    with open("README.md", "r", encoding = "utf-8") as fh:
        long_description = fh.read()
except FileNotFoundError:
    pass

setup(
    name = "blackwall",
    version = "0.1.0",
    author = "Brandon Petersen",
    author_email = "petersen.brandon@sudomail.com", 
    description = "Blackwall is a GenAI detection tool running on cli",
    long_description = long_description,
    long_description_content_type = "text/markdown",
    url = "https://github.com/bdpetersen13/Blackwall",
    packages = find_packages(where = "src"),
    package_dir = {"": "src"},
    python_requires = ">=3.10",
    install_requires = requirements,
    entry_points = {
        "console_scripts": [
            "blackwall=blackwall.cli:main",
            "blackwall-batch=blackwall.cli:batch_process",
        ],
    },
    classifiers = [
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Operating System :: OS Independent",
    ],
    include_package_data = True,
    zip_safe = False,
)