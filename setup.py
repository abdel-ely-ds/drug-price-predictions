import setuptools
from setuptools import setup

TEST_DEPS = ["pytest==5.0.1", "pytest-runner==5.1", "pytest-cov==2.7.1", "nox"]

with open("requirements.txt") as f:
    requirements = f.readlines()

setup(
    name="drugs-price",
    version="0.1.0",
    description="Predicting drugs prices",
    keywords=["regression", "drugs"],
    author="abdel.ely.ds",
    license="MIT",
    classifiers=["Programming Language :: Python :: 3.7"],
    zip_safe=True,
    include_package_data=True,
    packages=setuptools.find_packages("src"),
    package_dir={"": "src"},
    entry_points={"console_scripts": ["drugs-price=drugs.cli.cli:main"]},
    install_requires=requirements,
    tests_require=TEST_DEPS,
    extras_require={"test": TEST_DEPS},
)
