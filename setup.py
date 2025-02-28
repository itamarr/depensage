"""
DepenSage - Expense Tracking with Neural Classification

Setup script for installing the DepenSage package.
"""

from setuptools import setup, find_packages

setup(
    name="depensage",
    version="0.1.0",
    description="Expense Tracking with Neural Classification",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    author="Itamar Rosenfeld Rauch",
    packages=find_packages(),
    python_requires=">=3.7",
    install_requires=[
        "pandas>=1.0.0",
        "numpy>=1.18.0",
        "tensorflow>=2.4.0",
        "scikit-learn>=0.23.0",
        "google-api-python-client>=2.0.0",
        "google-auth-httplib2>=0.1.0",
        "google-auth-oauthlib>=0.4.0",
    ],
    entry_points={
        "console_scripts": [
            "depensage=depensage.main:main",
        ],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: End Users/Desktop",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
    ],
)
