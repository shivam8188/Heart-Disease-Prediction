from setuptools import setup, find_packages
from typing import List

HYPEN_E_DOT = "-e ."


def get_requirements(file_path: str) -> List[str]:
    with open(file_path) as f:
        reqs = [r.strip() for r in f.readlines()]
        reqs = [r for r in reqs if r and r != HYPEN_E_DOT]
    return reqs


setup(
    name="HeartDiseasePrediction",
    version="2.0.0",
    author="Your Name",
    author_email="your@email.com",
    description="ML-powered heart disease risk prediction web app",
    packages=find_packages(),
    install_requires=get_requirements("requirements.txt"),
    python_requires=">=3.9",
)
