from setuptools import setup, find_packages
from typing import List

HYPEN_DOT_E = '-e .'
def get_requirements(file_path: str) -> List[str]:
    requirements = []
    with open(file_path) as file:
        requirements = [req.strip() for req in file.readlines()]
        requirements = [req for req in requirements if req]
        if HYPEN_DOT_E in requirements:
            requirements.remove(HYPEN_DOT_E)
    return requirements
setup(
    name='mlproject',
    version='0.0.1',
    author='Deep',
    author_email='deepanilmohite@gmail.com',
    packages=find_packages(),
    install_requires=get_requirements('requirements.txt')
)