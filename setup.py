import re
from setuptools import setup

with open("README.md", "r") as fh:
    readme = fh.read()

regex = r"^Version:\s(\d+.\d+.\d+)$"
match = re.search(regex, readme, re.MULTILINE)

if match:
    current_version = match.group(1)
else:
    raise Exception("Current version is missing from the README.md")

with open("requirements.txt", "r") as r:
    requirements = [p for p in r.read().split("\n") if p]
    dependencies = [p.split("=")[0] for p in requirements]

setup(
    version=current_version,
    install_requires=requirements,
    dependency_links=dependencies
)
