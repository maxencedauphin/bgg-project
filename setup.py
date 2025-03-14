from setuptools import find_packages
from setuptools import setup

with open("requirements.txt") as f:
    content = f.readlines()
requirements = [x.strip() for x in content if "git+" not in x]

setup(name='bgg-project',
      version="0.0.1",
      description="BGG project - Batch #1835 Data Science Flex",
      license="MIT",
      author="Maxence Dauphin, Bernhard Riemer, Mónica Costa, Tahar Guenfoud, Konstantin Shapovalov",
      author_email="contact@lewagon.org",
      url="https://github.com/maxencedauphin/bgg-project",
      install_requires=requirements,
      packages=find_packages(),
      test_suite="tests",
      # include_package_data: to install data from MANIFEST.in
      include_package_data=True,
      zip_safe=False)
