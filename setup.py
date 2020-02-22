from setuptools import setup, find_packages



with open('README.md') as f:
    readme = f.read()

with open('LICENSE') as f:
    license = f.read()

install_requires = ['mordred','scikit-learn','rdkit']


pkgs = find_packages(exclude=('example', 'docs', 'data'))

setup(
    name="orcband",
    version="1.0.0",
    description="predict organic bandgap",
    long_description=readme,
    author="Yuhuan Meng, Liang Xu, Zhi Peng, Hongbo,Qiao",
    author_email="xuliang1@uw.edu",
    url="https://github.com/xuliang2019/Orc_Band.git",
    license= license,
    packages=pkgs,
    install_requires=install_requires
)
