from setuptools import setup, find_packages

with open('requirements.txt') as f:
    requirements = f.read().splitlines()

setup(
    name='lora_confirm',
    version='0.0.1',
    packages=find_packages(),
    install_requires=requirements,
    description='Code used to produce the results in the ConFIRM paper, and follow up works.',
    author='William Gazeley',
    author_email='william@asklora.ai',
    url='https://github.com/WilliamGazeley/ConFIRM',
    classifiers=[
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3.11',
    ],
)
