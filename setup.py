from setuptools import setup, find_packages

VERSION = '0.0.0'
DESCRIPTION = "Miro AI's Computer Vision Toolbox"
LONG_DESCRIPTION = "Miro AI's most frequently used computer vision utility functions"

setup(
    name='miro-vision-toolbox',
    version= VERSION, description= DESCRIPTION,long_description=LONG_DESCRIPTION,
    author= "John Ho", author_email="<john@miro.io>",
    packages= find_packages(),
    install_requires=[
        'numpy>=1.20.1',
        'Pillow>=8.1.2',
        'validators>=0.18.2'
    ],
    #entry_points=[dict] # if your package is intended to be ran as a CLI tool
    )
