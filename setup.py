from setuptools import setup

setup(
    name='shape2seq',
    version='0.1',
    url='https://github.com/trsherborne/shape2seq',
    download_url='',
    author='Tom Sherborne',
    author_email='tr.sherborne@gmail.com',
    license='Apache-2.0',
    description='Image captioning models for Shapeworld.',
    long_description='Generate image captions for Shapeworld abstract microworlds.',
    keywords=['tensorflow', 'tf', 'imagecaptioning'],
    platforms=['linux', 'mac'],
    lib={'': 'shape2seq'},
    packages=['shape2seq'],
    install_requires=['numpy', 'pillow', 'tensorflow'],
    extras_require={
        'gpu': ['tensorflow-gpu'],
    }
)