from setuptools import setup

version = {}
with open("pixel_art/version.py") as f:
    exec(f.read(), version)

setup(name='pixel_art',
      version=version['__version__'],
      description='Turn images into pixel art',
      author='Adam Spannbauer',
      author_email='spannbaueradam@gmail.com',
      url='https://github.com/AdamSpannbauer/pixel_art',
      packages=['pixel_art'],
      license='MIT',
      install_requires=[
          'numpy',
          'pandas',
          'imutils',
          'scikit-learn',
          'scipy'
      ],
      extras_require={
          'cv2': ['opencv-contrib-python >= 3.4.0']
      },
      keywords=['pixel art', 'computer vision', 'image processing', 'opencv'],
      )
