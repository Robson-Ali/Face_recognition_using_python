import setuptools


with open("README.md", "r") as fh:
    long_description = fh.read()

requirements = [
    "dlib", "opencv-python", "numpy", "imutils", "tqdm"
]

test_requirements = [
]

setuptools.setup(
    name="FaceReco",
    version="0.1.0",
    author="Robson-Ali",
    author_email="aliye.r@yahoo.com",
    description="Face Recognition package in python",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Robson-Ali/Face_recognition_using_python/new/main",
    packages=setuptools.find_packages(),
    package_data={
        'FaceReco': ['*']
    },
    install_requires=requirements,
    license="MIT license",
    zip_safe=False,
    keywords='face_recognition',
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
