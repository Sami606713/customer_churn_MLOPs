from setuptools import setup,find_packages

def get_requires(file_path):
    with open(file_path,"r")as f:
        requirements=f.readlines()
    
    requirements=[package.strip() for package in requirements ]

    return requirements

setup(
    name="Customer-Churn-Prediction",
    description="In this project i will do the customer churn prediction mean that whic customer can leave and which customer can't leave",
    author="SAMI-ULLAH",
    author_email="sami606713@gmail.com",
    packages=find_packages(),
    install_requires=get_requires('requirements.txt')
)
