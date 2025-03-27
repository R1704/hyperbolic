from setuptools import setup, find_packages

setup(
    name="hyperbolic_graph_embedding",
    version="0.1.0",
    description="Hyperbolic graph embedding models",
    author="Your Name",
    packages=find_packages(),
    include_package_data=True,
    python_requires=">=3.6",
    install_requires=[
        "numpy",
        "torch",
        "matplotlib",
        "networkx",
        "scikit-learn",
        # Add any other dependencies your project needs
    ],
)