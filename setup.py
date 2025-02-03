from setuptools import setup, find_packages

setup(
    name="ai_recruiter",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "langchain>=0.1.0",
        "openai>=1.0.0",
        "chromadb>=0.4.0",
        "pytest>=7.0.0",
        "pytest-asyncio>=0.23.0",
    ],
) 