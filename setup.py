from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="klynx",
    version="0.1.0",  # Replace with actual version based on PyPI
    author="qzx",
    author_email="qzx@example.com", # Needs real email
    description="An advanced, highly customizable autonomous agent framework built on top of LangGraph and LiteLLM.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/qzx997/klynx-kit",
    packages=find_packages(),
    install_requires=[
        "langgraph>=0.2.0",
        "langchain>=0.3.0",
        "langchain-core>=0.3.0",
        "langchain-openai>=0.2.0",
        "openai>=1.0.0",
        "textual>=0.70.0",
        "chromadb>=0.4.0",
        "sentence-transformers>=2.2.0",
        "tavily-python>=0.3.0",
        "python-dotenv>=1.0.0",
        "pydantic>=2.0.0",
        "litellm",
        "playwright"
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.8',
    entry_points={
        'console_scripts': [
            'klynx-tui=klynx.tui_app:main',
        ],
    },
)
