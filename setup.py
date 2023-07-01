from setuptools import setup
import os

VERSION = "0.1"


def get_long_description():
    with open(
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "README.md"),
        encoding="utf8",
    ) as fp:
        return fp.read()


setup(
    name="llm-palm",
    description="Plugin for LLM adding support for Google's PaLM 2 model",
    long_description=get_long_description(),
    long_description_content_type="text/markdown",
    author="Simon Willison",
    url="https://github.com/simonw/llm-palm",
    project_urls={
        "Issues": "https://github.com/simonw/llm-palm/issues",
        "CI": "https://github.com/simonw/llm-palm/actions",
        "Changelog": "https://github.com/simonw/llm-palm/releases",
    },
    license="Apache License, Version 2.0",
    classifiers=["License :: OSI Approved :: Apache Software License"],
    version=VERSION,
    packages=["llm_palm"],
    entry_points={"llm": ["llm_palm = llm_palm"]},
    install_requires=["llm", "google-generativeai"],
    extras_require={"test": ["pytest", "pytest-asyncio"]},
    python_requires=">=3.7",
)
