# setup.py
from setuptools import setup, find_packages

setup(
    name="intelligence-analytics-platform",
    version="1.0.0",
    description="AI-powered intelligence analytics platform with location extraction",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.8",
    install_requires=[
        "fastapi>=0.104.1",
        "uvicorn>=0.24.0",
        "redis>=5.0.1",
        "clickhouse-connect>=0.6.14",
        "pydantic>=2.5.0",
        "PyYAML>=6.0.1",
        "requests>=2.31.0",
        "geohash2>=1.1",
        "openai>=1.3.5",
        "anthropic>=0.7.7",
        "python-dotenv>=1.0.0",
        "pandas>=2.0.0",
    ],
    entry_points={
        "console_scripts": [
            "intel-platform=main:main",
        ],
    },
)