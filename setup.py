# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import logging
import re

from setuptools import setup, find_packages

PACKAGE_NAME = "weirdo"
readme_dir = os.path.dirname(__file__)
readme_path = os.path.join(readme_dir, 'README.md')

try:
    with open(readme_path, 'r') as f:
        readme_markdown = f.read()
except Exception:
    logging.warning("Failed to load %s" % readme_path)
    readme_markdown = ""

with open('%s/__init__.py' % PACKAGE_NAME, 'r') as f:
    version = re.search(
        r'^__version__\s*=\s*[\'"]([^\'"]*)[\'"]',
        f.read(),
        re.MULTILINE).group(1)

if not version:
    raise RuntimeError("Cannot find version information")

# Read requirements
with open('requirements.txt') as f:
    requirements = [
        req.strip() for req in f.read().splitlines()
        if req.strip() and not req.startswith('#')
    ]

if __name__ == '__main__':
    setup(
        name=PACKAGE_NAME,
        version=version,
        description="Metrics of immunological foreignness for candidate T-cell epitopes",
        author="Alex Rubinsteyn",
        author_email="alex.rubinsteyn@unc.edu",
        url="https://github.com/pirl-unc/%s" % PACKAGE_NAME,
        license="Apache-2.0",
        classifiers=[
            'Development Status :: 4 - Beta',
            'Environment :: Console',
            'Operating System :: OS Independent',
            'Intended Audience :: Science/Research',
            'Programming Language :: Python :: 3',
            'Programming Language :: Python :: 3.9',
            'Programming Language :: Python :: 3.10',
            'Programming Language :: Python :: 3.11',
            'Programming Language :: Python :: 3.12',
            'Topic :: Scientific/Engineering :: Bio-Informatics',
        ],
        python_requires='>=3.9',
        install_requires=requirements,
        extras_require={
            'dev': [
                'pytest>=7.0.0',
                'pytest-cov>=4.0.0',
                'pylint>=2.0.0',
                'sphinx>=5.0.0',
                'sphinx-rtd-theme>=1.0.0',
                'sphinx-autodoc-typehints>=1.0.0',
                'tqdm>=4.0.0',
            ],
            'docs': [
                'sphinx>=5.0.0',
                'sphinx-rtd-theme>=1.0.0',
                'sphinx-autodoc-typehints>=1.0.0',
            ],
        },
        long_description=readme_markdown,
        long_description_content_type='text/markdown',
        packages=find_packages(exclude=['test', 'test.*', 'examples']),
        package_data={
            PACKAGE_NAME: [
                'logging.conf',
                'matrices/*',
            ]
        },
        include_package_data=True,
        entry_points={
            'console_scripts': [
                'weirdo=weirdo.cli:run'
            ]
        },
    )
