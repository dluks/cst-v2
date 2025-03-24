#!/bin/bash

# Export dependencies to requirements.txt
poetry export --format=requirements.txt --output=requirements.txt --without-hashes

# Also export dev dependencies to requirements-dev.txt
poetry export --format=requirements.txt --output=requirements-dev.txt --with dev --without-hashes

# Print the requirements for verification
echo "Exported dependencies to requirements.txt:"
cat requirements.txt
echo -e "\nExported dev dependencies to requirements-dev.txt:"
cat requirements-dev.txt 