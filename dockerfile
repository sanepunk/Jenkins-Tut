# 1. Use the official Jenkins agent as the base
FROM jenkins/inbound-agent:latest

# 2. Switch to ROOT so we can install software
USER root

# 3. Update the OS and install Python 3, Pip, and Venv
# (We install 'python3-full' or equivalent to ensure venv works)
RUN apt-get update && apt-get install -y \
    python3 \
    python3-venv \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

# 4. Switch back to JENKINS user for security
USER jenkins