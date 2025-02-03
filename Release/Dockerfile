FROM python:3.11

RUN apt-get update && apt-get install -y wget libgl1-mesa-glx libegl1 && rm -rf /var/lib/apt/lists/*

RUN wget https://github.com/MiniZinc/MiniZincIDE/releases/download/2.8.7/MiniZincIDE-2.8.7-bundle-linux-x86_64.tgz -O minizinc.tar.gz && \
    tar -xf minizinc.tar.gz && \
    mv MiniZincIDE-2.8.7-bundle-linux-x86_64 minizinc && \
    cp -r minizinc/bin/* /usr/local/bin && \
    cp -r minizinc/lib/* /usr/local/lib && \
    cp -r minizinc/share/* /usr/local/share && \
    rm -rf minizinc minizinc.tar.gz

WORKDIR /app

COPY . /app

RUN pip install --no-cache-dir gurobipy z3-solver minizinc pulp numpy

RUN chmod +x /app/run_model.sh

ENTRYPOINT ["/app/run_model.sh"]
