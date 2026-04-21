FROM vllm/vllm-openai:nightly

WORKDIR /opt/gemma4-dflash-spark-vllm

COPY LICENSE README.md ./
COPY docker ./docker
COPY scripts ./scripts

RUN chmod +x docker/*.sh scripts/*.sh scripts/*.py \
 && python3 -m pip install --no-input datasets

WORKDIR /workspace

ENTRYPOINT ["/opt/gemma4-dflash-spark-vllm/docker/entrypoint.sh"]
CMD ["dflash"]
