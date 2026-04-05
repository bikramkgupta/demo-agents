# Plano router wrapper — decodes config from PLANO_CONFIG_B64 env var
FROM katanemo/plano:latest
COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh
ENTRYPOINT ["/entrypoint.sh"]
