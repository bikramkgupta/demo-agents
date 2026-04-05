#!/bin/sh
# Write Plano config from base64-encoded env var
if [ -n "$PLANO_CONFIG_B64" ]; then
  echo "$PLANO_CONFIG_B64" | base64 -d > /app/plano_config.yaml
  echo "Wrote plano_config.yaml from PLANO_CONFIG_B64"
  cat /app/plano_config.yaml
fi
# Start supervisord (the original Plano entrypoint)
exec /usr/local/bin/supervisord -c /etc/supervisor/conf.d/supervisord.conf
