#!/bin/sh
# Entrypoint wrapper for Node-RED container.
# Injects credentialSecret from CREDENTIAL_SECRET env var into settings.js
# before starting Node-RED. This ensures credentials survive container rebuilds.

SETTINGS_FILE="/data/settings.js"

if [ -n "$CREDENTIAL_SECRET" ] && [ -f "$SETTINGS_FILE" ]; then
    # Check if credentialSecret is already set (uncommented)
    if grep -q '^\s*credentialSecret:' "$SETTINGS_FILE"; then
        # Replace existing value
        sed -i "s|^\(\s*\)credentialSecret:.*|\\1credentialSecret: \"$CREDENTIAL_SECRET\",|" "$SETTINGS_FILE"
    elif grep -q '//credentialSecret:' "$SETTINGS_FILE"; then
        # Uncomment and set value
        sed -i "s|^\(\s*\)//credentialSecret:.*|\\1credentialSecret: \"$CREDENTIAL_SECRET\",|" "$SETTINGS_FILE"
    else
        # Insert after flowFile line
        sed -i "/flowFile:/a\\    credentialSecret: \"$CREDENTIAL_SECRET\"," "$SETTINGS_FILE"
    fi
    echo "Node-RED: credentialSecret configured from environment."
fi

# Hand off to the original Node-RED entrypoint
exec /usr/src/node-red/entrypoint.sh "$@"
