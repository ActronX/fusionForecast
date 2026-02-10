#!/bin/sh
set -e

SETTINGS_FILE="/data/settings.js"
DEFAULT_SETTINGS="/usr/src/node-red/node_modules/node-red/settings.js"

echo "Node-RED Entrypoint: Checking settings..."

# Pre-initialize settings.js if missing so we can inject secrets
if [ ! -f "$SETTINGS_FILE" ]; then
    echo "Node-RED Entrypoint: initializing settings.js from defaults..."
    cp "$DEFAULT_SETTINGS" "$SETTINGS_FILE"
else
    echo "Node-RED Entrypoint: settings.js found."
fi

if [ -n "$CREDENTIAL_SECRET" ] && [ -f "$SETTINGS_FILE" ]; then
    echo "Node-RED Entrypoint: Injecting credentialSecret..."
    
    # Check if credentialSecret is already set (uncommented)
    if grep -q '^\s*credentialSecret:' "$SETTINGS_FILE"; then
        echo "Node-RED Entrypoint: Replacing existing credentialSecret"
        # Replace existing value
        sed -i "s|^\(\s*\)credentialSecret:.*|\\1credentialSecret: \"$CREDENTIAL_SECRET\",|" "$SETTINGS_FILE"
    elif grep -q '//credentialSecret:' "$SETTINGS_FILE"; then
        echo "Node-RED Entrypoint: Uncommenting credentialSecret"
        # Uncomment and set value
        sed -i "s|^\(\s*\)//credentialSecret:.*|\\1credentialSecret: \"$CREDENTIAL_SECRET\",|" "$SETTINGS_FILE"
    else
        echo "Node-RED Entrypoint: Appending credentialSecret"
        # Insert after flowFile line
        sed -i "/flowFile:/a\\    credentialSecret: \"$CREDENTIAL_SECRET\"," "$SETTINGS_FILE"
    fi
    echo "Node-RED: credentialSecret configured."
else
    echo "Node-RED Entrypoint: Skipping secret injection (SECRET empty or file missing)"
fi

# Hand off to the original Node-RED entrypoint
exec /usr/src/node-red/entrypoint.sh "$@"
