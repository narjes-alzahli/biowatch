#!/bin/bash
# Run this script on the DESTINATION computer to get transfer details

echo "=========================================="
echo "BioWatch Transfer - Destination Info"
echo "=========================================="
echo ""
echo "Copy the information below and send it to the source computer:"
echo ""

# Basic info
echo "1. USERNAME:"
echo "   $(whoami)"
echo ""

echo "2. HOSTNAME:"
echo "   $(hostname)"
echo ""

echo "3. IP ADDRESS:"
if command -v hostname &> /dev/null && hostname -I &> /dev/null; then
    echo "   $(hostname -I | awk '{print $1}')"
elif [[ "$OSTYPE" == "darwin"* ]]; then
    echo "   $(ipconfig getifaddr en0 2>/dev/null || ipconfig getifaddr en1 2>/dev/null || echo 'Check System Preferences > Network')"
else
    echo "   Run: ip addr show | grep 'inet '"
fi
echo ""

echo "4. RECOMMENDED DESTINATION PATH:"
echo "   $HOME/biowatch"
echo ""

echo "5. AVAILABLE SPACE:"
df -h "$HOME" | tail -1 | awk '{print "   Available: " $4 " (Total: " $2 ", Used: " $3 ")"}'
echo ""

echo "6. SSH ACCESS TEST:"
if command -v ssh &> /dev/null; then
    echo "   ✓ SSH is available"
    echo "   Test from source: ssh $(whoami)@$(hostname -I 2>/dev/null | awk '{print $1}' || hostname)"
else
    echo "   ✗ SSH not found - may need to install/configure"
fi
echo ""

echo "=========================================="
echo "TRANSFER COMMAND FOR SOURCE COMPUTER:"
echo "=========================================="
USERNAME=$(whoami)
HOSTNAME_OR_IP=$(hostname -I 2>/dev/null | awk '{print $1}' || hostname)
DEST_PATH="$HOME/biowatch"

echo ""
echo "Option 1 (using IP):"
echo "  ./scripts/transfer_dataset.sh ${USERNAME}@${HOSTNAME_OR_IP}:${DEST_PATH} rsync"
echo ""
echo "Option 2 (using hostname):"
echo "  ./scripts/transfer_dataset.sh ${USERNAME}@$(hostname):${DEST_PATH} rsync"
echo ""
echo "Option 3 (if same user, can use):"
echo "  ./scripts/transfer_dataset.sh ${HOSTNAME_OR_IP}:${DEST_PATH} rsync"
echo ""
