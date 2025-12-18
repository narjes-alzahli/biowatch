#!/bin/bash
# Fast dataset transfer script for BioWatch
# Transfers dataset (14GB) and related files to another computer

set -e

SOURCE_DIR="/Users/narjes/biowatch"
DEST_DIR="${1:-/path/to/destination/biowatch}"

echo "=========================================="
echo "BioWatch Dataset Transfer Script"
echo "=========================================="
echo ""
echo "Source: $SOURCE_DIR"
echo "Destination: $DEST_DIR"
echo ""

# Check if destination is provided
if [ "$DEST_DIR" == "/path/to/destination/biowatch" ]; then
    echo "Usage: $0 <destination_path> [method]"
    echo ""
    echo "Methods:"
    echo "  1. rsync    - Fast sync over network (recommended for same network)"
    echo "  2. tar      - Create compressed archive (for external drive/cloud)"
    echo "  3. scp      - Secure copy over network"
    echo ""
    echo "Example:"
    echo "  $0 user@remote:/path/to/biowatch rsync"
    echo "  $0 /Volumes/ExternalDrive/biowatch tar"
    exit 1
fi

METHOD="${2:-rsync}"

case "$METHOD" in
    rsync)
        echo "Method: rsync (fast network sync)"
        echo ""
        echo "This will transfer:"
        echo "  - dataset/ (14GB)"
        echo "  - temp_wcs_camera_traps/ (95MB)"
        echo ""
        read -p "Continue? (y/n) " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            exit 1
        fi
        
        # Create destination directory
        if [[ "$DEST_DIR" == *":"* ]]; then
            # Remote destination
            ssh "${DEST_DIR%%:*}" "mkdir -p ${DEST_DIR#*:}"
        else
            mkdir -p "$DEST_DIR"
        fi
        
        echo "Transferring dataset/..."
        rsync -avz --progress \
            "$SOURCE_DIR/dataset/" \
            "$DEST_DIR/dataset/"
        
        echo "Transferring temp_wcs_camera_traps/..."
        rsync -avz --progress \
            "$SOURCE_DIR/temp_wcs_camera_traps/" \
            "$DEST_DIR/temp_wcs_camera_traps/"
        
        echo ""
        echo "✓ Transfer complete!"
        ;;
    
    tar)
        echo "Method: tar (compressed archive)"
        echo ""
        ARCHIVE_NAME="biowatch_dataset_$(date +%Y%m%d_%H%M%S).tar.gz"
        echo "Creating archive: $ARCHIVE_NAME"
        echo "This may take 10-20 minutes for 14GB..."
        
        cd "$SOURCE_DIR"
        tar -czf "$ARCHIVE_NAME" \
            --exclude='*.pyc' \
            --exclude='__pycache__' \
            --exclude='.DS_Store' \
            dataset/ temp_wcs_camera_traps/
        
        echo ""
        echo "Archive created: $ARCHIVE_NAME"
        echo "Size: $(du -h "$ARCHIVE_NAME" | cut -f1)"
        echo ""
        echo "To extract on destination:"
        echo "  tar -xzf $ARCHIVE_NAME -C $DEST_DIR"
        ;;
    
    scp)
        echo "Method: scp (secure copy)"
        echo ""
        echo "This will transfer:"
        echo "  - dataset/ (14GB)"
        echo "  - temp_wcs_camera_traps/ (95MB)"
        echo ""
        read -p "Continue? (y/n) " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            exit 1
        fi
        
        echo "Transferring dataset/..."
        scp -r "$SOURCE_DIR/dataset" "$DEST_DIR/"
        
        echo "Transferring temp_wcs_camera_traps/..."
        scp -r "$SOURCE_DIR/temp_wcs_camera_traps" "$DEST_DIR/"
        
        echo ""
        echo "✓ Transfer complete!"
        ;;
    
    *)
        echo "Unknown method: $METHOD"
        echo "Available methods: rsync, tar, scp"
        exit 1
        ;;
esac

echo ""
echo "=========================================="
echo "Next steps on destination computer:"
echo "=========================================="
echo "1. Verify files:"
echo "   ls -lh $DEST_DIR/dataset/"
echo "   ls -lh $DEST_DIR/temp_wcs_camera_traps/"
echo ""
echo "2. Check annotations:"
echo "   python3 -c \"import json; f=open('$DEST_DIR/dataset/annotations.json'); d=json.load(f); print(f'Images: {len(d[\"images\"])}, Annotations: {len(d[\"annotations\"])}')\""
echo ""
