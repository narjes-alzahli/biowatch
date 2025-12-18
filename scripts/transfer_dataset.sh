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
    echo "  1. http     - Start HTTP server (EASIEST - no SSH needed, same WiFi)"
    echo "  2. rsync    - Fast sync over network (requires SSH)"
    echo "  3. tar      - Create compressed archive (for external drive/cloud)"
    echo "  4. scp      - Secure copy over network (requires SSH)"
    echo ""
    echo "Examples:"
    echo "  $0 http                    # Start HTTP server (download from other computer)"
    echo "  $0 user@remote:/path/to/biowatch rsync"
    echo "  $0 /Volumes/ExternalDrive/biowatch tar"
    exit 1
fi

METHOD="${2:-rsync}"

case "$METHOD" in
    http)
        echo "Method: HTTP Server (no SSH needed!)"
        echo ""
        echo "This will start a web server on this computer."
        echo "On the OTHER computer, open a browser and download the files."
        echo ""
        
        # Get IP address
        if [[ "$OSTYPE" == "darwin"* ]]; then
            IP=$(ipconfig getifaddr en0 2>/dev/null || ipconfig getifaddr en1 2>/dev/null || echo "localhost")
        else
            IP=$(hostname -I | awk '{print $1}' 2>/dev/null || echo "localhost")
        fi
        
        PORT=8000
        
        echo "=========================================="
        echo "HTTP Server Starting..."
        echo "=========================================="
        echo ""
        echo "On the OTHER computer, open a web browser and go to:"
        echo ""
        echo "  http://${IP}:${PORT}/"
        echo ""
        echo "Or download directly:"
        echo "  http://${IP}:${PORT}/dataset.tar.gz"
        echo ""
        echo "Creating compressed archive first..."
        echo "(This will take 10-20 minutes for 14GB)"
        echo ""
        
        cd "$SOURCE_DIR"
        ARCHIVE_NAME="dataset.tar.gz"
        
        # Create archive in background or show progress
        if command -v pv &> /dev/null; then
            tar -czf - dataset/ temp_wcs_camera_traps/ | pv -s 15G > "$ARCHIVE_NAME"
        else
            tar -czf "$ARCHIVE_NAME" \
                --exclude='*.pyc' \
                --exclude='__pycache__' \
                --exclude='.DS_Store' \
                dataset/ temp_wcs_camera_traps/
        fi
        
        ARCHIVE_SIZE=$(du -h "$ARCHIVE_NAME" | cut -f1)
        echo ""
        echo "✓ Archive created: $ARCHIVE_NAME ($ARCHIVE_SIZE)"
        echo ""
        echo "=========================================="
        echo "Starting HTTP Server..."
        echo "=========================================="
        echo ""
        echo "Server running at: http://${IP}:${PORT}/"
        echo ""
        echo "On the OTHER computer:"
        echo "  1. Open browser: http://${IP}:${PORT}/"
        echo "  2. Click 'dataset.tar.gz' to download"
        echo "  3. After download, extract: tar -xzf dataset.tar.gz"
        echo ""
        echo "Press Ctrl+C to stop the server when done."
        echo ""
        
        # Start HTTP server
        python3 -m http.server "$PORT" 2>/dev/null || python -m SimpleHTTPServer "$PORT"
        ;;
    
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
