#!/bin/bash
# Fast dataset transfer script for BioWatch
# Transfers dataset (14GB) and related files to another computer

set -e

SOURCE_DIR="/Users/narjes/biowatch"

# Check if first argument is a method (http) or destination path
if [ "$1" == "http" ]; then
    # HTTP method - no destination needed
    METHOD="http"
    DEST_DIR=""
elif [ -z "$1" ] || [ "$1" == "/path/to/destination/biowatch" ]; then
    echo "Usage: $0 <destination_path> [method]"
    echo "   OR: $0 http"
    echo ""
    echo "Methods:"
    echo "  http     - Start HTTP server (EASIEST - no SSH needed, same WiFi)"
    echo "  rsync    - Fast sync over network (requires SSH)"
    echo "  tar      - Create compressed archive (for external drive/cloud)"
    echo "  scp      - Secure copy over network (requires SSH)"
    echo ""
    echo "Examples:"
    echo "  $0 http                                    # Start HTTP server"
    echo "  $0 user@remote:/path/to/biowatch rsync    # rsync to remote"
    echo "  $0 /Volumes/ExternalDrive/biowatch tar    # create archive"
    exit 1
else
    DEST_DIR="$1"
    METHOD="${2:-rsync}"
fi

echo "=========================================="
echo "BioWatch Dataset Transfer Script"
echo "=========================================="
echo ""
echo "Source: $SOURCE_DIR"
if [ -n "$DEST_DIR" ]; then
    echo "Destination: $DEST_DIR"
fi
echo "Method: $METHOD"
echo ""

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
        
        # Find available port (start from 8000, try up to 8010)
        PORT=8000
        while lsof -Pi :$PORT -sTCP:LISTEN -t >/dev/null 2>&1; do
            PORT=$((PORT + 1))
            if [ $PORT -gt 8010 ]; then
                echo "Error: Could not find available port (tried 8000-8010)"
                exit 1
            fi
        done
        
        if [ $PORT -ne 8000 ]; then
            echo "Note: Port 8000 was in use, using port $PORT instead"
            echo ""
        fi
        
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
        
        # Check if archive already exists
        if [ -f "$ARCHIVE_NAME" ]; then
            echo "✓ Archive already exists: $ARCHIVE_NAME ($(du -h "$ARCHIVE_NAME" | cut -f1))"
            echo "  Using existing archive (skip to save time)"
        else
            # Create archive
            echo "Creating archive (this takes 10-20 minutes)..."
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
