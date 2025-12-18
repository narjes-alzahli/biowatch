# Transfer Dataset to Another Computer

## Same WiFi (Easiest Method)

**On this computer:**
```bash
./scripts/transfer_dataset.sh http
```

This creates an archive and starts a web server. You'll see a URL like `http://192.168.x.x:8000/`

**On the other computer:**
1. Open browser and go to the URL shown
2. Download `dataset.tar.gz` (~17GB)
3. Extract: `tar -xzf dataset.tar.gz`

That's it!

## Other Methods

### External Drive
```bash
./scripts/transfer_dataset.sh /Volumes/DriveName/biowatch tar
```
Then copy the archive manually.

### Network (requires SSH)
```bash
./scripts/transfer_dataset.sh user@remote-ip:/path/to/biowatch rsync
```

## Troubleshooting

- **Port 8000 in use?** The script automatically uses the next available port (8001, 8002, etc.)
- **Can't access URL?** Make sure both computers are on the same WiFi
- **Archive already exists?** The script will use it instead of recreating
