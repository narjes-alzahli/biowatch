# Quick Transfer Guide - Same WiFi

## Easiest Method (No SSH Setup Needed!)

Since both computers are on the same WiFi, use the HTTP server method:

### Step 1: On THIS Computer (Source)

```bash
cd /Users/narjes/biowatch
./scripts/transfer_dataset.sh http
```

This will:
- Create a compressed archive (`dataset.tar.gz`) - takes 10-20 minutes
- Start a web server
- Show you a URL like `http://192.168.1.100:8000/`

### Step 2: On the OTHER Computer

1. **Open a web browser** (Chrome, Firefox, Safari, etc.)

2. **Go to the URL** shown on this computer (e.g., `http://192.168.1.100:8000/`)

3. **Download** `dataset.tar.gz` (this will be ~10-12GB compressed, 14GB uncompressed)

4. **After download completes**, extract it:
   ```bash
   cd ~/Downloads  # or wherever you downloaded it
   tar -xzf dataset.tar.gz
   ```

5. **Move to your desired location**:
   ```bash
   mv dataset ~/biowatch/dataset
   mv temp_wcs_camera_traps ~/biowatch/temp_wcs_camera_traps
   ```

### Step 3: Verify

On the other computer, check the files:

```bash
ls -lh ~/biowatch/dataset/ | head -10
ls -lh ~/biowatch/temp_wcs_camera_traps/

# Check annotations
python3 -c "
import json
with open('~/biowatch/dataset/annotations.json') as f:
    data = json.load(f)
    print(f'Images: {len(data[\"images\"])}')
    print(f'Annotations: {len(data[\"annotations\"])}')
"
```

## Troubleshooting

### Can't access the URL?
- Make sure both computers are on the same WiFi network
- Check firewall settings (may need to allow port 8000)
- Try using the IP address directly instead of hostname

### Download is slow?
- WiFi speed limits transfer (usually 5-50 MB/s depending on router)
- For 14GB, expect 5-30 minutes depending on WiFi speed
- You can pause and resume downloads in most browsers

### Need faster transfer?
- Use an external drive: create archive with `tar` method, copy to drive, transfer manually
- Set up SSH (see TRANSFER_CHECKLIST.md for instructions)
