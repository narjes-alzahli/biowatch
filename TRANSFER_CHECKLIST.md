# Dataset Transfer Checklist

Before transferring the dataset, gather this information from the **destination computer**:

## 1. Connection Details

### For Network Transfer (rsync/scp):
- **IP Address or Hostname**: 
  ```bash
  # On destination computer, run:
  hostname -I    # Linux
  ipconfig getifaddr en0    # macOS
  # OR just use the hostname:
  hostname
  ```
  
- **Username**: 
  ```bash
  # On destination computer:
  whoami
  ```

- **SSH Access**: Can you SSH to it?
  ```bash
  # Test from this computer:
  ssh username@destination-ip
  # OR
  ssh username@hostname
  ```

### For External Drive:
- **Mount Point**: Where is the drive mounted?
  ```bash
  # On destination computer (or this one if using same drive):
  df -h | grep -i external
  # Common locations:
  # macOS: /Volumes/DriveName
  # Linux: /media/username/DriveName or /mnt/DriveName
  ```

## 2. Destination Path

Where do you want the dataset on the destination computer?

**Common locations:**
- `/home/username/biowatch` (Linux)
- `/Users/username/biowatch` (macOS)
- `C:\Users\username\biowatch` (Windows - use WSL or network share)

**Check available space:**
```bash
# On destination computer:
df -h /path/to/destination
# Need at least 15GB free (14GB dataset + overhead)
```

## 3. Network Speed Test (Optional)

If transferring over network, check speed:
```bash
# On destination computer, start a simple server:
python3 -m http.server 8000

# On this computer, test download speed:
curl -o /dev/null http://destination-ip:8000/test-file
```

## 4. Example Commands Based on Info

Once you have the info, use these commands:

### Example 1: Same network, Linux destination
```bash
# If destination is: user@192.168.1.100:/home/user/biowatch
./scripts/transfer_dataset.sh user@192.168.1.100:/home/user/biowatch rsync
```

### Example 2: Same network, macOS destination
```bash
# If destination is: user@macbook.local:/Users/user/biowatch
./scripts/transfer_dataset.sh user@macbook.local:/Users/user/biowatch rsync
```

### Example 3: External drive
```bash
# If drive is mounted at: /Volumes/MyDrive
./scripts/transfer_dataset.sh /Volumes/MyDrive/biowatch tar
```

### Example 4: Manual rsync (if script doesn't work)
```bash
rsync -avz --progress \
  /Users/narjes/biowatch/dataset/ \
  user@destination:/path/to/biowatch/dataset/

rsync -avz --progress \
  /Users/narjes/biowatch/temp_wcs_camera_traps/ \
  user@destination:/path/to/biowatch/temp_wcs_camera_traps/
```

## 5. Quick Info Gathering Script

Run this on the **destination computer** to get all info at once:

```bash
echo "=== Destination Computer Info ==="
echo "Hostname: $(hostname)"
echo "IP Address: $(hostname -I 2>/dev/null || ipconfig getifaddr en0 2>/dev/null || echo 'Check manually')"
echo "Username: $(whoami)"
echo "Home Directory: $HOME"
echo "Available Space:"
df -h $HOME | tail -1
echo ""
echo "Recommended destination: $HOME/biowatch"
```

## 6. After Transfer - Verification

On the destination computer, verify:

```bash
# Check files transferred
ls -lh /path/to/biowatch/dataset/ | head -10
ls -lh /path/to/biowatch/temp_wcs_camera_traps/

# Check annotations
python3 -c "
import json
with open('/path/to/biowatch/dataset/annotations.json') as f:
    data = json.load(f)
    print(f'Images: {len(data[\"images\"])}')
    print(f'Annotations: {len(data[\"annotations\"])}')
"
```
