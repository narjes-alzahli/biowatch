# Hardware Requirements for BioWatch Multi-Modal Object Detection

## Quick Summary

**Minimum (CPU-only, slow):**
- CPU: 4+ cores
- RAM: 16 GB
- GPU: None (but training will be very slow - days/weeks)

**Recommended (GPU training):**
- GPU: 8+ GB VRAM (NVIDIA with CUDA support)
- CPU: 4+ cores
- RAM: 16 GB
- Storage: 50+ GB free space

**Optimal (fast training):**
- GPU: 16+ GB VRAM (RTX 3090, A100, etc.)
- CPU: 8+ cores
- RAM: 32 GB
- Storage: 100+ GB free space (for checkpoints and logs)

---

## Detailed Requirements

### GPU Requirements

#### VRAM (Video Memory) Requirements

The model memory usage depends on:
- **Batch size** (default: 16)
- **Input image size** (default: 640×640)
- **Fusion method** (feature fusion processes both RGB and thermal = 2x memory)
- **Backbone** (ResNet50 vs ResNet101)

**Memory Estimates:**

| Configuration | Approximate VRAM |
|--------------|------------------|
| ResNet50, batch=4, early fusion | ~4-6 GB |
| ResNet50, batch=8, early fusion | ~6-8 GB |
| ResNet50, batch=16, early fusion | ~10-12 GB |
| ResNet50, batch=16, feature fusion | ~14-16 GB |
| ResNet101, batch=8, feature fusion | ~12-14 GB |
| ResNet101, batch=16, feature fusion | ~18-22 GB |

**GPU Recommendations:**

1. **Budget Option (8 GB VRAM):**
   - NVIDIA RTX 3060, RTX 3070, GTX 1080 Ti
   - Use: `--batch-size 4 --fusion-method early --backbone resnet50`
   - Training time: ~2-3 days for 100 epochs

2. **Recommended (12-16 GB VRAM):**
   - NVIDIA RTX 3080, RTX 4070, RTX 4060 Ti 16GB
   - Use: `--batch-size 8 --fusion-method feature --backbone resnet50`
   - Training time: ~1-2 days for 100 epochs

3. **Optimal (16+ GB VRAM):**
   - NVIDIA RTX 3090, RTX 4090, A100, V100
   - Use: `--batch-size 16 --fusion-method feature --backbone resnet50`
   - Training time: ~12-24 hours for 100 epochs

#### CUDA Support

- **CUDA Version**: 11.8 or higher (PyTorch 2.0+)
- **cuDNN**: Included with PyTorch
- **GPU Compute Capability**: 7.0+ (Pascal, Volta, Turing, Ampere, Ada)

**Check GPU compatibility:**
```bash
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"None\"}')"
```

### CPU Requirements

**Minimum:**
- 4 CPU cores
- 2.5+ GHz clock speed
- Used for: Data loading, preprocessing, CPU fallback

**Recommended:**
- 8+ CPU cores
- 3.0+ GHz clock speed
- Helps with: Faster data loading, multi-worker data loading

**Note:** CPU is mainly used for data loading. The actual training happens on GPU.

### RAM Requirements

**Minimum:**
- 16 GB RAM
- For: Loading images, caching, data augmentation

**Recommended:**
- 32 GB RAM
- For: Larger batches, faster data loading, multiple workers

**Memory Usage:**
- Dataset loading: ~2-4 GB
- Data augmentation: ~1-2 GB
- Model weights: ~200-500 MB
- Training buffers: ~1-2 GB
- **Total: ~5-10 GB RAM** (excluding OS)

### Storage Requirements

**Minimum:**
- 50 GB free space
- For: Dataset, checkpoints, logs

**Recommended:**
- 100+ GB free space
- For: Multiple checkpoints, experiment logs, tensorboard data

**Storage Breakdown:**
- Dataset: ~20-30 GB (already have this)
- Checkpoints: ~500 MB - 2 GB each
- Training logs: ~1-5 GB
- Model outputs: ~1-10 GB

---

## Configuration for Limited Resources

### If you have 4-6 GB VRAM:

```bash
python models/train.py \
    --batch-size 2 \
    --fusion-method early \
    --backbone resnet50 \
    --input-size 512 512
```

### If you have 8 GB VRAM:

```bash
python models/train.py \
    --batch-size 4 \
    --fusion-method early \
    --backbone resnet50 \
    --input-size 640 640
```

### If you have 12 GB VRAM:

```bash
python models/train.py \
    --batch-size 8 \
    --fusion-method feature \
    --backbone resnet50 \
    --input-size 640 640
```

### If you have 16+ GB VRAM:

```bash
python models/train.py \
    --batch-size 16 \
    --fusion-method feature \
    --backbone resnet50 \
    --input-size 640 640
```

---

## CPU-Only Training (Not Recommended)

**Requirements:**
- 16+ CPU cores
- 64+ GB RAM
- **Training time: 1-2 weeks for 100 epochs**

**Configuration:**
```bash
python models/train.py \
    --batch-size 1 \
    --fusion-method early \
    --backbone resnet50 \
    --input-size 512 512
```

**Note:** PyTorch will automatically use CPU if no GPU is available, but training will be extremely slow.

---

## Cloud GPU Options

If you don't have a GPU, consider cloud options:

1. **Google Colab (Free):**
   - Free: T4 GPU (16 GB VRAM, limited hours)
   - Pro: V100/A100 (faster, more hours)
   - Cost: Free or $10/month

2. **Kaggle Notebooks (Free):**
   - P100 GPU (16 GB VRAM)
   - 30 hours/week free
   - Cost: Free

3. **AWS/GCP/Azure:**
   - Spot instances: $0.50-2/hour
   - On-demand: $1-5/hour
   - Various GPU options

4. **Lambda Labs:**
   - RTX 3090: ~$0.50/hour
   - A100: ~$1.10/hour

---

## Performance Estimates

### Training Time (100 epochs, ~100K images)

| GPU | Batch Size | Time (hours) |
|-----|------------|--------------|
| RTX 3060 (12GB) | 8 | ~36-48 |
| RTX 3080 (10GB) | 8 | ~24-36 |
| RTX 3090 (24GB) | 16 | ~12-18 |
| RTX 4090 (24GB) | 16 | ~8-12 |
| CPU (16 cores) | 1 | ~200-300 |

*Times are approximate and depend on dataset size, image resolution, and other factors.*

---

## Memory Optimization Tips

1. **Reduce batch size**: Halve batch size to reduce VRAM by ~40%
2. **Use early fusion**: Uses less memory than feature fusion
3. **Reduce input size**: 512×512 uses ~60% less memory than 640×640
4. **Use gradient accumulation**: Simulate larger batches with smaller VRAM
5. **Mixed precision training**: Use FP16 to reduce memory by ~50% (future feature)

---

## Checking Your System

### Check GPU:
```bash
nvidia-smi
```

### Check CUDA:
```bash
python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'No GPU')"
```

### Check RAM:
```bash
# Linux/Mac
free -h

# Or
python -c "import psutil; print(f'RAM: {psutil.virtual_memory().total / (1024**3):.1f} GB')"
```

---

## Recommendations

**For most users:**
- Get a GPU with at least 8 GB VRAM (RTX 3060 or better)
- Use batch size 4-8 with early fusion
- Training will take 1-3 days

**For serious training:**
- Get a GPU with 16+ GB VRAM (RTX 3090/4090 or better)
- Use batch size 16 with feature fusion
- Training will take 12-24 hours

**If no GPU available:**
- Consider cloud GPU options (Colab, Kaggle)
- Or use CPU with very small batch size (expect 1-2 weeks)

