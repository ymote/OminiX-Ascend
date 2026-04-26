#!/usr/bin/env python3
"""Compare engine block-1 dumps to CLI tap dumps."""
import numpy as np

def cossim(a, b):
    a = a.astype(np.float64).flatten()
    b = b.astype(np.float64).flatten()
    na = np.linalg.norm(a); nb = np.linalg.norm(b)
    if na == 0 or nb == 0: return 0
    return float(a @ b / (na*nb))

def absratio(a, b):
    am = float(np.abs(a).max()); bm = float(np.abs(b).max())
    return am/bm if bm else float('inf')

# Block 1
eng_resid1 = np.fromfile("/tmp/qie_5536_eng_real/block01/13_img_resid1.f32", dtype=np.float32)
eng_resid2 = np.fromfile("/tmp/qie_5536_eng_real/block01/24_img_resid2.f32", dtype=np.float32)
cli_resid1 = np.fromfile("/tmp/qie_5529_cli_blocks/block01/qie_cli_blk01_13_img_resid1.f32.bin", dtype=np.float32)
cli_resid2 = np.fromfile("/tmp/qie_5529_cli_blocks/block01/qie_cli_blk01_24_img_resid2.f32.bin", dtype=np.float32)

print(f"=== block 01 engine vs CLI ===")
print(f"13_img_resid1: cos={cossim(eng_resid1, cli_resid1):.6f}  absratio={absratio(eng_resid1, cli_resid1):.4f}")
print(f"24_img_resid2: cos={cossim(eng_resid2, cli_resid2):.6f}  absratio={absratio(eng_resid2, cli_resid2):.4f}")
print(f"  eng resid1 mean_abs={np.abs(eng_resid1).mean():.4e}  max_abs={np.abs(eng_resid1).max():.4e}")
print(f"  cli resid1 mean_abs={np.abs(cli_resid1).mean():.4e}  max_abs={np.abs(cli_resid1).max():.4e}")
print(f"  eng resid2 mean_abs={np.abs(eng_resid2).mean():.4e}  max_abs={np.abs(eng_resid2).max():.4e}")
print(f"  cli resid2 mean_abs={np.abs(cli_resid2).mean():.4e}  max_abs={np.abs(cli_resid2).max():.4e}")

# Block 0 too
print(f"\n=== block 00 engine vs CLI ===")
eng_resid1_0 = np.fromfile("/tmp/qie_5536_eng_real/block00/13_img_resid1.f32", dtype=np.float32)
eng_resid2_0 = np.fromfile("/tmp/qie_5536_eng_real/block00/24_img_resid2.f32", dtype=np.float32)
cli_resid1_0 = np.fromfile("/tmp/qie_5529_cli_blocks/block00/qie_cli_blk00_13_img_resid1.f32.bin", dtype=np.float32)
cli_resid2_0 = np.fromfile("/tmp/qie_5529_cli_blocks/block00/qie_cli_blk00_24_img_resid2.f32.bin", dtype=np.float32)
print(f"13_img_resid1: cos={cossim(eng_resid1_0, cli_resid1_0):.6f}  absratio={absratio(eng_resid1_0, cli_resid1_0):.4f}")
print(f"24_img_resid2: cos={cossim(eng_resid2_0, cli_resid2_0):.6f}  absratio={absratio(eng_resid2_0, cli_resid2_0):.4f}")
