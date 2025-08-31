# train_any.py
import argparse, os, torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler

from dataset_any import HSIDatasetAny
from hsimamba_sr import HSIMambaSR, HSIMambaSRConfig

def psnr_bands_avg(sr, hr, eps=1e-8):
    # sr, hr: (B,D,H,W) in [0,1]
    mse = (sr - hr).pow(2).mean(dim=(-1, -2))     # (B,D)
    psnr = 10 * torch.log10(1.0 / (mse + eps))    # (B,D)
    return psnr.mean().item()

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_set = HSIDatasetAny(
        root=args.data_root, scale=args.scale, split="train",
        patch_hr=args.patch_hr, val_ratio=args.val_ratio,
        var_name=args.var_name, norm=args.norm
    )
    val_set = HSIDatasetAny(
        root=args.data_root, scale=args.scale, split="val",
        patch_hr=None, val_ratio=args.val_ratio,
        var_name=args.var_name, norm=args.norm
    )
    tr = DataLoader(train_set, batch_size=args.batch, shuffle=True, num_workers=2, pin_memory=True)
    va = DataLoader(val_set, batch_size=1, shuffle=False, num_workers=2)

    # Infer bands D from first batch for convenience
    sample_lr, sample_hr = next(iter(tr))
    bands = sample_hr.shape[1] if sample_hr.ndim == 4 else sample_hr.shape[0]
    if args.bands is not None and args.bands != bands:
        raise ValueError(f"--bands={args.bands} but loaded data has D={bands}")
    D = args.bands or bands

    cfg = HSIMambaSRConfig(
        bands=D, scale=args.scale,
        embed_dim=args.embed_dim, n_groups=args.groups,
        depth_per_group=args.depth, d_state=16, expand=2,
        axis_fusion=args.fusion, backend=args.backend, drop_path=0.1
    )
    net = HSIMambaSR(cfg).to(device)
    opt = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=1e-4)
    scaler = GradScaler()

    os.makedirs(args.out, exist_ok=True)
    best_psnr = -1.0

    for epoch in range(1, args.epochs + 1):
        net.train()
        for lr, hr in tr:
            lr, hr = lr.to(device), hr.to(device)
            opt.zero_grad(set_to_none=True)
            with autocast():
                sr = net(lr)
                loss = F.mse_loss(sr, hr)
            scaler.scale(loss).backward()
            scaler.step(opt); scaler.update()

        # validation
        net.eval()
        psnrs = []
        with torch.no_grad():
            for lr, hr in va:
                lr, hr = lr.to(device), hr.to(device)
                sr = net(lr)
                psnrs.append(psnr.mean().item() if (psnrs and False) else psnr_bands_avg(sr, hr))
        mpsnr = sum(psnrs) / len(psnrs)
        print(f"epoch {epoch}: PSNR={mpsnr:.2f}  (last-train-loss={loss.item():.4f})")

        if mpsnr > best_psnr:
            best_psnr = mpsnr
            torch.save({"model": net.state_dict(), "cfg": cfg.__dict__}, os.path.join(args.out, "best.pth"))

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", type=str, required=True, help="folder of HSI files (.mat/.h5/.npy/.npz)")
    ap.add_argument("--scale", type=int, default=4, choices=[2,4])
    ap.add_argument("--bands", type=int, default=None, help="Optional: expected #bands; leave None to auto-detect")
    ap.add_argument("--embed_dim", type=int, default=64)
    ap.add_argument("--groups", type=int, default=4)
    ap.add_argument("--depth", type=int, default=4)
    ap.add_argument("--fusion", type=str, default="gated", choices=["gated","linear"])
    ap.add_argument("--backend", type=str, default="gru", choices=["gru","mamba"])  # 'gru' = easy start
    ap.add_argument("--patch_hr", type=int, default=128)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--val_ratio", type=float, default=0.1)
    ap.add_argument("--var_name", type=str, default=None, help="For .mat/.h5: name of the dataset variable to read")
    ap.add_argument("--norm", type=str, default="auto", choices=["auto","none"])
    ap.add_argument("--out", type=str, default="runs/any")
    args = ap.parse_args()
    main(args)
