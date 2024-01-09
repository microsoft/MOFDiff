from pathlib import Path
import argparse
from tqdm import tqdm
from collections import defaultdict
import torch
from omegaconf import OmegaConf
from torch.optim import Adam
from torch_geometric.loader import DataLoader

from scipy.spatial import KDTree
from mofdiff.common.atomic_utils import arrange_decoded_mofs
from mofdiff.common.eval_utils import load_mofdiff_model
from mofdiff.common.data_utils import logmod
from mofdiff.data.dataset import MOFDataset

from pytorch_lightning import seed_everything


def load_data(args, n_points=100, samples=None, batch_size=32):
    args["n_points"] = n_points
    args["samples"] = samples
    return DataLoader(MOFDataset(**args), batch_size=batch_size, shuffle=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_path",
        type=str,
        default="PATH/mofdiff_ckpt",
    )
    parser.add_argument(
        "--bb_cache_path",
        type=str,
        default="PATH/bb_emb_space.pt",
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default="PATH/MetalOxo.lmdb",
    )
    parser.add_argument("--n_samples", default=1024, type=int)
    parser.add_argument("--batch_size", default=256, type=int)
    parser.add_argument(
        "--lr", default=3e-4, type=float, help="learning rate for optimization"
    )
    parser.add_argument(
        "--property",
        default="working_capacity_vacuum_swing [mmol/g]",
        type=str,
        help="property to optimize",
    )
    parser.add_argument(
        "--target_v",
        default=15.0,
        type=float,
        help="target value for the property to optimize",
    )
    parser.add_argument("--max", action="store_true", help="maximize property")
    parser.add_argument("--min", action="store_true", help="minimize property")
    parser.add_argument(
        "--grad_steps", default=5001, type=int, help="number of optimization steps"
    )
    parser.add_argument("--seed", default=123, type=int)

    # get datamodule prop_list
    args = parser.parse_args()
    print(args.property)
    seed_everything(args.seed)
    model_path = Path(args.model_path)
    model, cfg, bb_encoder = load_mofdiff_model(model_path)
    model = model.to("cuda")
    all_data, all_z = torch.load(args.bb_cache_path)
    kdtree = KDTree(all_z)

    prop_list = OmegaConf.to_container(cfg.data.prop_list, resolve=True)
    assert (
        args.property in prop_list
    ), "specified property not in the dataset. choose from: {}".format(prop_list)
    prop_to_index = {prop: i for i, prop in enumerate(prop_list)}
    prop_idx = prop_to_index[args.property]
    prop_mean, prop_std = model.scaler.means[prop_idx], model.scaler.stds[prop_idx]

    target_v = torch.tensor(args.target_v, dtype=torch.float32, device="cuda")
    if cfg.data.logmod:
        target_v = logmod(target_v)
    target_v = (target_v - prop_mean) / prop_std

    data_cfg = OmegaConf.to_container(cfg.data, resolve=True)
    data_cfg.update(
        {
            "path": args.data_path,
            "bb_encoder": bb_encoder,
            "keep_bbs": True,
        }
    )

    loader = load_data(
        data_cfg,
        n_points=args.n_samples,
        batch_size=args.batch_size,
    )

    def optimize_batch(z, model, lr, num_grad_steps):
        n_opt = z.shape[0]
        z.requires_grad = True
        opt = Adam([z], lr=lr)
        model.freeze()

        all_mofs = []
        all_z = []
        all_props = []
        for _ in tqdm(range(num_grad_steps)):
            opt.zero_grad()

            props = model.fc_property(z)[:, prop_idx]
            if args.max:
                loss = -props.mean()
            elif args.min:
                loss = props.mean()
            else:
                loss = ((props - target_v) ** 2).mean()

            loss.backward()
            opt.step()

        samples = model.sample(z.shape[0], z, save_freq=False)
        mofs = arrange_decoded_mofs(samples, kdtree, all_data, False)
        for j in range(n_opt):
            all_mofs.append(mofs[j].cpu())
        all_z.append(z.clone().detach().cpu())
        all_props.append(model.predict_property(z).clone().detach().cpu()[:, prop_idx])
        all_z = torch.stack(all_z).transpose(0, 1)
        all_props = torch.stack(all_props).transpose(0, 1)
        return {"mofs": all_mofs, "z": all_z, "prop": all_props}

    opt_kwargs = {
        "lr": args.lr,
        "num_grad_steps": args.grad_steps,
    }

    output = defaultdict(list)
    for idx, batch in enumerate(loader):
        batch = batch.to("cuda")
        z = model.encode(batch)[2].detach().requires_grad_(True)
        results = optimize_batch(z, model, **opt_kwargs)
        for k, v in results.items():
            output[k].extend(v)
        batch = batch.to("cpu")

    output["z"] = torch.stack(output["z"], dim=0)
    output["prop"] = torch.stack(output["prop"], dim=0)
    output["opt_args"] = opt_kwargs
    output["prop_name"] = args.property
    output["target_v"] = args.target_v
    output["max"] = args.max
    output["n_samples"] = (args.n_samples,)

    prop_name = args.property.split(" ")[0].replace("/", "-").replace("_", "-")
    if args.max:
        savename = f"optimized_{prop_name}_max_{args.n_samples}_seed_{args.seed}"
    else:
        savename = f"optimized_{prop_name}_{args.target_v:.0f}_{args.n_samples}_seed_{args.seed}"
    (model_path / savename).mkdir(exist_ok=True)
    torch.save(output, model_path / savename / "samples.pt")
