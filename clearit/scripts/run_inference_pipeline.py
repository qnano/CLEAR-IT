# clearit/scripts/run_inference_pipeline.py
import yaml, torch, numpy as np, pandas as pd
from pathlib import Path
from tqdm import tqdm
import argparse

from clearit.config import OUTPUTS_DIR, DATASETS_DIR, MODELS_DIR
from clearit.inference.pipeline import load_encoder_head
from clearit.data.classification.manager import ClassificationDataManager

def run_inference(recipe_path: Path):
    rec = yaml.safe_load(recipe_path.read_text())
    if rec.get('type') not in ('evaluate','infer','test'):
        raise ValueError("Recipe type must be 'evaluate' (or 'infer'/'test')")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device.type=='cpu':
        print("Warning: running inference on CPU.")

    for ev in rec.get('evaluations', rec.get('tests', [])):
        tid, eid, hid = ev['test_id'], ev['encoder_id'], ev['head_id']
        ds, an = ev['dataset_name'], ev['annotation_name']
        idx_npz = Path(DATASETS_DIR)/ds/an/ev['data_index_list']

        df = pd.read_csv(Path(DATASETS_DIR)/ds/an/'labels.csv')
        arr = np.load(idx_npz)
        idxs = arr[arr.files[0]]
        df_samples = df.iloc[idxs].reset_index(drop=True)

        # force at least one worker
        num_workers = max(1, rec.get('num_workers', 2))
        loader, _ = ClassificationDataManager.get_dataloader(
            dataset_name=ds,
            df_samples=df_samples,
            config={'batch_size': ev.get('batch_size',64),
                    'img_size':    ev.get('img_size',64),
                    'label_mode':  ev.get('label_mode','multilabel'),
                    'num_classes': ev.get('num_classes',2)},
            device=device,
            num_workers=num_workers,
            test_size=0,
            verbose=True
        )

        model = load_encoder_head(eid, hid, device=device)

        out_dir = OUTPUTS_DIR/'tests'/tid
        out_dir.mkdir(parents=True, exist_ok=True)

        head_dir = MODELS_DIR / 'heads' / hid
        head_cfg = yaml.safe_load((head_dir / 'conf_head.yaml').read_text())
        #head_cfg   = yaml.safe_load((Path(OUTPUTS_DIR).parent/'models'/'heads'/hid/'conf_head.yaml').read_text())
        nc, ml = head_cfg['num_classes'], head_cfg['label_mode']=='multilabel'

        results = {}
        with torch.no_grad():
            for imgs, labs, locs, fnames in tqdm(loader, desc=f"Test {tid}"):
                imgs = imgs.to(device)
                out  = model(imgs)
                preds = (torch.sigmoid(out) if ml else torch.softmax(out,1)).cpu().numpy()
                labs  = labs.numpy()
                xs, ys= locs

                for b, full in enumerate(fnames):
                    base = Path(full).stem
                    if base not in results:
                        if ml:
                            results[base] = {
                                **{f"sigmoid_{i}": [] for i in range(nc)},
                                **{f"target_{i}": []  for i in range(nc)},
                                "cell_x":[],"cell_y":[]
                            }
                        else:
                            results[base] = {"prediction":[],"target":[],"cell_x":[],"cell_y":[]}

                    results[base]["cell_x"].append(int(xs[b]))
                    results[base]["cell_y"].append(int(ys[b]))
                    if ml:
                        for i in range(nc):
                            results[base][f"sigmoid_{i}"].append(float(preds[b,i]))
                            results[base][f"target_{i}"].append(int(labs[b,i]))
                    else:
                        pcls = int(preds[b].argmax())
                        results[base]["prediction"].append(pcls)
                        results[base]["target"].append(int(labs[b]))

        for base, data in results.items():
            pd.DataFrame(data).to_csv(out_dir/f"{base}.csv", index=False)

        # ---------------------------------------------------------
        #  Write out a conf_test.yaml describing this run
        # ---------------------------------------------------------
        test_conf = {
            'annotation_name':   an,
            'data_index_list':   ev['data_index_list'],
            'dataset_name':      ds,
            'encoder_id':        eid,
            'head_id':           hid,
            'id':                tid,
            'label_mode':        head_cfg['label_mode'],
        }
        with open(out_dir / 'conf_test.yaml', 'w') as f:
            yaml.dump(test_conf, f, default_flow_style=False, sort_keys=False)


        print(f"✔  Finished test {tid}, outputs in {out_dir}")

if __name__=='__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--recipe', required=True, help='Path to inference recipe YAML')
    args = p.parse_args()
    run_inference(Path(args.recipe))
