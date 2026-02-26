from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors


def select_exemplars(
    X_pca: np.ndarray,
    meta: pd.DataFrame,
    labels: np.ndarray,
    exemplars_per_cluster: int = 10,
    per_patient_max: int = 2,
    density_k: int = 15,
) -> pd.DataFrame:
    """
    Select exemplars per cluster using a combined score of
    distance-to-centroid (smaller is better) and local density (higher is better),
    with per-patient caps to diversify.
    """
    meta = meta.copy()
    meta["cluster_id"] = labels

    centroids = (
        pd.DataFrame(X_pca)
        .assign(cluster_id=labels)
        .groupby("cluster_id")
        .mean()
        .values
    )

    rows = []
    for cid in sorted(pd.unique(labels)):
        idx = np.where(labels == cid)[0]
        Xc = X_pca[idx]

        c = centroids[cid][None, :]
        dist_centroid = np.linalg.norm(Xc - c, axis=1)

        k_loc = min(density_k, max(1, len(idx) - 1))
        nnc = NearestNeighbors(n_neighbors=k_loc + 1, metric="euclidean").fit(Xc)
        dloc, _ = nnc.kneighbors(Xc)
        dloc = dloc[:, 1:] if dloc.shape[1] > 1 else dloc
        local_density = 1.0 / (1e-8 + dloc.mean(axis=1))

        r1 = pd.Series(dist_centroid).rank(method="average", ascending=True).values
        r2 = pd.Series(local_density).rank(method="average", ascending=False).values
        combo = r1 + r2

        sub = pd.DataFrame(
            {
                "global_idx": idx,
                "cluster_id": cid,
                "dist_centroid": dist_centroid,
                "local_density": local_density,
                "rank_score": combo,
            }
        )
        rows.append(sub)

    ex_df = pd.concat(rows, ignore_index=True)
    ex_df["rank_within_cluster"] = ex_df.groupby("cluster_id")["rank_score"].rank(method="first")
    ex_df = ex_df.sort_values(["cluster_id", "rank_within_cluster"])

    def _take_with_caps(df_cluster: pd.DataFrame, max_total: int, per_patient_max: int) -> list[int]:
        taken = []
        per_pt_counts = {}
        for _, r in df_cluster.iterrows():
            gid = int(r["global_idx"])
            pid = meta.iloc[gid]["patient"]
            if per_pt_counts.get(pid, 0) >= per_patient_max:
                continue
            taken.append(gid)
            per_pt_counts[pid] = per_pt_counts.get(pid, 0) + 1
            if len(taken) >= max_total:
                break
        return taken

    chosen = []
    for cid, sub in ex_df.groupby("cluster_id", sort=True):
        chosen.extend(_take_with_caps(sub, exemplars_per_cluster, per_patient_max))

    exemplars = meta.iloc[chosen].copy()
    exemplars["exemplar_rank"] = ex_df.set_index("global_idx").loc[chosen, "rank_within_cluster"].values
    return exemplars
