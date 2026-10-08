import torch
import numpy as np
import spacy
from sklearn.decomposition import PCA
from scipy.linalg import sqrtm
from tqdm import tqdm
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional
from sklearn.cross_decomposition import PLSRegression
import pickle

@dataclass
class SliceData:
    key: str
    vector: np.ndarray
@dataclass
class GaussianDist:
    mu: np.ndarray
    sigma: np.ndarray
    count: int

class StyleProjector:
    def __init__(self, k: int = 4):
        self.k = k
        self.projection_matrix: Optional[np.ndarray] = None

        self.pls = None

    def fit(self, human_embeds: np.ndarray, ai_embeds: np.ndarray):

        print(f"[Projector] Preparing data for PLS (k={self.k})...")

        X = np.vstack([human_embeds, ai_embeds])
        Y = np.hstack([np.zeros(len(human_embeds)), np.ones(len(ai_embeds))])

        print(f"[Projector] Fitting PLS on {len(X)} samples with labels...")
        self.pls = PLSRegression(n_components=self.k, scale=False)
        self.pls.fit(X, Y)
        self.projection_matrix = self.pls.x_rotations_


        print(f"[Projector] PLS fitted. Projection matrix shape: {self.projection_matrix.shape}")

    def save(self, path: str):
        if self.projection_matrix is None:
            raise ValueError("Matrix is empty, cannot save.")
        np.save(path, self.projection_matrix)

    def load(self, path: str):
        self.projection_matrix = np.load(path)
        self.k = self.projection_matrix.shape[1]
        print(f"[Projector] Loaded matrix from {path}, shape: {self.projection_matrix.shape}")

    def get_torch_matrix(self, device):
        if self.projection_matrix is None:
            raise ValueError("Projection matrix not loaded/fitted.")
        return torch.tensor(self.projection_matrix, dtype=torch.float32, device=device)


class FeatureExtractor:
    def __init__(self, tokenizer, model, projector: StyleProjector, device: str,
                 window: int = 2, stride: int = 1):
        self.tokenizer = tokenizer
        self.model = model
        self.projector = projector
        self.device = device
        self.window = window
        self.stride = stride
        self.P = projector.get_torch_matrix(device)
        self.nlp = spacy.load("en_core_web_sm", disable=["ner", "lemmatizer"])

    def process_texts(self, texts: List[str], batch_size: int = 32, spacy_workers: int = 4) -> List[List[SliceData]]:
        all_results = []
        print("[Extractor] Running spaCy tagging (CPU)...")
        docs = list(tqdm(self.nlp.pipe(texts, n_process=spacy_workers, batch_size=1024), total=len(texts)))

        print("[Extractor] Running Embedding & Projection (GPU)...")

        for i in tqdm(range(0, len(docs), batch_size)):
            batch_docs = docs[i : i + batch_size]
            batch_results = self._batch_forward(batch_docs)
            all_results.extend(batch_results)

        return all_results

    @torch.no_grad()
    def _batch_forward(self, docs) -> List[List[SliceData]]:
        str_batch = [[t.text for t in d] for d in docs]
        pos_batch = [[t.pos_ for t in d] for d in docs]

        # Tokenize
        enc = self.tokenizer(
            str_batch, is_split_into_words=True, padding=True, truncation=True,
            max_length=512, return_tensors="pt"
        ).to(self.device)

        # Forward
        outputs = self.model(**enc)
        hidden = outputs.last_hidden_state # [B, L, H]

        batch_slices = []
        for b_idx in range(len(docs)):
            doc_slices = []
            word_ids = enc.word_ids(b_idx)
            seq_hidden = hidden[b_idx]
            doc_pos = pos_batch[b_idx]
            doc_len = len(doc_pos)

            w_indices = [w if w is not None else -1 for w in word_ids]
            w_indices = torch.tensor(w_indices, device=self.device)
            mask = w_indices >= 0

            if mask.sum() > 0:
                valid_hidden = seq_hidden[mask]
                valid_indices = w_indices[mask]

                max_w_idx = valid_indices.max().item()
                effective_len = min(doc_len, max_w_idx + 1)

                word_embeds = torch.zeros(effective_len, seq_hidden.size(1), device=self.device)
                counts = torch.zeros(effective_len, 1, device=self.device)

                valid_mask = valid_indices < effective_len
                if valid_mask.any():
                    valid_indices = valid_indices[valid_mask]
                    valid_hidden = valid_hidden[valid_mask]
                    word_embeds.index_add_(0, valid_indices, valid_hidden)
                    ones_source = torch.ones(valid_indices.size(0), 1, device=self.device, dtype=counts.dtype)
                    counts.index_add_(0, valid_indices, ones_source)
                    word_embeds = word_embeds / counts.clamp(min=1e-9) # Mean pooling
                    projected_words = torch.matmul(word_embeds, self.P) # [Words, k]
                    projected_cpu = projected_words.cpu().numpy()

                    for start in range(0, effective_len - self.window + 1, self.stride):
                        end = start + self.window
                        vec_flat = projected_cpu[start:end].flatten() # shape: window*k
                        # Key
                        key = "_".join(doc_pos[start:end])
                        doc_slices.append(SliceData(key=key, vector=vec_flat))

            batch_slices.append(doc_slices)

        return batch_slices


class StyleLibrary:
    def __init__(self):
        self.lib = {}
        self.threshold = 0.5

    def set_threshold(self, t: float):
        self.threshold = t
        print(f"[Library] Threshold set to: {self.threshold}")

    def build(self, human_slices_list: List[List[SliceData]], ai_slices_list: List[List[SliceData]]):
        raw_data = {}
        print("[Library] Aggregating vectors...")
        for slices in human_slices_list:
            for s in slices:
                if s.key not in raw_data: raw_data[s.key] = {'human': [], 'ai': []}
                raw_data[s.key]['human'].append(s.vector)
        for slices in ai_slices_list:
            for s in slices:
                if s.key not in raw_data: raw_data[s.key] = {'human': [], 'ai': []}
                raw_data[s.key]['ai'].append(s.vector)

        print("[Library] Fitting & Pre-computing (Optimized)...")
        keys_to_remove = []

        for key, data in tqdm(raw_data.items()):
            h_vecs = np.array(data['human'])
            a_vecs = np.array(data['ai'])

            if len(h_vecs) < 10 or len(a_vecs) < 10:
                keys_to_remove.append(key)
                continue

            mu_h, cov_h = self._fit_gaussian(h_vecs)
            mu_a, cov_a = self._fit_gaussian(a_vecs)

            w2_dist = self._compute_wasserstein(mu_h, cov_h, mu_a, cov_a)

            self.lib[key] = {
                'human': self._pack_dist(mu_h, cov_h),
                'ai':    self._pack_dist(mu_a, cov_a),
                'weight': float(w2_dist)
            }

        print(f"[Library] Built library with {len(self.lib)} keys.")

    def _pack_dist(self, mu, sigma):
        try:
            inv_sigma = np.linalg.inv(sigma)
            sign, logdet = np.linalg.slogdet(sigma)
            if sign <= 0: logdet = -100.0
        except np.linalg.LinAlgError:
            inv_sigma = np.eye(len(mu))
            logdet = 0.0

        return {
            "mu": mu,            # (k,)
            "inv_sigma": inv_sigma, # (k, k)
            "logdet": logdet     # float
        }

    def _fit_gaussian(self, X):
        mu = np.mean(X, axis=0)
        cov = np.cov(X, rowvar=False) + np.eye(X.shape[1]) * 1e-6
        return mu, cov

    def _compute_wasserstein(self, mu1, sigma1, mu2, sigma2):
        diff = mu1 - mu2
        term1 = np.sum(diff**2)
        sqrt_sigma1 = sqrtm(sigma1)
        if np.iscomplexobj(sqrt_sigma1): sqrt_sigma1 = sqrt_sigma1.real
        inner = sqrt_sigma1 @ sigma2 @ sqrt_sigma1
        sqrt_inner = sqrtm(inner)
        if np.iscomplexobj(sqrt_inner): sqrt_inner = sqrt_inner.real
        term2 = np.trace(sigma1 + sigma2 - 2 * sqrt_inner)
        return max(0, term1 + term2)

    def score(self, slices: List[SliceData]) -> Tuple[float, float, float]:
        groups = {}
        for s in slices:
            if s.key not in groups: groups[s.key] = []
            groups[s.key].append(s.vector)

        total_diff = 0.0
        total_weight = 0.0

        for key, vec_list in groups.items():
            entry = self.lib.get(key)
            if not entry: continue

            w = entry['weight']
            if w < 1e-3: continue

            X = np.stack(vec_list)

            scores_h = self._batch_mahalanobis(X, entry['human'])
            scores_a = self._batch_mahalanobis(X, entry['ai'])

            diff = scores_h - scores_a
            total_diff += np.sum(diff) * w
            total_weight += w * len(vec_list)


        if total_weight == 0:
            return 0.0, 0.0, 0.0

        final_score = total_diff / total_weight
        return 0.0, 0.0, final_score

    def _batch_mahalanobis(self, X: np.ndarray, dist: dict) -> np.ndarray:
        mu = dist['mu']             # (k,)
        inv = dist['inv_sigma']     # (k, k)

        diff = X - mu
        left = diff @ inv
        mahal = np.sum(left * diff, axis=1)

        return mahal + dist['logdet']

    def save(self, path):
        data_to_save = {
            "lib": self.lib,
            "threshold": self.threshold
        }
        with open(path, 'wb') as f:
            pickle.dump(data_to_save, f)

    def load(self, path):
        with open(path, 'rb') as f:
            data = pickle.load(f)

        if isinstance(data, dict) and "lib" in data:
            self.lib = data["lib"]
            self.threshold = data.get("threshold", 0.5)
        else:
            self.lib = data
            self.threshold = 0.5

        print(f"[Library] Loaded. Keys: {len(self.lib)}, Threshold: {self.threshold}")
