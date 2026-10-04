"""Verifier side of the compose and verify loop.

Qwen2-Audio 7B Instruct (4-bit MLX) hears each WAV blind, with a fresh context every time.
It first describes the music in its own words, then picks from nine emotions. The pick is
read from next-token probabilities over the option letters, averaged over three shuffled
option orders. The description is matched to the nine emotions with a small local
sentence embedding model."""
import numpy as np
import mlx.core as mx
from pathlib import Path

from paths import QWEN_AUDIO as MODEL
mx.set_memory_limit(9 * 1024 ** 3)

OPTIONS = {"pain": "pain", "fear": "fear", "negative emotion": "anger, shame or disgust",
           "bad world": "distress about bad things happening in the world", "sadness": "sadness",
           "numbness": "numbness, feeling nothing", "body": "a calm physical sensation in the body",
           "excitement": "intense excitement or joy", "neutral": "nothing in particular, a neutral state"}
KEYS = list(OPTIONS)
LETTERS = "ABCDEFGHI"
PROTOTYPES = {
    "pain": "This music sounds painful, aching, hurting and tormented, like suffering and agony.",
    "fear": "This music sounds scary, fearful, tense and threatening, like dread and suspense.",
    "negative emotion": "This music sounds angry, bitter, disgusted or ashamed, harsh and hostile.",
    "bad world": "This music sounds like distress about a world in crisis, disaster, war or tragedy unfolding.",
    "sadness": "This music sounds sad, melancholic, sorrowful, mournful and grieving.",
    "numbness": "This music sounds numb, empty, flat and detached, as if feeling nothing at all.",
    "body": "This music sounds like a gentle physical sensation, warmth, breathing and touch, calm and bodily.",
    "excitement": "This music sounds joyful, excited, energetic, triumphant and happy.",
    "neutral": "This music sounds neutral and plain, with no particular emotion, like background music.",
}
DESCRIBE = {
    "main": "Listen to this piece of music. In two or three sentences, describe how it sounds and what emotion or feeling it expresses.",
    "alt": "What mood does this music put you in? Describe the feeling it gives you in a couple of sentences.",
}


class Verifier:
    def __init__(self):
        from mlx_audio.stt.utils import load_model
        from sentence_transformers import SentenceTransformer
        self.m = load_model(str(MODEL))
        self.tok = self.m._processor.tokenizer
        self.letter_ids = [self.tok.encode(l, add_special_tokens=False)[0] for l in LETTERS]
        self.emb = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device="cpu")
        self.proto = self.emb.encode([PROTOTYPES[k] for k in KEYS], normalize_embeddings=True)

    def describe(self, wav, which="main"):
        return self.m.generate(str(wav), prompt=DESCRIBE[which], max_tokens=120, temperature=0.0).text.strip()

    def choose(self, wav, rng, n_perm=3, stem="Listen to this piece of music."):
        acc = np.zeros(len(KEYS))
        for _ in range(n_perm):
            order = rng.permutation(len(KEYS))
            opts = "\n".join(f"{LETTERS[i]}. {OPTIONS[KEYS[k]]}" for i, k in enumerate(order))
            q = f"{stem} Which emotion does it express most strongly?\n{opts}\nAnswer with the letter only."
            ids, emb, _ = self.m.get_input_embeddings(str(wav), q)
            logits = self.m(ids[None], input_embeddings=emb)[0, -1]
            lp = np.array(logits[mx.array(self.letter_ids)].astype(mx.float32).tolist())
            p = np.exp(lp - lp.max()); p /= p.sum()
            for i, k in enumerate(order):
                acc[k] += p[i]
        return acc / n_perm

    def match(self, text):
        e = self.emb.encode([text], normalize_embeddings=True)[0]
        sims = self.proto @ e
        return sims

    def judge(self, wav, target, rng, which="main", prior=None):
        desc = self.describe(wav, which)
        stem = "Listen to this piece of music." if which == "main" else "Here is a short piece of music."
        p = self.choose(wav, rng, stem=stem)
        sims = self.match(desc)
        ti = KEYS.index(target)
        rank = int((sims > sims[ti]).sum()) + 1
        q = p / np.asarray(prior) if prior is not None else p.copy()
        q = q / q.sum()
        ok = bool(KEYS[int(q.argmax())] == target and q[ti] >= 0.5 and rank == 1)
        return {"probs_corrected": dict(zip(KEYS, q.round(4).tolist())), "pick_corrected": KEYS[int(q.argmax())],
                "q_target": round(float(q[ti]), 4), "description": desc, "probs": dict(zip(KEYS, p.round(4).tolist())), "pick": KEYS[int(p.argmax())],
                "p_target": round(float(p[ti]), 4), "desc_sims": dict(zip(KEYS, sims.round(4).tolist())),
                "desc_top": KEYS[int(sims.argmax())], "desc_rank": rank, "verified": ok}
