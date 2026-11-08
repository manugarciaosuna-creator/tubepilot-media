"""
Kronos Forecast — prototipo de panel de predicción de velas (SOLO VISUALIZACIÓN).

No envía órdenes, no usa claves de API, no toca ninguna cuenta.
Descarga velas públicas de Binance, genera N trayectorias futuras con Kronos
(si está instalado) o con un modelo base estadístico, y las sirve a index.html.

Uso:
    python server.py               # datos reales de Binance
    python server.py --demo        # datos sintéticos (sin internet)
    python server.py --port 8000
"""

import argparse
import json
import math
import os
import random
import sys
import time
import urllib.parse
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
BINANCE_HOSTS = ["https://api.binance.com", "https://data-api.binance.vision"]
INTERVAL_MS = {
    "1m": 60_000, "5m": 300_000, "15m": 900_000, "30m": 1_800_000,
    "1h": 3_600_000, "4h": 14_400_000, "1d": 86_400_000,
}
DEMO = False


# ---------------------------------------------------------------- datos

def fetch_klines(symbol: str, interval: str, limit: int) -> pd.DataFrame:
    if DEMO:
        return synthetic_klines(interval, limit)
    q = urllib.parse.urlencode({"symbol": symbol, "interval": interval, "limit": min(limit, 1000)})
    last_err = None
    for host in BINANCE_HOSTS:
        try:
            req = urllib.request.Request(f"{host}/api/v3/klines?{q}", headers={"User-Agent": "kronos-forecast-proto"})
            with urllib.request.urlopen(req, timeout=15) as r:
                rows = json.loads(r.read())
            df = pd.DataFrame(rows, columns=[
                "open_time", "open", "high", "low", "close", "volume", "close_time",
                "amount", "trades", "tb_base", "tb_quote", "ignore"])
            df = df[["open_time", "open", "high", "low", "close", "volume", "amount"]].astype(float)
            df["open_time"] = df["open_time"].astype("int64")
            return df
        except Exception as e:  # probar el siguiente host
            last_err = e
    raise RuntimeError(f"No se pudo descargar de Binance: {last_err}")


def synthetic_klines(interval: str, limit: int, seed: int = 7) -> pd.DataFrame:
    """Serie tipo BTC con volatilidad variable, para probar sin red."""
    rng = np.random.default_rng(seed)
    step = INTERVAL_MS[interval]
    now = (int(time.time() * 1000) // step) * step
    n = limit
    vol = 0.006 * np.exp(np.cumsum(rng.normal(0, 0.05, n)) * 0.3)
    rets = rng.standard_t(4, n) * vol * 0.7 + 0.00005
    close = 80_000 * np.exp(np.cumsum(rets))
    open_ = np.concatenate([[close[0]], close[:-1]])
    wick = np.abs(rng.normal(0, 1, (2, n))) * vol * close * 0.6
    high = np.maximum(open_, close) + wick[0]
    low = np.minimum(open_, close) - wick[1]
    volume = rng.lognormal(6, 0.4, n)
    return pd.DataFrame({
        "open_time": now - step * np.arange(n)[::-1],
        "open": open_, "high": high, "low": low, "close": close,
        "volume": volume, "amount": volume * close,
    })


# ---------------------------------------------------------------- modelos

class BaselineModel:
    """Bootstrap por bloques de los retornos recientes. Sirve de referencia
    honesta: si Kronos no le gana a esto, no aporta nada."""
    name = "baseline (bootstrap)"

    def sample_paths(self, df: pd.DataFrame, horizon: int, samples: int, seed: int = 0) -> np.ndarray:
        rng = np.random.default_rng(seed)
        r = np.diff(np.log(df["close"].to_numpy()))
        r = r - r.mean()  # sin deriva: no inventamos tendencia
        block = 4
        last = df["close"].iloc[-1]
        out = np.empty((samples, horizon))
        for s in range(samples):
            path = []
            while len(path) < horizon:
                i = rng.integers(0, len(r) - block)
                path.extend(r[i:i + block])
            out[s] = last * np.exp(np.cumsum(path[:horizon]))
        return out


class KronosModel:
    def __init__(self, size: str = "small", device: str = "cpu"):
        kronos_dir = os.environ.get("KRONOS_DIR", os.path.join(HERE, "Kronos"))
        sys.path.insert(0, kronos_dir)
        from model import Kronos, KronosTokenizer, KronosPredictor  # del repo shiyu-coder/Kronos
        tok = KronosTokenizer.from_pretrained("NeoQuasar/Kronos-Tokenizer-base")
        mdl = Kronos.from_pretrained(f"NeoQuasar/Kronos-{size}")
        self.predictor = KronosPredictor(mdl, tok, device=device, max_context=512)
        self.name = f"Kronos-{size}"

    def sample_paths(self, df: pd.DataFrame, horizon: int, samples: int, seed: int = 0) -> np.ndarray:
        import torch
        step = int(df["open_time"].iloc[1] - df["open_time"].iloc[0])
        x_ts = pd.Series(pd.to_datetime(df["open_time"], unit="ms"))
        y_ts = pd.Series(pd.to_datetime(df["open_time"].iloc[-1] + step * np.arange(1, horizon + 1), unit="ms"))
        x = df[["open", "high", "low", "close", "volume", "amount"]].reset_index(drop=True)
        out = np.empty((samples, horizon))
        for s in range(samples):
            torch.manual_seed(seed + s)
            pred = self.predictor.predict(df=x, x_timestamp=x_ts, y_timestamp=y_ts,
                                          pred_len=horizon, T=1.0, top_p=0.9, sample_count=1,
                                          verbose=False)
            out[s] = pred["close"].to_numpy()
        return out


def load_model(choice: str):
    if choice == "baseline":
        return BaselineModel()
    try:
        m = KronosModel(size=os.environ.get("KRONOS_SIZE", "small"))
        print(f"[ok] Modelo cargado: {m.name}")
        return m
    except Exception as e:
        print(f"[aviso] Kronos no disponible ({e}). Uso el modelo base.")
        return BaselineModel()


# ---------------------------------------------------------------- lógica

def summarize(paths: np.ndarray, last: float) -> dict:
    q = np.percentile(paths, [5, 25, 50, 75, 95], axis=0)
    final = paths[:, -1]
    return {
        "quantiles": {k: v.round(2).tolist() for k, v in zip(["p5", "p25", "p50", "p75", "p95"], q)},
        "prob_up": float((final > last).mean()),
        "median_change_pct": float((np.median(final) / last - 1) * 100),
        "range_90_pct": [float((q[0, -1] / last - 1) * 100), float((q[4, -1] / last - 1) * 100)],
    }


def forecast(model, symbol, interval, context, horizon, samples, show=120):
    df = fetch_klines(symbol, interval, context)
    paths = model.sample_paths(df, horizon, samples)
    last = float(df["close"].iloc[-1])
    step = INTERVAL_MS[interval]
    t0 = int(df["open_time"].iloc[-1])
    hist = df.tail(show)
    return {
        "symbol": symbol, "interval": interval, "model": model.name, "demo": DEMO,
        "context": len(df), "horizon": horizon, "samples": samples,
        "generated_at": int(time.time() * 1000),
        "last_price": last,
        "history": hist[["open_time", "open", "high", "low", "close"]].round(2).values.tolist(),
        "future_times": [t0 + step * i for i in range(1, horizon + 1)],
        "paths": paths.round(2).tolist(),
        **summarize(paths, last),
    }


def backtest(model, symbol, interval, context, horizon, samples, windows):
    """Evaluación walk-forward: ¿acierta la dirección? ¿el rango p5-p95
    contiene el precio real? Se compara con el modelo base."""
    need = context + horizon * windows
    df = fetch_klines(symbol, interval, need)
    if len(df) < context + horizon:
        raise RuntimeError("Pocas velas para el backtest")
    base = BaselineModel()
    rows = []
    for w in range(windows):
        end = len(df) - horizon * (windows - w)
        if end < 64:
            continue
        ctx = df.iloc[max(0, end - context):end]
        actual = df["close"].iloc[end + horizon - 1]
        last = ctx["close"].iloc[-1]
        row = {"t": int(ctx["open_time"].iloc[-1]), "last": last, "actual": float(actual)}
        for tag, m in (("model", model), ("base", base)):
            p = m.sample_paths(ctx, horizon, samples, seed=w)[:, -1]
            lo, med, hi = np.percentile(p, [5, 50, 95])
            row[tag] = {
                "median": float(med),
                "dir_ok": bool(np.sign(med - last) == np.sign(actual - last)),
                "in_band": bool(lo <= actual <= hi),
                "abs_err_pct": float(abs(med / actual - 1) * 100),
            }
        rows.append(row)

    def agg(tag):
        n = len(rows)
        return {
            "hit_rate": sum(r[tag]["dir_ok"] for r in rows) / n,
            "coverage_90": sum(r[tag]["in_band"] for r in rows) / n,
            "mae_pct": sum(r[tag]["abs_err_pct"] for r in rows) / n,
        }
    return {"model": model.name, "windows": len(rows), "summary": {"model": agg("model"), "base": agg("base")}, "rows": rows}


# ---------------------------------------------------------------- servidor

MODEL = None
CACHE = {}


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        sys.stderr.write("%s\n" % (fmt % args))

    def _send(self, code, body, ctype="application/json"):
        data = body if isinstance(body, bytes) else json.dumps(body).encode()
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        u = urllib.parse.urlparse(self.path)
        qs = dict(urllib.parse.parse_qsl(u.query))
        if u.path in ("/", "/index.html"):
            with open(os.path.join(HERE, "index.html"), "rb") as f:
                return self._send(200, f.read(), "text/html; charset=utf-8")
        try:
            symbol = qs.get("symbol", "BTCUSDT").upper()
            interval = qs.get("interval", "1h")
            if interval not in INTERVAL_MS:
                raise ValueError("intervalo no válido")
            context = max(64, min(int(qs.get("context", 512)), 512))
            horizon = max(1, min(int(qs.get("horizon", 24)), 96))
            samples = max(1, min(int(qs.get("samples", 30)), 200))
            if u.path == "/api/forecast":
                key = (symbol, interval, context, horizon, samples)
                hit = CACHE.get(key)
                if hit and time.time() - hit[0] < 60:
                    return self._send(200, hit[1])
                res = forecast(MODEL, symbol, interval, context, horizon, samples)
                CACHE[key] = (time.time(), res)
                return self._send(200, res)
            if u.path == "/api/backtest":
                windows = max(3, min(int(qs.get("windows", 20)), 60))
                return self._send(200, backtest(MODEL, symbol, interval, context, horizon, min(samples, 30), windows))
            return self._send(404, {"error": "no encontrado"})
        except Exception as e:
            return self._send(500, {"error": str(e)})


def main():
    global DEMO, MODEL
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--demo", action="store_true", help="datos sintéticos, sin internet")
    ap.add_argument("--model", choices=["auto", "baseline"], default="auto")
    a = ap.parse_args()
    DEMO = a.demo
    MODEL = load_model(a.model)
    print(f"Panel en http://localhost:{a.port}  (modelo: {MODEL.name}{', DEMO' if DEMO else ''})")
    ThreadingHTTPServer(("127.0.0.1", a.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
