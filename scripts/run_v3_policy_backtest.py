"""Reproducible, validation-selected, causal V3 decision-policy experiment.

Run: .venv/Scripts/python.exe -m scripts.run_v3_policy_backtest
Only reads frozen models/data. TEST/FORWARD are evaluated after the validation
rule is frozen in this script; they never participate in selection.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict, replace
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from alpha_engine.ml.backtest import BacktestConfig, buy_and_hold, confidence_bins, execution_grid, simulate
from scripts.build_v2_dataset import V1, temporal_splits
from scripts.prepare_xgboost_v3_12m import AUDIT, CANONICAL, PAIRED, SPLIT_BOUNDS

ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "models/experiments"
OUT = ROOT / "reports/v3_policy_backtest"
THRESHOLDS = (0.52, 0.54, 0.55, 0.56, 0.58, 0.60)
FEES = (0.0, 0.001)
SLIPPAGES = (0.0, 0.0002, 0.0005)
REFERENCE_FEE = 0.001
REFERENCE_SLIPPAGE = 0.0002
DIAGNOSTIC_THRESHOLD = 0.55


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verified_model(stem: str):
    path = MODEL_DIR / f"{stem}.joblib"
    meta_path = MODEL_DIR / f"{stem}_metadata.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    if sha256(path) != meta["model_sha256"] or meta.get("experimental_only") is not True:
        raise ValueError(f"Frozen experimental model mismatch: {stem}")
    bundle = joblib.load(path)
    if bundle["features"] != meta["features"]:
        raise ValueError(f"Feature order mismatch: {stem}")
    return bundle["estimator"], meta


def summarize_gates(part: pd.DataFrame, grid: pd.DataFrame, config: BacktestConfig) -> list[dict]:
    p = part.p_up.to_numpy(dtype=float)
    y = part.target.to_numpy(dtype=int)
    rows = []
    for threshold in THRESHOLDS:
        current = replace(config, threshold=threshold)
        metrics, _, _ = simulate(grid, current)
        selected = (p >= threshold) | (1 - p >= threshold)
        n = int(selected.sum())
        up = int(y[selected].sum())
        down = n - up
        direction = (p[selected] >= 0.5).astype(int)
        quality = float((direction == y[selected]).mean()) if n else None
        balanced = None
        if up and down:
            balanced = float(0.5 * ((direction[y[selected] == 1] == 1).mean()
                                    + (direction[y[selected] == 0] == 0).mean()))
        rows.append({"threshold": threshold, "coverage_pct": 100 * n / len(part),
                     "covered_n": n, "accuracy": quality, "balanced_accuracy": balanced,
                     "predicted_UP": int((p >= threshold).sum()),
                     "predicted_DOWN": int((1 - p >= threshold).sum()),
                     "actual_UP_covered": up, "actual_DOWN_covered": down,
                     "trades": metrics["trades"], "net_pnl": metrics["net_pnl"],
                     "return_pct": metrics["return_pct"],
                     "max_drawdown_pct": metrics["max_drawdown_pct"]})
    return rows


def choose_validation_candidate(gates: list[dict], buy_hold: dict) -> tuple[float, bool, str]:
    """Predeclared quality and cost filter; never rank by PnL or trade count."""
    eligible = [row for row in gates if row["trades"] >= 30
                and row["net_pnl"] > max(0, buy_hold["net_pnl"])
                and row["max_drawdown_pct"] <= buy_hold["max_drawdown_pct"]
                and row["balanced_accuracy"] is not None]
    if not eligible:
        return DIAGNOSTIC_THRESHOLD, False, (
            "Ningún gate alcanzó en VALIDATION al menos 30 trades, PnL neto superior a "
            "no operar y buy-and-hold, y drawdown no mayor que buy-and-hold bajo costo de referencia. "
            "Se congela 0.55 solo como política diagnóstica, sin recomendar BUY/SELL.")
    chosen = max(eligible, key=lambda row: (row["balanced_accuracy"], row["coverage_pct"]))
    return chosen["threshold"], True, (
        "Gate elegible por los criterios predefinidos en VALIDATION; entre elegibles "
        "se eligió mayor balanced accuracy y luego cobertura. No se usó TEST/FORWARD.")


def chart(curve: pd.DataFrame, destination: Path, split: str) -> None:
    # A deterministic display reduction; drawdown metrics use every observation.
    stride = max(1, len(curve) // 2500)
    shown = curve.iloc[::stride]
    if shown.index[-1] != curve.index[-1]:
        shown = pd.concat([shown, curve.iloc[[-1]]])
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.08,
                        subplot_titles=("Capital virtual (USD)", "Drawdown (%)"))
    fig.add_trace(go.Scatter(x=shown.timestamp, y=shown.equity, mode="lines", name="Equity"), row=1, col=1)
    fig.add_trace(go.Scatter(x=shown.timestamp, y=shown.drawdown_pct, mode="lines", name="Drawdown"), row=2, col=1)
    fig.update_layout(title=f"V3 política diagnóstica/candidata — {split}", template="plotly_white", height=720)
    fig.write_html(destination, include_plotlyjs="cdn", full_html=True)


def markdown(report: dict) -> str:
    def number(value, digits=2):
        return "—" if value is None else f"{value:.{digits}f}"

    lines = ["# V3: simulación histórica de política long only", "",
             "## A. Metodología", "",
             "Binance Spot BTCUSDT 1m, cohorte congelada de 12 meses. V3 se entrenó solo en TRAIN. "
             "VALIDATION compara gates; TEST y FORWARD se reportan después de congelar la regla. "
             "Cada período arranca sin posición y termina liquidado.",
             f"Capital virtual USD {report['config']['initial_capital']:,.2f}; cada BUY asigna "
             f"{report['config']['allocation']:.0%} del efectivo disponible. No es recomendación financiera.", "",
             "## B. Regla causal de ejecución", "",
             "Una vela fuente con apertura t cierra justo antes de t+1 minuto. Sus features y P(UP) "
             "se consideran disponibles tras ese cierre. BUY/SELL se ejecuta al open observado en t+2 minutos, "
             "un minuto completo después. Se rechazan gaps y no se ejecuta al close fuente. "
             "El modelo fue entrenado para el close de t+1; ejecutar en t+2 cambia el horizonte económico. "
             "El último open de cada período solo permite liquidar.", "",
             "## C. Thresholds evaluados en VALIDATION", "",
             "Costo de referencia: fee 0.10% por lado y slippage 0.02% por lado. "
             "Cobertura es la fracción con P(UP) o P(DOWN) sobre el gate; incluye señales DOWN que no generan BUY.", "",
             "| Gate | Cobertura | Trades | PnL neto USD | Return | Max DD | Accuracy cubierta | Bal. accuracy | UP / DOWN señales |",
             "|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for row in report["validation_gates"]:
        lines.append(f"| {row['threshold']:.2f} | {row['coverage_pct']:.2f}% | {row['trades']} | "
                     f"{row['net_pnl']:+.2f} | {row['return_pct']:+.2f}% | {row['max_drawdown_pct']:.2f}% | "
                     f"{number(row['accuracy'], 3)} | {number(row['balanced_accuracy'], 3)} | "
                     f"{row['predicted_UP']} / {row['predicted_DOWN']} |")
    lines += ["", "## D. Política candidata", "", report["policy"]["reason"], "",
              "## Métricas predictivas por confianza", "",
              "Confianza = max(P(UP), P(DOWN)). Accuracy y balanced accuracy se calculan por dirección predicha; "
              "balanced accuracy es indefinida si en el bin aparece una sola clase real.", "",
              "| Período | Bin [min, max) | n | Cobertura | Accuracy | Bal. accuracy | UP/DOWN reales | UP/DOWN predichos |",
              "|---|---|---:|---:|---:|---:|---:|---:|"]
    for period, info in report["periods"].items():
        for row in info["confidence_bins"]:
            lines.append(f"| {period} | [{row['min_confidence']:.2f}, {row['max_confidence']:.2f}) | "
                         f"{row['n']} | {row['coverage_pct']:.2f}% | {number(row['accuracy'], 3)} | "
                         f"{number(row['balanced_accuracy'], 3)} | {row['actual_UP']}/{row['actual_DOWN']} | "
                         f"{row['predicted_UP']}/{row['predicted_DOWN']} |")
    for period, heading in (("TEST", "E"), ("FORWARD", "F")):
        s = report["periods"][period]["strategy"]
        lines += ["", f"## {heading}. {period}", "",
                  f"Fuente UTC: {report['periods'][period]['source_range_utc'][0]} a "
                  f"{report['periods'][period]['source_range_utc'][1]}. "
                  f"Grilla de valoración: opens UTC {report['periods'][period]['execution_range_utc'][0]} a "
                  f"{report['periods'][period]['execution_range_utc'][1]}; cada operación usa el open "
                  "t+2 de su señal fuente.", "",
                  "| Métrica | Valor |", "|---|---:|"]
        for key, label, suffix in (
            ("initial_capital", "Capital inicial", " USD"), ("final_capital", "Capital final", " USD"),
            ("return_pct", "Return", "%"), ("gross_pnl", "PnL bruto", " USD"),
            ("net_pnl", "PnL neto", " USD"), ("trades", "Trades", ""),
            ("winning_trades", "Ganadores", ""), ("losing_trades", "Perdedores", ""),
            ("flat_trades", "Neutros", ""), ("win_rate_pct", "Win rate", "%"),
            ("average_trade_net", "Average trade neto", " USD"),
            ("max_drawdown_pct", "Max drawdown", "%"), ("fees_paid", "Fees pagadas", " USD"),
            ("slippage_cost", "Costo slippage", " USD"), ("time_exposure_pct", "Exposición temporal", "%"),
            ("BUY_count", "BUY", ""), ("SELL_count", "SELL", ""),
            ("HOLD_count", "HOLD", ""), ("WAIT_count", "WAIT", "")):
            value = s[key]
            formatted = str(value) if isinstance(value, int) else number(value)
            lines.append(f"| {label} | {formatted}{suffix} |")
    lines += ["", "## G. Costos", "",
              "Escenarios ilustrativos, sin afirmar que sean las tarifas reales del usuario ni del exchange. "
              "Fee y slippage se aplican en BUY y SELL.", "",
              "| Período | Fee/lado | Slippage/lado | PnL neto USD | Return |",
              "|---|---:|---:|---:|---:|"]
    for period, info in report["periods"].items():
        for row in info["cost_scenarios"]:
            lines.append(f"| {period} | {row['fee']:.2%} | {row['slippage']:.2%} | "
                         f"{row['net_pnl']:+.2f} | {row['return_pct']:+.2f}% |")
    lines += ["", "## H. Baselines", "",
              "Buy-and-hold invierte 100% al primer open y liquida al último, con el costo de referencia. "
              "V1 experimental usa el mismo gate y la misma ejecución con cohorte pareada; fue entrenado "
              "en marzo-julio 2026, ventana más corta que V3. No se usa el V1 productivo porque su "
              "entrenamiento solapa TEST/FORWARD.", "",
              "| Período | No operar | Buy-and-hold | V1 misma regla | V3 regla congelada |",
              "|---|---:|---:|---:|---:|"]
    for period, info in report["periods"].items():
        b = info["baselines"]
        lines.append(f"| {period} | {b['no_trade']['return_pct']:+.2f}% | "
                     f"{b['buy_and_hold']['return_pct']:+.2f}% | "
                     f"{b['V1_same_rule']['return_pct']:+.2f}% | "
                     f"{info['strategy']['return_pct']:+.2f}% |")
    lines += ["", "## I. Limitaciones", "",
              "TEST y FORWARD ya fueron observados en el desarrollo previo de los modelos; no equivalen "
              "a evidencia prospectiva nueva. El precio open histórico supone ejecución completa al "
              "precio fijado más slippage constante; no modela latencia variable, spread, profundidad, "
              "impacto, tamaños mínimos ni restricciones operativas. La política es long only, sin margen "
              "ni operaciones reales. El horizonte de señal H1 difiere del horizonte tras el retraso conservador. "
              "Al elevar el gate aumenta la accuracy aparente por señales DOWN; balanced accuracy y "
              "número de BUY muestran que eso no demuestra utilidad económica.", "",
              "## J. Archivos", "",
              "`summary.json`, `trades_{validation,test,forward}.csv`, "
              "`equity_{validation,test,forward}.csv` y "
              "`equity_drawdown_{validation,test,forward}.html` se generan con "
              "`.venv/Scripts/python.exe -m scripts.run_v3_policy_backtest`. "
              "Código: `alpha_engine/ml/backtest.py` y `scripts/run_v3_policy_backtest.py`.", "",
              "## K. Tests", "",
              "Las pruebas de timing, costos, liquidación, gaps y selección están en "
              "`tests/test_v3_policy_backtest.py`. El resultado de ejecución se informa por separado.", ""]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capital", type=float, default=1000.0)
    parser.add_argument("--allocation", type=float, default=0.20)
    parser.add_argument("--output", type=Path, default=OUT)
    args = parser.parse_args()
    config = BacktestConfig(initial_capital=args.capital, allocation=args.allocation,
                            threshold=DIAGNOSTIC_THRESHOLD, fee=REFERENCE_FEE,
                            slippage=REFERENCE_SLIPPAGE)
    audit = json.loads(AUDIT.read_text(encoding="utf-8"))
    if sha256(PAIRED) != audit["cohort"]["sha256"] or sha256(CANONICAL) != audit["canonical"]["sha256"]:
        raise ValueError("Frozen data fingerprint mismatch")
    v3, v3_meta = verified_model("xgboost_v3_market_flow_12m")
    v1, v1_meta = verified_model("xgboost_v1_retrained_6m")
    if v3_meta["dataset_sha256"] != audit["cohort"]["sha256"] or v1_meta["features"] != list(V1):
        raise ValueError("Model and cohort metadata mismatch")
    frame = pd.read_parquet(PAIRED)
    candles = pd.read_parquet(CANONICAL, columns=["timestamp", "open"])
    splits, split_audit = temporal_splits(frame, SPLIT_BOUNDS)
    if split_audit != audit["splits"]:
        raise ValueError("Split audit mismatch")
    bounds = {name: (pd.Timestamp(start, tz="UTC"), pd.Timestamp(stop, tz="UTC"))
              for name, (start, stop) in SPLIT_BOUNDS.items() if name != "TRAIN"}
    def prepare(name: str):
        part = splits[name].copy()
        part["p_up"] = v3.predict_proba(part[v3_meta["features"]])[:, 1]
        part["v1_p_up"] = v1.predict_proba(part[v1_meta["features"]])[:, 1]
        start, stop = bounds[name]
        bars = candles.loc[(candles.timestamp >= start) & (candles.timestamp < stop)]
        return part, execution_grid(bars, part[["timestamp", "p_up"]])

    validation, validation_grid = prepare("VALIDATION")
    bh_validation = buy_and_hold(validation_grid, config.initial_capital, config.fee, config.slippage)
    gates = summarize_gates(validation, validation_grid, config)
    threshold, candidate, reason = choose_validation_candidate(gates, bh_validation)
    frozen = replace(config, threshold=threshold)
    prepared = {"VALIDATION": (validation, validation_grid)}
    # Only now score the held-out periods; neither can affect the frozen gate.
    for name in ("TEST", "FORWARD"):
        prepared[name] = prepare(name)
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    report = {"created_at_utc": datetime.now(timezone.utc).isoformat(),
              "data_sha256": audit["cohort"]["sha256"], "canonical_sha256": audit["canonical"]["sha256"],
              "models": {"V3": {"name": v3_meta["model_version"], "sha256": v3_meta["model_sha256"]},
                         "V1": {"name": v1_meta["name"], "sha256": v1_meta["model_sha256"]}},
              "config": asdict(frozen), "fee_scenarios": FEES, "slippage_scenarios": SLIPPAGES,
              "method": {"market": "Binance Spot BTCUSDT 1m historical opens; virtual spot only",
                         "signal": "P(UP) from source candle t, known after its close at t+1m",
                         "fill": "open of candle t+2m (one full minute after signal candle close)",
                         "policy": "flat: BUY if P(UP)>=gate else WAIT; long: SELL if P(DOWN)>=gate else HOLD",
                         "boundary": "each split starts flat; final open closes any long; no cross-split position",
                         "selection": "VALIDATION only; minimum 30 trades, net > no-trade and buy-hold, DD <= buy-hold; rank eligible by balanced accuracy then coverage",
                         "observed_periods": "TEST and FORWARD were already observed during earlier model development; not fully prospective evidence",
                         "costs": "illustrative scenarios, not verified user/exchange fees; constant deterministic fills without spread or market impact"},
              "validation_gates": gates,
              "policy": {"threshold": threshold, "candidate_supported": candidate, "reason": reason},
              "periods": {}}
    for name in ("VALIDATION", "TEST", "FORWARD"):
        part, grid = prepared[name]
        selected_metrics, trades, curve = simulate(grid, frozen)
        trades.to_csv(output / f"trades_{name.lower()}.csv", index=False)
        curve.to_csv(output / f"equity_{name.lower()}.csv", index=False)
        chart(curve, output / f"equity_drawdown_{name.lower()}.html", name)
        scenarios = []
        for fee in FEES:
            for slippage in SLIPPAGES:
                scenario = replace(frozen, fee=fee, slippage=slippage)
                result, _, _ = simulate(grid, scenario)
                scenarios.append({"fee": fee, "slippage": slippage, **result})
        v1_grid = execution_grid(grid[["timestamp", "open"]],
                                 part[["timestamp", "v1_p_up"]].rename(columns={"v1_p_up": "p_up"}))
        v1_metrics, _, _ = simulate(v1_grid, frozen)
        report["periods"][name] = {
            "source_range_utc": [part.timestamp.iloc[0].isoformat(), part.timestamp.iloc[-1].isoformat()],
            "execution_range_utc": [grid.timestamp.iloc[0].isoformat(), grid.timestamp.iloc[-1].isoformat()],
            "source_rows": len(part), "execution_bars": len(grid),
            "confidence_bins": confidence_bins(part.p_up.to_numpy(), part.target.to_numpy()),
            "strategy": selected_metrics, "cost_scenarios": scenarios,
            "baselines": {"no_trade": {"initial_capital": config.initial_capital,
                                       "final_capital": config.initial_capital, "return_pct": 0,
                                       "net_pnl": 0, "max_drawdown_pct": 0, "trades": 0},
                          "buy_and_hold": buy_and_hold(grid, config.initial_capital, config.fee, config.slippage),
                          "V1_same_rule": v1_metrics},
            "files": {"trades": f"trades_{name.lower()}.csv", "equity": f"equity_{name.lower()}.csv",
                      "chart": f"equity_drawdown_{name.lower()}.html"}}
        print(f"{name}: gate={threshold:.2f}, trades={selected_metrics['trades']}, "
              f"net={selected_metrics['net_pnl']:.2f}, return={selected_metrics['return_pct']:.2f}%", flush=True)
    (output / "summary.json").write_text(json.dumps(report, indent=2, ensure_ascii=False,
                                              allow_nan=False, default=str) + "\n", encoding="utf-8")
    (output / "report.md").write_text(markdown(report), encoding="utf-8")
    print(f"Candidate supported: {candidate}. Report: {output / 'summary.json'}", flush=True)


if __name__ == "__main__":
    main()
