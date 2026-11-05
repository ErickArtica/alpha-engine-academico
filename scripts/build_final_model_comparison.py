"""Build the final, read-only FASE 8 evidence pack from frozen reports.

No estimator is loaded or fitted. LIVE is descriptive and never selects a
candidate. Existing phase reports are inputs; this script does not rewrite them.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"
MODEL = ROOT / "models/xgboost_v1.joblib"
EXPECTED_SHA = "8e23d25eeb7fcec82e1f71faf5d127ce4479318b9b5b1a76fc36b6691d147a8c"
PHASE_FILES = {
    "phase1": "live_model_diagnostic.json",
    "phase2": "calibration_analysis_v1.json",
    "parity": "feature_parity_v1.json",
    "microstructure": "microstructure_feasibility_v1.json",
    "tfi": "model_ablation_tfi_v1.json",
    "tfi_walk": "walk_forward_tfi_v1.json",
    "hold": "target_hold_experiment_v1.json",
    "hold_walk": "target_hold_walk_forward_v1.json",
    "horizon": "horizon_analysis_v1.json",
    "horizon_walk": "horizon_walk_forward_v1.json",
    "news_audit": "alpha_vantage_data_audit_v1.json",
    "news": "news_sentiment_ablation_v1.json",
    "news_walk": "news_sentiment_walk_forward_v1.json",
    "live_final": "final_live_evaluation_v1.json",
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_inputs(root: Path = ROOT) -> dict:
    return {name: json.loads((root / "reports" / file).read_text(encoding="utf-8"))
            for name, file in PHASE_FILES.items()}


def validate_inputs(data: dict, model_sha: str) -> None:
    if model_sha != EXPECTED_SHA:
        raise ValueError("Product model SHA-256 differs from frozen reference")
    if data["tfi"]["baseline_reproduction"]["test_class_flips"] != 0:
        raise ValueError("Experimental H1 baseline does not reproduce classes")
    if data["tfi"]["baseline_reproduction"]["test_max_abs_prob_difference"] != 0:
        raise ValueError("Experimental H1 baseline does not reproduce probabilities")
    if sorted(data["hold"]["candidate_lock"]["candidates_bps"]) != [0.5, 1.0]:
        raise ValueError("HOLD candidate lock changed")
    if data["horizon"]["candidate_lock"]["candidates_minutes"] != [1, 5]:
        raise ValueError("Horizon candidate lock changed")
    if data["live_final"]["model_version"] != "xgboost_v1":
        raise ValueError("LIVE report contains another model")


def _m(metrics: dict) -> dict:
    keys = ("n", "accuracy", "balanced_accuracy", "roc_auc", "pr_auc_up",
            "precision_up", "recall_up", "precision_down", "recall_down",
            "macro_f1", "brier", "log_loss", "predicted_up_fraction")
    return {key: metrics.get(key) for key in keys}


def candidate_registry(data: dict) -> list[dict]:
    """Keep incompatible tasks separate and attach each candidate's control."""
    tfi = data["tfi"]["models"]
    hold = data["hold"]["results"]
    horizon = data["horizon"]["final_test"]["common_TEST_results"]
    news = data["news"]["test"]
    return [
        {"id": "P0", "name": "xgboost_v1", "task": "binary H1", "features": "8 technical",
         "control": None, "status": "PRODUCTION_REFERENCE", "test": _m(tfi["A"]["test"]),
         "reason": "Exact experimental reproduction; no replacement has robust paired evidence. LIVE discrimination is weak."},
        {"id": "P1", "name": "H1 + TFI", "task": "binary H1", "features": "8 technical + TFI",
         "control": "P0 on Phase 4 cohort", "status": "NOT_SUPPORTED", "test": _m(tfi["B"]["test"]),
         "reason": "TEST ROC AUC lower; paired intervals include zero; mixed folds."},
        {"id": "P2", "name": "HOLD 0.5 bp", "task": "ternary H1", "features": "8 technical",
         "control": "ternary majority/always-HOLD; binary is conceptual reference only",
         "status": "NOT_SUPPORTED", "test": {"accuracy": hold["0.5"]["test"]["accuracy"],
             "balanced_accuracy": hold["0.5"]["test"]["balanced_accuracy"],
             "macro_f1": hold["0.5"]["test"]["macro_f1"],
             "signal_coverage": hold["0.5"]["test"]["signal_coverage"],
             "directional_accuracy": hold["0.5"]["test"]["directional_accuracy"]},
         "reason": "Conceptual abstention benefit did not yield better directional quality; separate multiclass task."},
        {"id": "P3", "name": "HOLD 1 bp", "task": "ternary H1", "features": "8 technical",
         "control": "ternary majority/always-HOLD; binary is conceptual reference only",
         "status": "NOT_SUPPORTED", "test": {"accuracy": hold["1.0"]["test"]["accuracy"],
             "balanced_accuracy": hold["1.0"]["test"]["balanced_accuracy"],
             "macro_f1": hold["1.0"]["test"]["macro_f1"],
             "signal_coverage": hold["1.0"]["test"]["signal_coverage"],
             "directional_accuracy": hold["1.0"]["test"]["directional_accuracy"]},
         "reason": "Lower signal coverage without stronger directional quality; separate multiclass task."},
        {"id": "P4", "name": "H3", "task": "binary H3", "features": "8 technical",
         "control": "H1 on Phase 6 development cohort", "status": "NOT_SUPPORTED",
         "validation": _m(data["horizon"]["development"]["validation"]["H3"]["metrics"]),
         "test": None, "reason": "Not locked for final TEST; weaker development ROC AUC and no stable advantage."},
        {"id": "P5", "name": "H5", "task": "binary H5", "features": "8 technical",
         "control": "H1 on identical Phase 6 common cohort", "status": "RESEARCH_ONLY",
         "test": _m(horizon["H5"]["metrics"]),
         "reason": "Less tiny target movement and longer window, but TEST ROC AUC/macro F1 lower and class shift toward UP."},
        {"id": "P6", "name": "H1 + Alpha Vantage sentiment", "task": "binary H1",
         "features": "8 technical + 5 news", "control": "H1 no-news on identical Phase 7 rows",
         "status": "NOT_SUPPORTED", "test": _m(news["H1"]["B"]["metrics"]),
         "reason": "Paired TEST worse in core metrics; historical provider availability unobserved."},
        {"id": "P7", "name": "H5 + Alpha Vantage sentiment", "task": "binary H5",
         "features": "8 technical + 5 news", "control": "H5 no-news on identical Phase 7 rows",
         "status": "NOT_SUPPORTED", "test": _m(news["H5"]["B"]["metrics"]),
         "reason": "Paired TEST ROC AUC/balanced accuracy lower; no stable fold gain; availability unobserved."},
    ]


def scorecard(data: dict) -> list[dict]:
    """Qualitative evidence ledger without a synthetic numerical ranking."""
    descriptions = [
        ("P0", "LIVE ROC AUC 0.5149; TEST 0.5258", "LIVE 87.4% DOWN", "reference only",
         "TEST 0.5258 AUC; exact reproduction", "1,597 evaluated; weak near-baseline",
         "OHLCV LIVE available", "low with causal candles", "current/low", "PRODUCTION_REFERENCE"),
        ("P1", "TEST AUC 0.5237, below control 0.5258", "small TEST class change", "mixed 2/4 AUC-positive folds",
         "paired CI includes zero", "none; not deployed", "historical kline TFI available",
         "low with source-candle cutoff", "moderate extra data", "NOT_SUPPORTED"),
        ("P2", "ternary balanced acc. 0.3832", "HOLD predicted 4.6% on TEST",
         "development not robust", "95.4% coverage; 45.0% directional accuracy", "none; not deployed",
         "OHLCV available", "low with future label purge", "moderate target/evaluation change", "NOT_SUPPORTED"),
        ("P3", "ternary balanced acc. 0.3923", "HOLD predicted more often",
         "development not robust", "83.3% coverage; 39.7% directional accuracy", "none; not deployed",
         "OHLCV available", "low with future label purge", "moderate target/evaluation change", "NOT_SUPPORTED"),
        ("P4", "validation AUC 0.5189 vs H1 0.5413", "varies by fold", "no stable AUC gain",
         "not locked for final TEST", "none; not deployed", "OHLCV available",
         "low with horizon-aware purge", "moderate horizon/evaluation change", "NOT_SUPPORTED"),
        ("P5", "TEST AUC 0.5232 vs H1 0.5278", "TEST 75.5% UP",
         "macro F1 helped in folds; AUC did not", "TEST macro F1 0.4831 vs H1 0.5171",
         "none; not deployed", "OHLCV available", "low with horizon-aware purge",
         "moderate horizon/evaluation change", "RESEARCH_ONLY"),
        ("P6", "TEST AUC 0.5226 vs paired 0.5278", "similar TEST class share",
         "AUC fell in 3/4 folds", "paired core metrics worse", "none; not deployed",
         "news 598 articles; 15m coverage 17.7%", "publication-safe only; first-seen unknown",
         "high external API/dependency", "NOT_SUPPORTED"),
        ("P7", "TEST AUC 0.5213 vs paired 0.5232", "H5 UP-biased",
         "AUC fell in 4/4 folds", "paired ROC/balanced acc. worse", "none; not deployed",
         "news 598 articles; 60m coverage 48.3%", "publication-safe only; first-seen unknown",
         "high external API/dependency", "NOT_SUPPORTED"),
    ]
    keys = ("candidate", "predictive_discrimination", "class_balance", "walk_forward_stability",
            "test_evidence", "live_evidence", "data_availability", "leakage_risk",
            "operational_complexity", "result")
    return [dict(zip(keys, values, strict=True)) for values in descriptions]


def build_comparison(data: dict, model_sha: str, commit: str, now: str) -> dict:
    validate_inputs(data, model_sha)
    source_hashes = {name: sha256(REPORTS / file) for name, file in PHASE_FILES.items()}
    live = data["live_final"]
    tfi = data["tfi"]
    horizon = data["horizon"]["final_test"]["common_TEST_results"]
    news = data["news"]["test"]
    historical_gates = {}
    for cohort in ("validation_holdout", "test", "live_phase2"):
        gate = next(item for item in data["phase2"]["datasets"][cohort]["models"]["uncalibrated"]["confidence_gates"]
                    if item["threshold"] == .60)
        historical_gates[cohort] = {key: gate[key] for key in
                                    ("n", "coverage", "correct", "accuracy", "wilson_95", "always_down_accuracy_selected")}
    return {
        "generated_at_utc": now, "base_commit": commit, "product_sha256": model_sha,
        "inputs_sha256": source_hashes,
        "baseline_reproduction": tfi["baseline_reproduction"],
        "candidate_registry": candidate_registry(data),
        "scorecard": scorecard(data),
        "paired_controls": {
            "tfi": {"A": _m(tfi["models"]["A"]["test"]),
                    "B": _m(tfi["models"]["B"]["test"]),
                    "uncertainty": tfi["test_uncertainty"],
                    "folds_B_minus_A": [{"fold": row["fold"],
                                         "roc_auc": row["B"]["roc_auc"] - row["A"]["roc_auc"],
                                         "balanced_accuracy": row["B"]["balanced_accuracy"] - row["A"]["balanced_accuracy"]}
                                        for row in data["tfi_walk"]["folds"]]},
            "horizon_common_test": {"H1": _m(horizon["H1"]["metrics"]),
                                    "H5": _m(horizon["H5"]["metrics"])},
            "news": {key: {"A": _m(news[key]["A"]["metrics"]),
                            "B": _m(news[key]["B"]["metrics"]),
                            "uncertainty": news[key]["bootstrap_B_minus_A"]}
                     for key in ("H1", "H5")},
        },
        "live": {"cutoff_utc": live["analyzed_at_utc"], "n": live["evaluated"],
                 "accuracy": live["accuracy"], "balanced_accuracy": live["balanced_accuracy"],
                 "roc_auc_up": live["discrimination"]["roc_auc_up"],
                 "predicted_up_fraction": live["discrimination"]["predicted_up_fraction"],
                 "gate_60": live["gate_60_final"]},
        "historical_gate_60": historical_gates,
        "gate_decision": {
            "class_threshold": .50, "confidence_gate": .60,
            "recommendation": "KEEP_AS_EXPERIMENTAL_OPERATIONAL_GUARDRAIL_REMOVE_AS_QUALITY_CLAIM",
            "production_changed": False,
            "basis": "8/1597 LIVE rows passed; 5/8 correct versus 6/8 always DOWN on those rows. TEST and earlier LIVE cutoffs vary; no validated uplift."},
        "model_decision": "YES_AS_CURRENT_REFERENCE",
        "model_decision_reason": "No candidate demonstrated a robust, stable paired improvement. This does not establish strong predictive performance.",
        "limitations": ["Historical TEST price period has been inspected repeatedly.",
                        "LIVE was used only for descriptive external evaluation.",
                        "Historical news availability is publication-time proxy, not first-seen.",
                        "Minute observations and news clusters are temporally dependent."],
    }


def write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def pct(value: float | None) -> str:
    return "N/A" if value is None else f"{value:.2%}"


def comparison_md(report: dict, data: dict) -> str:
    live = data["live_final"]
    gate = live["gate_60_final"]
    lines = ["# Comparación final de modelos — FASE 8", "",
             f"Corte LIVE: {live['analyzed_at_utc']} UTC; {live['evaluated']} evaluaciones. Commit base `{report['base_commit']}`; SHA productivo `{report['product_sha256']}`.",
             "", "## Decisión", "",
             "**YES_AS_CURRENT_REFERENCE:** xgboost_v1, H1, ocho features, target binario. Ningún experimento demostró una mejora pareada robusta. No significa que xgboost_v1 sea un predictor fuerte.",
             "", "## Registro de candidatos", "",
             "| ID | Candidato | Tarea | Control legítimo | Estado | Evidencia |",
             "| --- | --- | --- | --- | --- | --- |"]
    for item in report["candidate_registry"]:
        lines.append(f"| {item['id']} | {item['name']} | {item['task']} | {item['control'] or 'referencia productiva'} | {item['status']} | {item['reason']} |")
    lines += ["", "## Scorecard de evidencia", "",
              "Sin score numérico global; cada candidato se compara con su control legítimo.", "",
              "| Candidate | Predictive discrimination | Class balance | Walk-forward stability | TEST evidence | LIVE evidence | Data availability | Leakage risk | Operational complexity | Result |",
              "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    for row in report["scorecard"]:
        lines.append("| " + " | ".join(str(row[key]) for key in (
            "candidate", "predictive_discrimination", "class_balance", "walk_forward_stability",
            "test_evidence", "live_evidence", "data_availability", "leakage_risk",
            "operational_complexity", "result")) + " |")
    lines += ["", "## Comparaciones pareadas TEST", "",
              "| Experimento | Control ROC AUC | Candidato ROC AUC | Control balanced acc. | Candidato balanced acc. |",
              "| --- | ---: | ---: | ---: | ---: |"]
    for name, item in report["paired_controls"].items():
        if name == "tfi":
            pairs = [("TFI", item["A"], item["B"])]
        elif name == "horizon_common_test":
            pairs = [("H5 vs H1", item["H1"], item["H5"])]
        else:
            pairs = [(f"NEWS {h}", row["A"], row["B"]) for h, row in item.items()]
        for label, control, candidate in pairs:
            lines.append(f"| {label} | {control['roc_auc']:.4f} | {candidate['roc_auc']:.4f} | {control['balanced_accuracy']:.4f} | {candidate['balanced_accuracy']:.4f} |")
    lines += ["", "HOLD δ=0.5/1 bp es multiclase: su accuracy no se compara directamente con el binario. Sus controles son majority/always-HOLD y su calidad direccional/coverage. H3 no llegó al TEST final por el lock anterior.",
              "", "## LIVE y baselines", "",
              f"Accuracy {pct(live['accuracy'])}, balanced accuracy {pct(live['balanced_accuracy'])}, ROC AUC {live['discrimination']['roc_auc_up']:.4f}, macro F1 {live['macro_f1']:.4f}. Predice UP {pct(live['predicted_classes']['UP']['share'])}; UP real {pct(live['observed_classes']['UP']['share'])}.",
              f"Always DOWN {pct(live['baselines']['train_majority_down']['accuracy'])}; always UP {pct(live['baselines']['always_up']['accuracy'])}; previous movement {pct(live['baselines']['previous_movement_same']['accuracy'])} sobre las mismas filas.",
              "", "## Gate de 60%", "",
              "| Corte | Casos | Cobertura | Accuracy condicionada | Always DOWN mismas filas | Wilson 95% (independencia) |",
              "| --- | ---: | ---: | ---: | ---: | ---: |"]
    for key, title in (("validation_holdout", "VALIDATION holdout F2"), ("test", "TEST original F2"),
                       ("live_phase2", "LIVE F2")):
        item = report["historical_gate_60"][key]
        lines.append(f"| {title} | {item['n']} | {pct(item['coverage'])} | {pct(item['accuracy'])} | {pct(item['always_down_accuracy_selected'])} | {pct(item['wilson_95'][0])}–{pct(item['wilson_95'][1])} |")
    lines.append(f"| LIVE final F8 | {gate['count']} | {pct(gate['coverage'])} | {pct(gate['accuracy'])} | {pct(gate['always_down_accuracy_same_rows'])} | {pct(gate['accuracy_wilson_95_independence_optimistic'][0])}–{pct(gate['accuracy_wilson_95_independence_optimistic'][1])} |")
    lines += ["",
              f"Sólo {gate['count']}/{live['evaluated']} ({pct(gate['coverage'])}) pasaron. Accuracy condicional {pct(gate['accuracy'])} ({gate['correct']}/{gate['count']}); always DOWN sobre esas mismas filas {pct(gate['always_down_accuracy_same_rows'])}. Wilson 95% ingenuo: {pct(gate['accuracy_wilson_95_independence_optimistic'][0])}–{pct(gate['accuracy_wilson_95_independence_optimistic'][1])}; la dependencia temporal amplía la cautela.",
              "Recomendación: mantenerlo sólo como guardrail operacional experimental mientras se revisa el producto; eliminar toda afirmación de que 60% es calidad validada. No se cambia producción en esta fase.",
              "", "## Reproducibilidad y límites", "",
              "El baseline A de FASE 4 reprodujo probabilidades y clases exactamente (diferencia máxima 0; cero flips). Los fingerprints de cada reporte están en el JSON. LIVE no participó en fit ni selección. TEST histórico fue observado en fases previas, por lo que la evidencia debe interpretarse con esa limitación.",
              "", "## Regla de gobernanza", "",
              "Promover sólo tras comparación pareada y causal, walk-forward temporal, baseline relevante, mejora en varias métricas, TEST/forward razonable, disponibilidad operacional y ausencia de leakage conocido. No existe un umbral universal de mejora.", ""]
    return "\n".join(lines)


def model_card(data: dict, report: dict, metadata: dict) -> str:
    live = data["live_final"]
    train, val, test = (metadata["split"][part] for part in ("train", "validation", "test"))
    return f"""# Model card — xgboost_v1 (referencia actual)

**Activo:** BTCUSDT. **Intervalo de entrada:** 1 minuto. **Horizonte:** cierre de la siguiente vela de 1 minuto. **Tarea:** clasificación binaria UP/DOWN.

**Target:** UP si `next_close > source_close`; DOWN si baja o empata. **Regla de clase:** P(UP) > 0.50 → UP. **Gate prescriptivo vigente en código:** max(P) ≥ 0.60; es un guardrail experimental, no un umbral de calidad validado.

**Features:** {', '.join(metadata['features'])}. Las ocho se calculan causalmente hasta la vela fuente cerrada. **SHA-256:** `{report['product_sha256']}`.

## Datos y método

43.199 velas cerradas entre {metadata['data_range']['start']} y {metadata['data_range']['end']}. TRAIN {train['rows']:,} ({train['start']} a {train['end']}); VALIDATION {val['rows']:,} ({val['start']} a {val['end']}); TEST {test['rows']:,} ({test['start']} a {test['end']}). Orden temporal con purga de una fila por frontera.

TEST histórico original: accuracy {metadata['evaluation']['test']['accuracy']:.2%}; recall UP {metadata['evaluation']['test']['recall']:.2%}. Reproducción experimental exacta de probabilidades y clases en FASE 4. El período TEST se consultó repetidamente en la investigación posterior y ya no es un holdout completamente virgen.

LIVE congelado {live['analyzed_at_utc']}: {live['evaluated']:,} evaluaciones; accuracy {live['accuracy']:.2%}; balanced accuracy {live['balanced_accuracy']:.2%}; ROC AUC {live['discrimination']['roc_auc_up']:.4f}; PR AUC UP {live['discrimination']['pr_auc_up']:.4f}; Brier {live['brier_up']:.4f}; log loss {live['log_loss_up']:.4f}; macro F1 {live['macro_f1']:.4f}. Predicciones DOWN {live['predicted_classes']['DOWN']['share']:.2%}; DOWN real {live['observed_classes']['DOWN']['share']:.2%}. Recall UP {live['per_class']['UP']['recall']:.2%}; recall DOWN {live['per_class']['DOWN']['recall']:.2%}.

El corte LIVE tiene {live['source_gap_count']} huecos entre velas fuente evaluadas; el mayor es de {live['largest_source_gap_minutes']:.0f} minutos. Por tanto, no representa cobertura continua de todo el intervalo calendario.

## Uso previsto y límites

Señal probabilística experimental para apoyo analítico académico; no ejecuta órdenes, no es asesoría financiera y no garantiza rentabilidad. LIVE accuracy está cerca de always DOWN ({live['baselines']['train_majority_down']['accuracy']:.2%}); ROC AUC cerca de azar; fuerte sesgo hacia DOWN y probabilidades concentradas cerca de 0.5. El target H1 incluye muchos movimientos pequeños y empates como DOWN. FASE 3 encontró sólo diferencias numéricas menores por ventana LIVE de 120 velas, pero detectó shift del nivel absoluto de EMA. Las probabilidades no tienen validación robusta como frecuencia de acierto; 60% pasó en sólo {live['gate_60_final']['count']} evaluaciones del corte. No modela costes reales de ejecución, fills ni condiciones futuras de mercado.

## Gobernanza

Mantener `xgboost_v1` como referencia por ausencia de evidencia robusta para sustituirlo, no por rendimiento predictivo fuerte. Toda sustitución exige datos causales, comparación pareada, walk-forward, TEST/forward, baselines, varias métricas y disponibilidad operacional. LIVE sólo se usa para evaluación externa.
"""


def readiness(data: dict, playwright: dict, now: str, tests_passed: int | None = None,
              tests_skipped: int | None = None) -> dict:
    def row(status: str, evidence: str) -> dict:
        return {"status": status, "evidence": evidence}
    live = data["live_final"]
    return {"generated_at_utc": now, "status": "PARTIAL_FOR_ACADEMIC_PRESENTATION",
            "scope": "PMV analytics, not predictive-strength or trading readiness",
            "checks": {
                "live_ingestion": row("PASS", "Persisted BTCUSDT candles and evaluated predictions through " + live["source_candle_range_utc"][1]),
                "cron": row("PASS", "Prior observed Supabase Cron HTTP 200 at T0/T+1/T+2; independent LIVE records continue."),
                "prediction_persistence": row("PASS", f"{live['evaluated']} evaluated persisted xgboost_v1 predictions at frozen cutoff."),
                "posterior_evaluation": row("PASS", "Persisted live_evaluations joined by prediction ID; no LIVE fitting."),
                "dashboard": row("PASS", "Five public pages showed LIVE data in read-only Playwright review."),
                "prescriptive_on_demand": row("PARTIAL", "On-demand button and guarded HTTP flow observed; this phase intentionally did not call LLM or verify a successful live recommendation."),
                "secret_separation": row("PARTIAL", "Source enforces read-only Streamlit key and separate endpoint secrets; actual cloud secret inventories are not independently visible."),
                "model_versioning": row("PASS", "Product SHA unchanged and experimental artifacts separate."),
                "auditability": row("PASS", "pipeline_runs, live_evaluations, recommendation audit services and phase reports exist."),
                "tests": row("PASS" if tests_passed is not None else "PENDING",
                             f"Final offline suite: {tests_passed} passed, {tests_skipped} skipped."
                             if tests_passed is not None else "Run final offline suite after report generation."),
                "documentation": row("PASS", "Phase 1–8 reports, model card and deployment plan present; README publication update awaits review."),
                "limitations_visible": row("PARTIAL", "UI separates LIVE/TEST/simulation and says probabilities are not accuracy; class bias and baseline comparison remain in reports, not dashboard."),
                "no_automatic_trading": row("PASS", "Read-only dashboard and analytical recommendation paths; no order endpoint in deployment service."),
                "public_console": row("PARTIAL" if playwright.get("console_error_count") else "PASS",
                                      f"No page exception; {playwright.get('console_error_count', 0)} browser network console error(s) observed."),
            },
            "dashboard_pages": {
                "Inicio": {"purpose": "Estado y frescura del pipeline", "status": "PASS",
                           "issue": "No muestra limitaciones predictivas agregadas en el primer vistazo",
                           "proposal": "Enlazar model card y una nota breve de desempeño LIVE"},
                "Descriptivo": {"purpose": "Velas, OHLCV e indicadores persistidos", "status": "PARTIAL",
                                "issue": "La tarjeta de última vela recorta el timestamp a 1440 px",
                                "proposal": "Ajustar el texto de esa tarjeta tras aprobación"},
                "Predictivo": {"purpose": "Clase, probabilidades, timing, vigencia e historial", "status": "PASS",
                               "issue": "Baseline, sesgo de clases y período LIVE quedan en reportes, no en la página",
                               "proposal": "Añadir contexto estadístico breve sin llamar accuracy a P(UP)"},
                "Prescriptivo": {"purpose": "Apoyo analítico bajo demanda", "status": "PARTIAL",
                                 "issue": "Mensaje azul de bajo contraste; gate 60% carece de explicación de calidad",
                                 "proposal": "Corregir contraste y wording; conservar acción explícita y NO_ACTION/HOLD"},
                "Evaluación": {"purpose": "Separar LIVE, TEST histórico y simulación", "status": "PARTIAL",
                               "issue": "LIVE no muestra balanced accuracy, matriz, recall por clase ni período",
                               "proposal": "Añadir métricas agregadas precomputadas tras aprobación"},
            },
            "browser": {"url": playwright.get("url"), "checked_at_utc": playwright.get("checked_at_utc"),
                             "pages": [p["page"] for p in playwright.get("pages", [])],
                             "durations_seconds": {p["page"]: p["seconds_to_marker"] for p in playwright.get("pages", [])}},
            "cost_note": "Streamlit Community Cloud, Render Free and Supabase quota use were previously observed; ongoing no-cost use is not guaranteed and account quotas were not rechecked in this phase."}


def readiness_md(item: dict) -> str:
    lines = ["# PMV readiness — FASE 8", "", f"Estado: **{item['status']}**. Ámbito: {item['scope']}.", "",
             "| Comprobación | Estado | Evidencia/límite |", "| --- | --- | --- |"]
    for key, value in item["checks"].items():
        lines.append(f"| {key} | {value['status']} | {value['evidence']} |")
    lines += ["", "## Auditoría del dashboard", "",
              "| Página | Propósito | Estado | Problema observado | Cambio propuesto |",
              "| --- | --- | --- | --- | --- |"]
    for name, value in item["dashboard_pages"].items():
        lines.append(f"| {name} | {value['purpose']} | {value['status']} | {value['issue']} | {value['proposal']} |")
    lines += ["", "## Rendimiento y coste", "",
              "Navegación medida con Playwright en una sesión pública; duración desde clic hasta marcador y banner LIVE, no una métrica global de disponibilidad. Una muestra puntual no prueba latencia constante.", "",
              item["cost_note"], "", "## Decisión", "",
              "La arquitectura demuestra un PMV de Business Analytics end-to-end. La calidad predictiva no queda validada como ventaja comercial ni como señal de trading autónomo.", ""]
    return "\n".join(lines)


def deployment_plan(data: dict) -> str:
    live = data["live_final"]
    return f"""# Plan de cambios posterior a FASE 8 — pendiente de aprobación

No se aplicó ningún cambio a producción en esta fase. Modelo, target, horizonte, threshold de clase y gate prescriptivo permanecen iguales.

## MUST_FIX_BEFORE_PRESENTATION

1. Aclarar explícitamente en la defensa y en Predictivo/Prescriptivo que `max(P) ≥ 60%` es un guardrail operacional **no validado como calidad predictiva**. Sólo {live['gate_60_final']['count']}/{live['evaluated']} evaluaciones LIVE pasaron en el corte; no afirmar “señal fuerte” por esa regla. Preparar y revisar el wording antes de desplegarlo.
2. Hacer visible junto a la explicación del modelo el sesgo DOWN, el baseline sobre la misma cohorte y el período evaluado. La página Evaluación actualmente muestra accuracy y conteos, pero no balanced accuracy, matriz de confusión ni recall por clase. Evitar que 52% se interprete como ventaja demostrada.
3. Corregir el bajo contraste del mensaje informativo azul de Prescriptivo observado en Playwright; verificar visualmente en desktop y móvil.
4. Verificar con el propietario el inventario real de secretos en Streamlit y Render antes de la presentación. El código impide una service key en Streamlit HTTP, pero esta revisión no puede ver los paneles privados.

## NICE_TO_HAVE

1. Añadir métricas agregadas precomputadas a Evaluación, sin consultas pesadas en cada visita: balanced accuracy, matriz, recall UP/DOWN y período.
2. Evitar el recorte del timestamp en la tarjeta «Última vela cerrada» de Descriptivo; medir latencia de navegación en varios momentos y revisar el error de red de consola observado antes de atribuir causa o hacer refactor de caché.
3. Publicar README actualizado con arquitectura REAL, límites predictivos y model card después de revisión.

Contenido propuesto para README: propósito académico y alcance sin órdenes; arquitectura LOCAL/CLOUD y rutas protegidas; modelo `xgboost_v1` H1 y ocho features; separación LIVE/TEST/simulación; métricas LIVE y baselines con fecha de corte; límites de calibración/sesgo/gate; ejecución local y pruebas; variables por servicio sin valores; cuotas y riesgos operacionales; enlaces a model card y reportes de FASE 1–8.

## FUTURE_RESEARCH (máximo cinco líneas)

1. Captura prospectiva L1 con corte causal (spread, OBI, microprice).
2. Noticias con `first_seen_at` real y disponibilidad causal.
3. Evaluar EMA relativa y otras representaciones estacionarias con un protocolo nuevo.
4. Ampliar histórico y regímenes con holdout verdaderamente nuevo.
5. Nuevos targets/horizontes sólo con hipótesis fijada antes de TEST y validación temporal.

## Secuencia de publicación

Revisión del informe → aprobación explícita del wording y métricas → cambios UI/docs aislados → tests y revisión visual → despliegue controlado → comprobación LIVE. No entrenar ni promover candidato experimental por esta fase. No modificar gate hasta una decisión de producto separada.
"""


def research_summary(data: dict, report: dict) -> str:
    live = data["live_final"]
    return f"""# Síntesis académica final — Alpha Engine

## Resultado predictivo

Ningún candidato experimental superó de forma robusta a `xgboost_v1`. El modelo sigue como referencia del PMV por falta de evidencia de reemplazo, no por ser un predictor fuerte. En {live['evaluated']} evaluaciones LIVE congeladas obtuvo accuracy {live['accuracy']:.2%}, balanced accuracy {live['balanced_accuracy']:.2%} y ROC AUC {live['discrimination']['roc_auc_up']:.4f}; always DOWN obtuvo {live['baselines']['train_majority_down']['accuracy']:.2%}. Predijo DOWN {live['predicted_classes']['DOWN']['share']:.2%} de las veces.

TFI no mostró mejora estable; HOLD identifica conceptualmente ruido de movimientos pequeños pero no mejoró calidad direccional; H3 careció de apoyo; H5 redujo movimientos diminutos y amplió la ventana operacional, sin superioridad general; NEWS_SENTIMENT fue técnicamente viable, pero su score no mejoró robustamente la predicción y su disponibilidad histórica es aproximada. EMA relativa es investigación futura.

Alpha Vantage podría estudiarse en el futuro como **contexto informativo** para explicaciones prescriptivas, separado de la inferencia XGBoost. Una noticia no debe presentarse como causa comprobada de una recomendación. No se implementó ni desplegó esa integración.

## Business Analytics

**Descriptivo:** velas, volumen e indicadores persistidos. **Predictivo:** predicción XGBoost versionada, probabilidad, timing y vigencia. **Prescriptivo:** solicitud explícita con guardrails, endpoint protegido, LLM y auditoría; puede producir NO_ACTION/HOLD y no ejecuta órdenes. **Evaluación:** LIVE posterior, TEST histórico y simulación económica separados.

## CRISP-DM

1. Business Understanding: PMV académico de apoyo a decisiones de scalping; sin órdenes.
2. Data Understanding: Binance BTCUSDT 1m, calidad temporal, microestructura y cobertura de noticias auditadas.
3. Data Preparation: OHLCV cerrado, ocho features causales, splits cronológicos y purga.
4. Modeling: xgboost_v1 y ablations experimentales TFI, HOLD, horizontes y NEWS_SENTIMENT.
5. Evaluation: TEST, walk-forward, LIVE persistido, baselines, calibración, incertidumbre y reportes de FASE 1–8.
6. Deployment: Supabase Cron → Render `/internal/run-cycle` → LiveCycle → Binance/features/xgboost_v1 → Supabase → Streamlit; Prescriptivo: Streamlit → endpoint protegido → RecommendationUseCase → LLM → Supabase.

Alpha Vantage, L1, TFI, HOLD y H5 permanecen fuera del pipeline productivo. Un resultado experimental negativo sigue siendo evidencia válida: delimita qué cambios **no** están justificados. La arquitectura funcional y la calidad predictiva son conclusiones distintas.

## Mensaje verificable para el profesor

“Alpha Engine integra ingesta LIVE, ETL causal, inferencia, persistencia, evaluación posterior, observabilidad y apoyo prescriptivo bajo demanda. Su modelo actual tiene desempeño LIVE cercano a referencias triviales; no afirmamos ventaja predictiva ni rentabilidad. Las fases experimentales documentan qué hipótesis no han demostrado mejora robusta.”
"""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tests-passed", type=int)
    parser.add_argument("--tests-skipped", type=int)
    args = parser.parse_args()
    data = load_inputs()
    model_sha_before = sha256(MODEL)
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    now = datetime.now(timezone.utc).isoformat()
    report = build_comparison(data, model_sha_before, commit, now)
    metadata = json.loads((ROOT / "models/xgboost_v1_metadata.json").read_text(encoding="utf-8"))
    browser_path = ROOT / "runtime/phase8_playwright.json"
    browser = json.loads(browser_path.read_text(encoding="utf-8")) if browser_path.exists() else {}
    readiness_report = readiness(data, browser, now, args.tests_passed, args.tests_skipped)
    write_json(REPORTS / "final_model_comparison_v1.json", report)
    (REPORTS / "final_model_comparison_v1.md").write_text(comparison_md(report, data), encoding="utf-8")
    (REPORTS / "model_card_xgboost_v1_final.md").write_text(model_card(data, report, metadata), encoding="utf-8")
    write_json(REPORTS / "pmv_readiness_v1.json", readiness_report)
    (REPORTS / "pmv_readiness_v1.md").write_text(readiness_md(readiness_report), encoding="utf-8")
    (REPORTS / "final_deployment_plan_v1.md").write_text(deployment_plan(data), encoding="utf-8")
    (REPORTS / "final_research_summary_v1.md").write_text(research_summary(data, report), encoding="utf-8")
    if sha256(MODEL) != model_sha_before:
        raise RuntimeError("Product model changed while writing reports")
    print(f"FASE 8 reports built; LIVE n={data['live_final']['evaluated']}; product SHA unchanged {model_sha_before}")


if __name__ == "__main__":
    main()
