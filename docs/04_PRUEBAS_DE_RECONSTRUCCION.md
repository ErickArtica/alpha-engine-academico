# 04 · Pruebas reproducibles y criterios de aceptación

Ejecute primero [01](01_INSTALACION_LOCAL.md) en un clon limpio y un proyecto Docker nuevo. Estos comandos no apuntan al entorno del autor. Las pruebas locales crean sus propios datos PostgreSQL; las pruebas cloud requieren cuentas propias según [02](02_DESPLIEGUE_NUBE.md). No se afirma que una instalación cloud nueva haya sido provisionada mediante las pruebas locales.

## A. Instalación independiente y automatización real local

```text
git rev-parse HEAD
python scripts/configure_local.py --project alpha-profesor --api-port 18000 --frontend-port 18501
docker compose --project-name alpha-profesor --env-file .env config --quiet
docker compose --project-name alpha-profesor --env-file .env build
docker compose --project-name alpha-profesor --env-file .env up -d
python scripts/verify_local.py --observe-scheduler
```

Si ya creó `.env`, omita el generador; no sobrescribe secretos. Resultado esperado: `status: PASS`, salud API y Streamlit 200, modelo `MODEL_V1_EIGHT_FEATURES_GATE_060_PASS`, disparador anónimo 401 y crecimiento entre `initial` y `after_scheduler`. La prueba necesita Binance disponible y una vela nueva cerrada; puede tardar varios minutos. Sin filas no es PASS, aunque los contenedores estén running.

Para inspección manual:

```text
docker compose --project-name alpha-profesor --env-file .env ps --all
docker compose --project-name alpha-profesor --env-file .env exec -T postgres psql -U alpha_engine -d alpha_engine -c "SELECT count(*) FROM candles; SELECT count(*),max(generated_at) FROM predictions; SELECT count(*) FROM live_evaluations; SELECT status,new_prediction FROM pipeline_runs ORDER BY started_at DESC LIMIT 3;"
docker compose --project-name alpha-profesor --env-file .env logs --tail 25 scheduler bootstrap api
```

El scheduler debe registrar HTTP 200 y al menos un ciclo nuevo. Bootstrap registra replay histórico local; no llamarlo precisión prospectiva. El verificador comprueba el checksum del artefacto, las ocho features y gate 0.60. No entrena ni recalibra.

## B. Pruebas offline de IA, sentimiento y seguridad

En una venv propia, desde el clon público:

```text
python -m venv .venv
```

PowerShell: `.venv\Scripts\Activate.ps1`; bash: `source .venv/bin/activate`. Después:

```text
python -m pip install -r requirements.txt
python -m pytest tests/test_marketaux_diagnostics.py tests/test_academic_demo.py tests/test_academic_provider_reliability.py tests/test_marketaux_disabled_release.py tests/test_prescriptive_news_context.py -q
python -m unittest discover -s deploy/northflank -p "test_*.py"
```

Esperado: suites aprobadas, sin claves externas. Los tests utilizan httpx MockTransport y estado temporal; no llaman proveedores ni producción. No ejecute indiscriminadamente tests marcados `integration` contra un servicio ajeno. Las dependencias del modelo están fijadas en `deploy/render/requirements.txt`; no sustituirlas por versiones incompatibles para ocultar un error.

Cobertura:

- Muse: BUY/SELL/HOLD/WAIT, hash/acción/riesgo, JSON inválido, contradicciones, ganancias garantizadas, timeout, 202/429/5xx, SSE incompleto y descarte del razonamiento parcial.
- Seguridad: login válido/inválido, sesiones, revocación/expiración, acceso anónimo, cuotas globales/por usuario, caché, concurrencia y fallos contabilizados. Estas son pruebas offline, no evidencia de un nuevo login cloud.
- Marketaux: BTCUSD/CC/cryptocurrency/Bitcoin USD con `match_score=112.536514` y `sentiment_score=0.13115`. Relevancia finita, no negativa y **sin límite de 100**; mínimo configurable preservado. Sentimiento sólo en [-1,1]; ausente/inválido no aporta score.
- Diagnósticos: códigos/contadores para estructura, fecha, título, URL, símbolo/entidad, relevancia, sentimiento y duplicados; no incluye textos/URLs/tokens.
- Temporalidad: artículo válido de dos horas se admite como dato de hasta 24 h, pero no entra al agregado reciente de una hora. Sin datos → N/A; nunca neutral inventado.

El defecto del techo de 100 se reproduce con ese fixture y su corrección se prueba. **No demuestra que fuera la causa de la noticia real descartada anteriormente**: el payload de aquella solicitud no fue conservado. No reconstruir retrospectivamente datos que no existen.

## C. Navegador: seis pantallas y simulación

Abra `http://127.0.0.1:18501/` y recorra:

| Pantalla | Resultado esperado |
|---|---|
| Inicio / Alpha Engine | Fuente PostgreSQL local, últimos timestamps y estado del pipeline |
| Descriptivo | OHLC/volumen e indicadores; sin excepción Streamlit |
| Predictivo | Predicciones locales de V1; probabilidades y timestamps |
| Prescriptivo | Política/gate conservados; proveedor N/A si no hay clave propia; ninguna llamada al abrir |
| Evaluación | Resultados del replay/nuevos registros distinguidos de TEST histórico |
| Simulación | Seleccionar **Replay Histórico**, avanzar al menos un paso y comprobar gráficos; no pulsar creación de cartera si sólo evalúa memoria |

En DevTools revise errores 500 y WebSocket de Streamlit `/_stcore/stream`; mantenga una página abierta al menos 35 s. Compruebe el healthcheck raíz. Un intento transitorio a `/<pagina>/_stcore/health` con 404 seguido de conexión correcta a raíz no justifica por sí solo cambiar proxy.

Para un smoke test Python del dashboard local, dentro de su contenedor y sin pulsar acciones externas:

```text
docker compose --project-name alpha-profesor --env-file .env exec -T streamlit python -c "from pathlib import Path; from streamlit.testing.v1 import AppTest; pages=[Path('/app/streamlit_app/Home.py'),*sorted(Path('/app/streamlit_app/pages').glob('[0-9]_*.py'))]; results=[AppTest.from_file(str(p),default_timeout=30).run() for p in pages]; assert all(not r.exception and not r.error for r in results); print('SIX_LOCAL_PAGES_PASS')"
```

## D. Backend DEMO y frontend cloud propios

Después de desplegar su infraestructura, sustituya los dominios ficticios y consulte:

```text
curl -i https://SU_PIPELINE.onrender.com/health
curl -i https://SU_DEMO.onrender.com/health
curl -i https://SU_FRONTEND.onrender.com/_stcore/health
curl -i -X POST https://SU_DEMO.onrender.com/api/academic/explain -H "Content-Type: application/json" -d "{}"
```

Esperado: 200, 200, 200 y 401, sin generación ni escritura de negocio. En Windows PowerShell clásico utilice `curl.exe` si `curl` es un alias. Una petición anónima a refresh-news también debe dar 401. D no debe exponer `/internal/run-cycle`, `/api/paper` ni recommendation: compruebe 404 en su servicio D, sin bearer. No llame el pipeline existente del autor.

En su frontend: ingresar con cuenta propia, consultar contexto, salir y comprobar rechazo posterior; verificar vigencia de 15 min y expiración global sin ampliarla para superar un test. Observar las cuotas antes/después y después de un reinicio de **su DEMO**. No gastar proveedores para comprobar cuota: las pruebas de agotamiento se realizan con mocks.

## E. Prueba NVIDIA opcional, una sola operación explícita

Requiere su backend D, usuario propio, clave/modelo disponibles, créditos gratuitos comprobados, autorización para gastar y flag LLM habilitado bajo cuotas. Tome un ID real de SU base, por ejemplo una predicción histórica cuya probabilidad máxima sea menor a 0.60. No copie un ID del autor. La decisión debe validarse antes de llamar al modelo.

```sql
SELECT prediction_id FROM public.predictions
WHERE model_version='xgboost_v1' AND greatest(prob_up,prob_down)<0.60
  AND generated_at<valid_until ORDER BY generated_at DESC LIMIT 1;
```

Primero consulta sin proveedor, reemplazando los placeholders:

```text
python scripts/probe_academic_provider.py --origin https://SU_DEMO.onrender.com --user profesor --prediction-id ID_REAL_DE_64_HEXADECIMALES --action context --expected-action WAIT --confirm-own-service
```

Sólo si los precontroles están aprobados, una explicación:

```text
python scripts/probe_academic_provider.py --origin https://SU_DEMO.onrender.com --user profesor --prediction-id ID_REAL_DE_64_HEXADECIMALES --action explain --expected-action WAIT --confirm-own-service --allow-one-provider-request --confirm-free-plan
```

La contraseña se introduce oculta. El script rechaza las URLs de presentación del autor, no cambia flags y no reintenta. Esperado: 200, EXPLAINED, modelo Muse y envolvente determinista preservada. Registra sólo métricas saneadas; consumo exacto sólo si lo informa el proveedor. Un resultado cached no prueba una generación nueva. Timeout/429/error consumen cuota si se reservó el intento; no repita automáticamente. Para observar el texto use el frontend DEMO de su instalación. No genere otra explicación únicamente para hacer una captura si ya consumió el máximo autorizado.

## F. Marketaux opcional: cuenta y derechos antes de activar

Sin derechos aprobados mantenga NEWS y RIGHTS en false; la prueba académica reproducible es la suite de mocks y el estado N/A. Si dispone de autorización de uso/exposición y de cuenta Free con saldo suficiente, configure el token sólo en D y revise explícitamente ambos flags antes de una prueba:

```text
python scripts/probe_academic_provider.py --origin https://SU_DEMO.onrender.com --user profesor --action refresh-news --confirm-own-service --allow-one-provider-request --confirm-free-plan
```

Esperado: 403 si faltan permisos de función/derechos; 429 si cuota/concurrencia lo impide; 200 REFRESHED o CACHED si procede. **No cambie permisos para forzar un 200.** Si hay HTTP 200 pero cero artículos elegibles, inspeccione `provider_diagnostics.reason_counts` y contadores de edad. Si sólo hay noticias de una a 24 horas, el agregado no es LIVE. No repita durante el cooldown de 30 minutos ni aumente la cuota. No publique artículos, snippets, scores o derivados con licencia incierta.

## G. Evidencia y limpieza

Registre SHA, sistema/fecha, proyecto Docker propio, salud, cantidades antes/después, pruebas aprobadas y limitaciones. No adjunte `.env`, usuarios/hashes reales, bases de datos ni logs con credenciales. El build usa ignores específicos para excluirlos del contexto.

Después de verificar, elimine sólo el entorno de práctica:

```text
docker compose --project-name alpha-profesor --env-file .env down --volumes --remove-orphans
```

No declare reconstrucción completa con un build solamente: debe haber base vacía inicial, bootstrap con datos propios, inferencia V1, evaluación, scheduler, seis páginas y simulación comprobadas. La reconstrucción cloud/proveedores reales se declara por separado de este PASS local.
