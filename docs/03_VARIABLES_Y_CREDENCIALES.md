# 03 · Variables y credenciales

**L**: Compose local; **P**: pipeline cloud; **D**: backend DEMO; **F**: lector público. Todos los valores de ejemplo son ficticios o configuración no secreta. Los defaults numéricos de investigación se conservan, no se recalibran para reconstruir el sistema.

Una misma KEY puede tener valores diferentes por servicio. No copie todo el inventario al frontend. Para L use `.env.local.example` y `scripts/configure_local.py`; para F, `.env.example`/`deploy/northflank/runtime.env.example`; para P/D configure el panel privado. La contraseña PostgreSQL no es una API key, un token administrativo ni una contraseña DEMO.

## Almacenamiento, API y proceso

| Nombre | Servicio | Obligatoria/opcional | Dónde obtenerla | Cómo configurarla | Restricciones de seguridad |
|---|---|---|---|---|---|
| `APP_ENV` | L/P/D | Obligatoria | Perfil elegido | local en Compose; production cloud | No habilita por sí sola permisos |
| `STORAGE_PROVIDER` | L/P/D/F | Obligatoria | Arquitectura | postgresql local; supabase cloud | No apuntar pruebas al almacenamiento original |
| `DATABASE_URL` | L | Obligatoria, Compose la genera | PostgreSQL propio | Derivada de POSTGRES_PASSWORD dentro del contenedor | Privada; no copiar al lector ni logs |
| `POSTGRES_PASSWORD` | L/postgres | Obligatoria | configure_local.py | `.env` ignorado, secreto nuevo hex | No usar la contraseña Supabase |
| `SUPABASE_URL` | P/D/F | Obligatoria cloud | Dashboard propio | URL HTTPS de su proyecto | No usar proyecto del autor |
| `SUPABASE_KEY` | P/D | Obligatoria cloud | API key secret/service_role propia | Environment privado servidor | Privilegiada, puede omitir RLS; nunca navegador/imagen |
| `SUPABASE_KEY` | F | Obligatoria | API key publishable/anon propia | Environment F | Sólo lecturas aprobadas con RLS/GRANT |
| `INTERNAL_CRON_SECRET` | L/P/D | Obligatoria en API | Generación aleatoria nueva | Environment; en P también Vault | Al menos 32 caracteres; distinto por servicio |
| `RECOMMENDATION_API_SECRET` | L/P/D | Obligatoria en API | Otro secreto nuevo | Environment; Compose lo propaga | Distinto de Cron; D lo exige para readiness, sin abrir rutas legacy |
| `COMPOSE_PROJECT_NAME` | L/operador | Obligatoria para aislar | Nombre nuevo propio | Helper y `--project-name` | No reutilizar el de otra instalación para limpiar |
| `LOCAL_API_PORT` | L/host | Opcional; guía 18000 | Puerto libre | Helper/`.env` | Sólo loopback; no detener puertos ajenos |
| `LOCAL_STREAMLIT_PORT` | L/host | Opcional; guía 18501 | Puerto libre | Helper/`.env` | Distinto de API y sólo loopback |
| `LOCAL_API_URL` | L/scheduler | Compose la fija | Red local Compose | http://api:8000 | Nunca backend productivo del autor |
| `PORT` | Render | Plataforma / F=8501 | Render/imagen | Start nativo usa `$PORT`; F escucha 8501 | No es el puerto host de Compose |
| `PYTHON_VERSION` | Render Native | Fijar versión | Versión soportada | 3.12.14 en panel | No cambia el Python de una imagen Docker |
| `SYMBOL` | L/P/D/F | Opcional; BTCUSDT | Contrato de mercado | Environment/default | Noticias usan BTCUSD, no BTCUSDT |
| `INTERVAL` | L/P/D/F | Opcional; 1m | Contrato V1 | Environment/default | Conservar 1m |
| `BINANCE_BASE_URL` | L/P | Opcional | Documentación oficial Binance | Environment; Compose lo propaga | Sólo datos públicos; ninguna clave de trading |
| `MODEL_PATH` | L/P/D | Default obligatorio válido | Artefacto incluido | models/xgboost_v1.joblib | Verificar checksum, no sustituir |
| `MODEL_CONFIDENCE_THRESHOLD` | L/P/D/F | Fija | Política | 0.60 | No es accuracy ni se modifica para demo |
| `PREDICTION_LOOKBACK_CANDLES` | L/P/D | Opcional | Settings | Default 120 | Mantener causalidad y compatibilidad V1 |
| `MAX_CANDLE_AGE_SECONDS` | L/P/D | Opcional | Settings | Default 120 | Guarda de vigencia; no eludir para etiquetar LIVE |
| `BACKFILL_DAYS` | Herramientas locales | Opcional | Investigación propia | Default 30 | No inicia automáticamente un backfill cloud |
| `V3_SHADOW_ENABLED` | L/P/F | Opcional, false | Investigación separada | Mantener false | No forma parte de V1 validado |

## Frontend público y UI local

| Nombre | Servicio | Obligatoria/opcional | Dónde obtenerla | Cómo configurarla | Restricciones de seguridad |
|---|---|---|---|---|---|
| `FRONTEND_PUBLIC_HOST` | F | Obligatoria si no hay NF_HOSTS | Dominio real propio | Host sin protocolo/puerto/ruta | Mantener CORS/XSRF |
| `NF_HOSTS` | F | Alternativa opcional | Dominios propios | Lista separada por comas | Nombre histórico; funciona en Render |
| `NORTHFLANK_PUBLIC_READ_ONLY` | F | Obligatoria true | Política de publicación | Environment true | No desactivar para habilitar DEMO |
| `NORTHFLANK_PUBLIC_DATA_APPROVED` | F | False inicial; true para arrancar | Revisión RLS/GRANT/datos | Panel sólo tras aprobación | False bloquea arranque, de forma intencional |
| `ACADEMIC_DEMO_ENABLED` | D/F | Necesaria con DEMO | Autorización propia | True tras preparar backend | En F sólo añade panel; D autoriza cada acción |
| `ACADEMIC_DEMO_API_URL` | F | Necesaria con DEMO | URL REAL de su D | HTTPS sin ruta final | Nunca servicios del autor en pruebas propias |
| `LOCAL_DEMO_MODE` | L | Opcional, true en ejemplo | Perfil local | `.env`/Compose | False obligatorio en F; no equivale a autenticación cloud |
| `PRESCRIPTIVE_BACKEND` | L | Necesaria para UI HTTP | Arquitectura | http en Compose | El lector F no tiene estas rutas |
| `RECOMMENDATION_API_URL` | L | Necesaria para UI HTTP | API propia | Compose fija URL interna de recommendation | Prohibida en F |
| `RECOMMENDATION_API_KEY` | L | Necesaria para UI HTTP | Su RECOMMENDATION_API_SECRET | Compose, sólo servidor Streamlit | Nunca al navegador ni F |
| `RECOMMENDATION_API_AUTH_SCHEME` | L | Necesaria con HTTP | Contrato API | bearer en Compose | No usar secreto Cron |
| `RECOMMENDATION_API_TIMEOUT_SECONDS` | L | Opcional | Settings | Default 50 | Camino legado, distinto de 115 s del cliente académico |
| `RECOMMENDATION_MAX_ATTEMPTS_PER_MINUTE` | L/P | Opcional | Settings | Default 3 | No confundir con cuota diaria de NVIDIA DEMO |
| `DASHBOARD_FALLBACK_ENABLED` | L/F | Opcional, false | Decisión explícita | Mantener false en prueba LIVE | Un snapshot no demuestra datos actuales |
| `DASHBOARD_FALLBACK_SNAPSHOT` | L | Sólo fallback explícito | Snapshot propio permitido | Ruta local propia | No se incluye historial privado del autor |
| `ALPHA_UI_TIMINGS` | L | Opcional | Operador | Activar sólo para diagnóstico | No necesita claves externas |

## Autenticación DEMO y proveedores

| Nombre | Servicio | Obligatoria/opcional | Dónde obtenerla | Cómo configurarla | Restricciones de seguridad |
|---|---|---|---|---|---|
| `ACADEMIC_DEMO_EXPIRES_AT` | D | Obligatoria con DEMO | Fecha UTC propia | ISO-8601, zona horaria, a no más de 48 h | No extender otro entorno; sesiones de 15 min |
| `ACADEMIC_DEMO_USERS_JSON` | D | Obligatoria con DEMO | configure_academic_users.py | JSON de hashes mediante panel privado | No publicar contraseñas ni hashes de cuentas reales |
| `ACADEMIC_DEMO_SESSION_SECRET` | D | Obligatoria con DEMO | Nuevo aleatorio >=48 caracteres | Environment privado | Distinto de claves/API/otros secretos |
| `ACADEMIC_DEMO_LLM_ENABLED` | D | Opcional, false inicial | Autorización y saldo | True sólo tras comprobación | 10 intentos globales/día UTC; 3/usuario; fallos cuentan |
| `ACADEMIC_DEMO_NEWS_ENABLED` | D | Opcional, false inicial | Cuenta/token y autorización | Mantener false hasta revisión separada | 48/día; 30 min entre intentos; sin polling por visitante |
| `ACADEMIC_DEMO_NEWS_RIGHTS_APPROVED` | D | Opcional, false inicial | Permiso aplicable del proveedor | True sólo con evidencia de derechos | Free no equivale a redistribución autorizada |
| `LLM_PROVIDER` | L/D | Necesaria con LLM | Proveedor propio | nvidia para Muse | No activa permiso de consumo por sí sola |
| `NVIDIA_API_KEY` | L/D | Opcional; necesaria para NVIDIA | Cuenta NVIDIA propia | Environment privado del backend | Nunca F, Git, Dockerfile o build-arg |
| `NVIDIA_MODEL` | L/D | Necesaria con NVIDIA | Catálogo oficial | meta/muse-glimmer-30b para DEMO | No reemplazar silenciosamente el modelo histórico |
| `NVIDIA_BASE_URL` | L/D | Opcional con default | NVIDIA | https://integrate.api.nvidia.com/v1 | DEMO exige endpoint canónico |
| `NVIDIA_REASONING_EFFORT` | L/D | Opcional | Modelo compatible | low | No altera reglas |
| `NVIDIA_TIMEOUT_SECONDS` | L/P | Opcional | Settings | Default 90 | No es el deadline académico |
| `LLM_LIVE_TIMEOUT_SECONDS` | L/P | Opcional | Settings | Default 45 | Camino legado; no aumentar para ocultar caducidad |
| `LLM_DEADLINE_MARGIN_SECONDS` | L/P | Opcional | Settings | Default 1 | Reserva margen para respuesta |
| `LLM_TEMPERATURE` | L/P | Opcional | Settings | Default 0.1 | Transporte académico Muse usa 0.95; no modifica V1 |
| `ACADEMIC_NVIDIA_CONNECT_SECONDS` | D | Opcional | Política acotada | Default 5; rango 1..10 | Sin reintentos |
| `ACADEMIC_NVIDIA_READ_SECONDS` | D | Opcional | Política acotada | Default 60; rango 30..90 | No mayor que total |
| `ACADEMIC_NVIDIA_TOTAL_SECONDS` | D | Opcional | Política acotada | Default 85; rango 60..90 | Deadline cancelable |
| `ACADEMIC_NVIDIA_MAX_TOKENS` | D | Opcional | Presupuesto del modelo | Default 2048; rango 1024..4096 | Revisar créditos antes de generar |
| `PRESCRIPTIVE_NEWS_ENABLED` | L/P | Opcional, false inicial | Autorización propia | Environment; Compose lo propaga | F exige false; D usa sus flags separados |
| `PRESCRIPTIVE_NEWS_PROVIDER` | L/P | Opcional | Proveedor legado | alpha_vantage o marketaux | Revisar dependencias antes de cambiar selector |
| `ALPHA_VANTAGE_API_KEY` | L/P | Opcional | Cuenta Alpha Vantage | Variable privada; Compose la propaga | No incluir en lector público ni activar por visitante |
| `ALPHA_VANTAGE_TIMEOUT_SECONDS` | L/P | Opcional | Settings | Default 6 | Sin consumo en reconstrucción básica |
| `MARKETAUX_API_TOKEN` | D o L/P legado | Opcional; necesaria con noticias | Cuenta Marketaux propia | **Variable de proceso** backend | Settings NO la lee desde .env; Compose no la propaga por defecto |
| `MARKETAUX_TIMEOUT_SECONDS` | L/P legado | Opcional | Settings | Default 6 | No es un permiso de consumo |
| `MARKETAUX_DAILY_BUDGET` | L/P legado | Opcional | Settings | Default legado 50 | DEMO usa 48; no ampliar límites |

## Parámetros opcionales conservados

Todos los nombres siguientes están en `Settings`. No son necesarios para arrancar con los defaults. Configurarlos cambia el experimento o activa infraestructura adicional y requiere revisión propia.

| Nombre | Servicio | Obligatoria/opcional | Dónde obtenerla | Cómo configurarla | Restricciones de seguridad |
|---|---|---|---|---|---|
| `PAPER_INITIAL_CAPITAL` | L/cartera | Opcional | Settings | Default 1000 | Capital virtual, no habilita trading |
| `PAPER_ALLOCATION` | L/cartera | Opcional | Settings | Default 0.20 | No habilita cartera cloud multiusuario |
| `PAPER_FEE_RATE` | L/cartera | Opcional | Settings | Default 0.001 | Parámetro académico |
| `PAPER_SLIPPAGE_RATE` | L/cartera | Opcional | Settings | Default 0.0002 | Parámetro académico |
| `RISK_VOLATILITY_MEDIUM`, `RISK_VOLATILITY_HIGH` | L/P/D | Opcionales | Settings | Conservar defaults 0.002 / 0.005 | No recalibrar para forzar acciones |
| `RISK_RSI_MEDIUM_LOW`, `RISK_RSI_MEDIUM_HIGH` | L/P/D | Opcionales | Settings | Conservar 35 / 65 | Validación de orden |
| `RISK_RSI_HIGH_LOW`, `RISK_RSI_HIGH_HIGH` | L/P/D | Opcionales | Settings | Conservar 25 / 75 | Validación de orden |
| `RISK_MIN_STOP_PCT`, `RISK_STOP_VOL_MULTIPLIER` | L/P/D | Opcionales | Settings | Conservar 0.002 / 2.0 | Riesgo determinista |
| `RISK_STOP_CAP_PCT`, `RISK_REWARD_RATIO` | L/P/D | Opcionales | Settings | Conservar 0.10 / 1.5 | No modificar durante reconstrucción |
| `OLLAMA_BASE_URL`, `OLLAMA_MODEL` | L/alternativo | Opcionales | Servicio Ollama propio | Configuración propia | No es fallback silencioso ni parte de Muse validado |
| `OLLAMA_NUM_CTX`, `OLLAMA_TIMEOUT_SECONDS` | L/alternativo | Opcionales | Settings | 4096 / 180 por defecto | Requiere modelo y capacidad local propios |
| `INFLUX_URL`, `INFLUX_ORG`, `INFLUX_BUCKET` | Legado | No para estas guías | Influx propio | No configurar en Compose normal | PostgreSQL no depende de Influx |
| `INFLUX_TOKEN`, `INFLUX_USERNAME`, `INFLUX_PASSWORD` | Legado | No para estas guías | Influx propio | Sólo servidor legado | Credenciales privadas |
| `INFLUX3_URL`, `INFLUX3_DATABASE`, `INFLUX3_TOKEN` | Legado | No para estas guías | Influx3 propio | Sólo servidor legado | No copiar token a F |
| `ALPHA_MEMORY_DIAGNOSTICS` | P/D | Opcional | Operador | true para diagnóstico Linux | RSS/peak saneados; no es métrica de todas las instancias |
| `MUSE_DIAGNOSTIC_ENABLED`, `MUSE_DIAGNOSTIC_TOKEN` | P/diagnóstico legado | No habilitar aquí | Procedimiento separado | Ausente/false; token independiente si se autoriza | No es el endpoint académico ni permiso de gastar |
| `MUSE_DIAGNOSTIC_STARTS_AT`, `MUSE_DIAGNOSTIC_EXPIRES_AT` | P/diagnóstico legado | Condicionales | Ventana propia revisada | ISO-8601 | No se usan en estas pruebas |
| `RENDER`, `RENDER_GIT_COMMIT`, `RENDER_SERVICE_ID`, `RENDER_INSTANCE_ID` | Plataforma | Automáticas | Render | No inventar/copiar IDs | Metadatos, no autenticación |
| `MUSE_DIAGNOSTIC_CHILD`, `PYTHONNOUSERSITE` | Proceso interno | Automáticas | Wrapper | No configurar manualmente | No habilitan DEMO |

## Credenciales de plataforma, no variables del runtime

| Credencial | Servicio | Obligatoria/opcional | Dónde obtenerla | Cómo configurarla | Restricciones de seguridad |
|---|---|---|---|---|---|
| Contraseña PostgreSQL cloud | Supabase propio | Para tareas administrativas | Al crear su proyecto | Gestor privado; conexión TLS | No confundir con key REST ni copiar a F |
| Token Supabase CLI/Management | Operador | Opcional | Su cuenta Supabase | Entrada privada de herramienta | No necesario para Compose/runtimes; no imagen |
| PAT classic GHCR publicación | Operador | Si publica imágenes | GitHub propio | Login interactivo, mínimo write:packages | No comando con token literal ni Git |
| PAT GHCR pull | Render/registro | Si imagen privada | GitHub propio | Credencial de registro read:packages | Separado del de publicación, no env de Streamlit |
| Contraseña DEMO individual | Usuario | Para acciones protegidas | configure_academic_users.py | Canal privado y hash en D | Nunca texto plano en Git ni hashes públicos |
| Secretos Vault | Supabase Cron | Para automatización cloud | Su URL/secreto pipeline | Dashboard Vault; nombres exactos en guía 02 | No volcar decrypted_secrets |

`PYTHON_IMAGE` es un argumento de build del Dockerfile público, no un secreto. `PYTHONPATH`, `HOME`, `PYTHONDONTWRITEBYTECODE` y otras variables de sistema vienen de las imágenes; no copie valores del equipo del autor.

Compruebe configuración sin revelar valores: `docker compose ... config --quiet`, healthchecks y booleanos de presencia. No publique `env`, `docker inspect`, `compose config` expandido, headers Authorization ni hashes reales. El código no incluye cuentas ni claves utilizables del autor. Autorización para gastar y licencia de redistribución son controles distintos.
