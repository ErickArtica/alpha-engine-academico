# ALPHA ENGINE — proyecto académico

## Guías actuales de reconstrucción

1. [Instalación local](docs/01_INSTALACION_LOCAL.md)
2. [Despliegue cloud](docs/02_DESPLIEGUE_NUBE.md)
3. [Variables y credenciales](docs/03_VARIABLES_Y_CREDENCIALES.md)
4. [Pruebas de reconstrucción](docs/04_PRUEBAS_DE_RECONSTRUCCION.md)

**Reconstrucción local independiente PASS (05/11/2026):** clon público sin parches, PostgreSQL vacío, bootstrap y scheduler reales, seis páginas y 122 pruebas offline. [Registro y SHA probado](docs/04_PRUEBAS_DE_RECONSTRUCCION.md#ejecución-independiente-registrada-5-de-noviembre-de-2026). No equivale a reconstruir el historial privado ni a desplegar cloud.

La corrección Marketaux permite relevancia superior a 100 y añade diagnósticos de descarte. Está preparada y probada en fuentes/local; no se ha desplegado ni activado Marketaux en Render. Noticias siguen apagadas por derechos pendientes. Compose usa `.env.local.example` y secretos propios; no depende de Supabase ni Render. El perfil local y el cloud son distintos.


Sistema de Business Analytics para BTC/USDT: analítica descriptiva, predictiva y prescriptiva. No ejecuta trading real ni garantiza rentabilidad.

**Aplicación pública funcionando:** https://alpha-engine-frontend.onrender.com/

Esta distribución pública contiene frontend, backend, dependencias, modelo V1, pruebas y SQL de referencia. Es una exportación limpia, sin historial del repositorio original, credenciales, cuentas DEMO, datos crudos ni respaldos.

## Estado real de las funciones

| Función | Estado comprobado |
|---|---|
| Inicio / Descriptivo | LIVE: lecturas de velas persistidas, indicadores y gráficos |
| Predictivo | LIVE: consulta de predicciones guardadas XGBoost V1; ocho features |
| Evaluación | LIVE: consulta de evaluaciones; métricas de muestras, no precisión global |
| Prescriptivo | Reglas BUY/SELL/HOLD/WAIT; gate experimental **0.60** conservado |
| Simulación | Replay histórico y cartera virtual en memoria; sin persistencia pública |
| Acceso DEMO | Backend independiente, cuentas individuales, sesiones de quince minutos, cuotas persistentes |
| NVIDIA Muse Glimmer | DEMO LIVE comprobado el 05/11/2026: una generación real WAIT coherente, 16.361 s y 1179 tokens, visible en el frontend autorizado. SSE con límite total de 85 s; cuotas persistentes; sin acceso anónimo. La prueba anterior falló por timeout |
| Marketaux / sentimiento | Cliente BTCUSD y agregado probados offline. Prueba técnica interna HTTP 200; no se obtuvo un sentimiento BTC LIVE validado. Sigue desactivado para terceros por derechos pendientes; token operativo no instalado en el backend DEMO. N/A no significa neutral |
| Carteras persistentes | Código disponible; excluidas del frontend público y del backend académico aislado |

La disponibilidad del endpoint público no equivale a acceso a las acciones protegidas. No se incluyen usuarios, contraseñas ni claves. El modo DEMO actual tiene expiración temporal; no se garantizan credenciales de presentación permanentes.

## Arquitectura y código

- `alpha_engine/`: dominio, repositorios, pipeline, features, XGBoost y adaptadores de IA/noticias.
- `deploy/northflank/`: frontend público seguro utilizado en Render (el nombre conserva la estrategia inicial).
- `streamlit_app/`: dashboard completo para entornos privados/locales; no es el perfil público desplegado.
- `deploy/render/academic_only.py`: backend académico ASGI/Uvicorn; únicamente health y API académica. No expone Cron, cartera ni rutas operativas originales.
- `deploy/render/academic_demo.py`, `academic_state.py`, `academic_service.py`: autenticación, cuotas y contexto determinista.
- `models/`: artefacto V1 validado y metadatos agregados; no contiene datasets de entrenamiento.
- `supabase/migrations/`: SQL histórico de referencia. **No representa necesariamente los permisos efectivos de producción y no se aplica automáticamente.**
- `deploy/render/sql/academic_demo_state.sql`: estado privado aditivo y dos RPC restringidas a service_role.
- `tests/` y `scripts/`: pruebas y herramientas de investigación; los scripts con efectos requieren un entorno propio y autorización.

## Instalación para el profesor

Requisitos: Python 3.12 o superior y, opcionalmente, Docker Linux. No necesita instalar nada para visitar la aplicación pública.

```bash
git clone https://github.com/ErickArtica/alpha-engine-academico.git
cd alpha-engine-academico
python -m venv .venv
# Windows: .venv\Scripts\activate
# Linux/macOS: source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pytest tests/test_academic_demo.py tests/test_academic_provider_reliability.py -q
```

Las pruebas académicas utilizan mocks y estado temporal; no necesitan claves reales. No ejecutar pruebas marcadas de integración ni scripts de ingesta contra proyectos ajenos. La instalación completa incluye dependencias de investigación; el frontend tiene su lista reducida independiente.

### Frontend público con infraestructura propia

```bash
docker build --platform linux/amd64 -f Dockerfile.frontend.northflank -t alpha-engine-academico:frontend .
docker run --rm --env-file .env.frontend -p 127.0.0.1:8501:8501 alpha-engine-academico:frontend
```

Preparar `.env.frontend` a partir de `.env.example` con la URL y clave **publicable/anon de su propio proyecto**. El perfil falla de forma segura hasta aprobar los datos con RLS revisado y `NORTHFLANK_PUBLIC_DATA_APPROVED=true`. Mantener `NORTHFLANK_PUBLIC_READ_ONLY=true`. No incorporar variables privadas del backend al frontend. Configurar `FRONTEND_PUBLIC_HOST=localhost` para la ejecución local. Healthcheck: `/_stcore/health`.

### Backend DEMO separado

```bash
docker build --platform linux/amd64 -f Dockerfile.backend -t alpha-engine-academico:backend .
docker run --rm --env-file .env.backend -p 127.0.0.1:10000:10000 alpha-engine-academico:backend
```

Preparar `.env.backend` desde `.env.backend.example`. Para ejecutar sin Docker: `uvicorn deploy.render.academic_only:app --host 127.0.0.1 --port 10000 --workers 1`. Healthcheck: `/health`.

El backend requiere claves de servidor de su propio entorno. El esquema privado debe revisarse y aplicarse separadamente; no hay migraciones en el arranque. Para habilitar DEMO, generar hashes PBKDF2 individuales con `deploy.render.academic_demo.password_hash`, secretos nuevos y distintos, y una fecha ISO-8601 con zona horaria a no más de cuarenta y ocho horas. No guardar contraseñas, hashes de usuarios reales ni tokens en Git. NVIDIA y noticias permanecen apagados por defecto.

### Render y GHCR

Construir y publicar imágenes propias con etiquetas nuevas; mantener el registro privado y desplegar por digest SHA256. En Render crear servicios independientes: frontend puerto 8501 y health `/_stcore/health`; backend puerto 10000 y health `/health`. Uvicorn utiliza un solo worker. Configurar variables exclusivamente en cada servicio. El frontend puede recibir `ACADEMIC_DEMO_ENABLED=true` y `ACADEMIC_DEMO_API_URL` del backend propio; nunca claves de NVIDIA ni service_role.

Seleccionar Free solo si hay cuota suficiente. Las horas gratuitas son compartidas y los servicios pueden dormir por inactividad. No modificar el despliegue de la presentación ni aplicar SQL sobre él. Conservar el digest anterior antes de cualquier actualización propia. No hay GitHub Actions ni webhooks de despliegue en esta distribución.

## Seguridad y límites académicos

El LLM explica una decisión calculada por reglas; no modifica XGBoost, probabilidades ni gate. El backend rechaza anónimos, aplica expiración y cuotas persistentes (NVIDIA: diez intentos globales y tres por usuario al día; fallos consumen cuota). Los proveedores requieren autorización independiente. No se ofrecen claves ni acceso administrativo a Supabase. Los permisos de tablas y RPC deben revisarse en una instalación propia antes de publicar datos.

Las cifras y artefactos de modelos corresponden a investigación histórica; no son asesoramiento financiero. `LICENSE` cubre el código propio; librerías, proveedores y datos conservan sus licencias.
