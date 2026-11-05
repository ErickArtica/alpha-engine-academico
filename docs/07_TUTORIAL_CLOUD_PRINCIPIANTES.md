# 07 · Publicar tu propio ALPHA ENGINE, paso a paso

Esta guía crea una instalación **tuya**, separada de la presentación del autor. Necesitas tus cuentas y claves; no tendrás que acceder a su computadora, Supabase, imágenes privadas ni usuarios DEMO. Primero completa el [tutorial local 06](06_TUTORIAL_LOCAL_PRINCIPIANTES.md): así comprobarás Docker y el código antes de introducir servicios remotos.

Revisión: 5 de noviembre de 2026. Se contrastaron comandos, rutas y variables con el repositorio público y se consultó documentación oficial. **No se ejecutó una nueva instalación cloud al redactar estas instrucciones ni se inspeccionaron tus paneles autenticados.** Los recorridos de clics son orientativos; si un botón cambia, sigue el enlace oficial de esa etapa. No adivines opciones ni elijas un plan de pago para continuar.

Las guías técnicas [02](02_DESPLIEGUE_NUBE.md), [03](03_VARIABLES_Y_CREDENCIALES.md) y [04](04_PRUEBAS_DE_RECONSTRUCCION.md) siguen vigentes. Aquí se explican sus pasos para principiantes. El cuestionario académico entregado no se modifica.

## Índice

1. [Entender servicios, cuentas y costes](#1-entender-servicios-cuentas-y-costes)
2. [Crear tu copia en GitHub](#2-crear-tu-copia-en-github)
3. [Crear Supabase y distinguir las claves](#3-crear-supabase-y-distinguir-las-claves)
4. [Crear las tablas mediante migraciones](#4-crear-las-tablas-mediante-migraciones)
5. [Desplegar el backend del pipeline](#5-desplegar-el-backend-del-pipeline)
6. [Configurar Vault y activar Cron](#6-configurar-vault-y-activar-cron)
7. [Crear el backend DEMO y tus usuarios](#7-crear-el-backend-demo-y-tus-usuarios)
8. [Construir y publicar tu imagen privada](#8-construir-y-publicar-tu-imagen-privada)
9. [Desplegar el frontend público](#9-desplegar-el-frontend-público)
10. [Probar pantallas, autenticación y NVIDIA](#10-probar-pantallas-autenticación-y-nvidia)
11. [Marketaux permanece desactivado](#11-marketaux-permanece-desactivado)
12. [Detener consumo y recuperar una versión](#12-detener-consumo-y-recuperar-una-versión)
13. [Errores y comprobación final](#13-errores-y-comprobación-final)

## 1. Entender servicios, cuentas y costes

El despliegue cloud **no es el mismo Compose local**. Aquí habrá tres aplicaciones en Render y una base en Supabase:

| Componente | Responsabilidad | Credenciales permitidas |
|---|---|---|
| Supabase | Base, permisos, Cron y Vault | Administración sólo en tu panel privado |
| Pipeline Render | Descarga Binance, calcula XGBoost V1, persiste predicciones/evaluaciones | Clave Supabase de servidor y dos secretos propios |
| DEMO Render | Login, sesiones, cuotas y explicaciones protegidas | Clave Supabase de servidor, secretos DEMO y NVIDIA opcional |
| Frontend Render | Seis pantallas públicas de consulta y panel DEMO | Sólo clave Supabase publicable/anon y URL de tu DEMO |
| GHCR | Guarda tu imagen Docker del frontend | Token para publicar; otro de sólo lectura para Render |

Ningún componente ejecuta trading real. XGBoost conserva sus ocho features y las reglas el gate **0.60**. Muse explica decisiones ya calculadas; no decide compras ni modifica probabilidades. El código HTTP es **ASGI/Uvicorn**; aunque se describa informalmente como «API», no necesitas agregar FastAPI.

- [ ] Crea cuentas propias en [GitHub](https://github.com/), [Supabase](https://supabase.com/dashboard) y [Render](https://dashboard.render.com/).
- [ ] Instala Git, Python y Docker siguiendo el tutorial 06. La computadora puede ser Windows 11; las imágenes deben ser Linux/amd64.
- [ ] Prepara un gestor de contraseñas y activa autenticación de dos factores cuando esté disponible.
- [ ] NVIDIA es opcional; no necesitas Marketaux para completar el recorrido.

En Render revisa el **workspace → Billing → Monthly Included Usage**, o su equivalente actual. Comprueba también métodos de pago y límites de gasto. Según la [documentación de Free](https://render.com/docs/free), hay **750 horas mensuales compartidas** y suspensión por inactividad; tres servicios continuos superarían ese total. El Cron de un minuto puede mantener activo el pipeline. Free no garantiza servicio permanente ni ausencia de costes por otros conceptos: revisa transferencia, minutos de construcción y condiciones de tu cuenta. No añadas discos, bases Render, planes Starter ni servicios de pago.

Si no puedes confirmar margen y ausencia de cargos no deseados, **quédate con la instalación local**. No uses un monitor externo para impedir que los servicios Free duerman.

## 2. Crear tu copia en GitHub

1. Inicia sesión en GitHub y abre [el repositorio académico](https://github.com/ErickArtica/alpha-engine-academico).
2. Pulsa **Fork**, elige tu cuenta y confirma **Create fork**. Obtendrás `https://github.com/TU_USUARIO/alpha-engine-academico`. Sigue [Fork a repository](https://docs.github.com/en/get-started/quickstart/fork-a-repo) si cambia la interfaz.
3. En PowerShell, descarga **tu fork** en otra carpeta. El siguiente comando te pregunta tu usuario, no una contraseña:

```powershell
$GitHubUser = (Read-Host 'Tu usuario de GitHub, sin arroba').Trim().ToLowerInvariant()
New-Item -ItemType Directory -Path "$env:USERPROFILE\Proyectos" -Force | Out-Null
Set-Location "$env:USERPROFILE\Proyectos"
git clone "https://github.com/$GitHubUser/alpha-engine-academico.git" alpha-cloud-profesor
Set-Location .\alpha-cloud-profesor
git remote -v
git rev-parse HEAD
```

Esperado: `origin` pertenece a **tu usuario**. Guarda el SHA de versión. No necesitas subir cambios funcionales para desplegar. No ejecutes `git add .`: podría incluir archivos privados nuevos. Estas instrucciones no publican ninguna credencial.

Si abres una terminal nueva, vuelve a esta carpeta; las variables PowerShell como `$GitHubUser` no persisten entre ventanas. No copies el `.env` de la práctica local a Render. El cloud usa variables por servicio.

- [ ] Tu fork y clon existen y no apuntan al repositorio privado del autor.

## 3. Crear Supabase y distinguir las claves

1. En tu [Dashboard Supabase](https://supabase.com/dashboard), elige o crea una organización **Free** y pulsa **New project**.
2. Pon un nombre propio, por ejemplo `alpha-profesor`, una contraseña de base nueva y una región cercana a tu Render. Guarda la contraseña en tu gestor; no la pegues en Git ni en comandos.
3. Confirma que no seleccionaste un plan o extra de pago y espera a que el proyecto esté listo.
4. Abre la configuración del proyecto y busca **API Keys**; la URL puede figurar en **Data API** o en el diálogo **Connect**. Las ubicaciones varían: usa la [guía oficial de claves](https://supabase.com/docs/guides/getting-started/api-keys).

Anota privadamente lo siguiente, sin subirlo al repositorio:

| Dato | Para qué se usa | Dónde irá |
|---|---|---|
| Project URL HTTPS, sin `/rest/v1` | Acceso REST/RPC | `SUPABASE_URL` de tus tres servicios |
| Clave **publishable** o legacy **anon** | Lectura sujeta a RLS/GRANT | `SUPABASE_KEY` del frontend |
| Clave **secret** o legacy **service_role** | Operaciones de servidor y RPC privadas | `SUPABASE_KEY` del pipeline y del DEMO, nunca del frontend |
| Contraseña PostgreSQL | Administración/conexiones PostgreSQL | Gestor privado; no se requiere para el runtime REST |

Las claves de servidor son privilegiadas y pueden omitir RLS; no son «de sólo lectura». La seguridad del backend exige además validación de rutas y autenticación. Una clave publicable no sustituye políticas RLS. Las contraseñas de usuarios DEMO se crean más adelante y son otra cosa.

- [ ] Identificaste URL, clave pública y clave de servidor de **tu proyecto nuevo**, sin mezclarlas.

## 4. Crear las tablas mediante migraciones

Una migración es un archivo SQL que prepara la base. **Esta etapa modifica exclusivamente tu Supabase nuevo y vacío.** No la ejecutes en el proyecto del autor ni sobre una base que ya tenga objetos ALPHA ENGINE. Las migraciones históricas no son todas idempotentes: no pulses Run varias veces ante la duda.

Antes de aplicar, verifica el nombre del proyecto y la organización en la parte superior del panel. Si la base ya contiene datos, detente y prepara un respaldo lógico con restauración aislada: no asumas backups automáticos en Free. Consulta [backups de Supabase](https://supabase.com/docs/guides/platform/backups); esta receta de inicio no es una restauración de producción.

### 4.1 Aplicar los SQL uno por uno

En Supabase abre **SQL Editor → New query** (puede aparecer `+`). En tu PowerShell, desde el clon, copia el primer archivo al portapapeles:

```powershell
$ArchivoMigracion = '.\supabase\migrations\20260921053406_initial_alpha_engine_persistence.sql'
Get-Content -Raw -Encoding UTF8 $ArchivoMigracion | Set-Clipboard
```

Vuelve al SQL Editor, pega con **Ctrl+V**, comprueba que empieza con SQL del proyecto, pulsa **Run** y espera confirmación sin errores. Crea una consulta nueva para el siguiente archivo. Cambia `$ArchivoMigracion` por cada ruta de esta tabla y repite `Get-Content ... | Set-Clipboard`. Las rutas y el orden exactos son:

| Orden | Ruta relativa desde el clon | Resultado principal |
|---:|---|---|
| 1 | `supabase/migrations/20260921053406_initial_alpha_engine_persistence.sql` | Tablas de mercado, recomendaciones y RLS base |
| 2 | `supabase/migrations/20260921073450_add_live_observability.sql` | Ejecuciones, vigencia y evaluaciones |
| 3 | `supabase/migrations/20260922021030_add_recommendation_request_audit.sql` | Auditoría de solicitudes |
| 4 | `supabase/migrations/20260922030000_add_public_dashboard_views.sql` | Dos vistas de auditoría; se restringen abajo |
| 5 | `supabase/migrations/20260922040000_allow_rate_limited_recommendation_audit.sql` | Estados de límite |
| 6 | `supabase/migrations/20260922075212_grant_server_read_public_views.sql` | Lecturas para el servidor |
| 7 | `supabase/migrations/20260923024505_schedule_render_live_cycle.sql` | Vault, pg_net, pg_cron y job por minuto |
| 8 | `supabase/migrations/20260924093000_add_prescriptive_news_context.sql` | Estructuras legadas de noticias; no activa proveedor |
| 9 | `supabase/migrations/20260926120000_paper_trading.sql` | Estructuras de cartera académica privada; no habilita cartera pública |
| 10 | `supabase/migrations/20260926160000_live_evaluation_pending.sql` | Función privada de evaluación |
| 11 | `deploy/render/sql/academic_demo_state.sql` | Estado privado DEMO y dos RPC |

Después del **archivo 7**, ve a **Cron → Jobs** y coloca el job **`alpha-engine-live-cycle` en Inactive** hasta configurar backend y Vault en la etapa 6. El [interruptor Active/Inactive está documentado oficialmente](https://supabase.com/docs/guides/cron/quickstart). Si Cron está bajo **Integrations**, sigue esa entrada. La migración ya crea el job: **no pulses Create job ni crees otro**. Antes de configurar Vault puede registrar errores por destino ausente; eso no justifica volver a aplicar la migración.

Si aparece un conflicto de nombre, falta una extensión o Run devuelve error: guarda el código saneado, detén esta etapa y revisa la guía técnica. No borres objetos para «probar otra vez». No uses `deploy/local/schema.sql` aquí: ese esquema corresponde a otro perfil.

### 4.2 Restringir las dos vistas de auditoría en tu proyecto nuevo

Las migraciones históricas concedían lectura pública de vistas internas. El frontend actual **no las necesita**. Antes de publicarlo, ejecuta en una consulta nueva del **proyecto propio recién creado**:

```sql
BEGIN;
REVOKE ALL ON public.public_pipeline_runs,
  public.public_recommendation_request_summary FROM PUBLIC, anon, authenticated;
GRANT SELECT ON public.public_pipeline_runs,
  public.public_recommendation_request_summary TO service_role;
COMMIT;
```

No apliques este ajuste a otras instalaciones ni revoques permisos de tablas/funciones adicionales sin revisar sus dependencias. Los permisos efectivos del autor no se deducen de ejecutar archivos históricos.

### 4.3 Comprobar seguridad antes de aprobar lecturas públicas

Ejecuta estas consultas de inspección en SQL Editor:

```sql
SELECT c.relname, c.relrowsecurity AS rls,
       has_table_privilege('anon',c.oid,'SELECT') AS anon_select,
       has_table_privilege('anon',c.oid,'INSERT,UPDATE,DELETE') AS anon_alguna_escritura,
       has_table_privilege('authenticated',c.oid,'SELECT') AS auth_select,
       has_table_privilege('authenticated',c.oid,'INSERT,UPDATE,DELETE') AS auth_alguna_escritura
FROM pg_class c JOIN pg_namespace n ON n.oid=c.relnamespace
WHERE n.nspname='public' AND c.relname IN
 ('candles','predictions','live_evaluations','academic_demo_state',
  'public_pipeline_runs','public_recommendation_request_summary');

SELECT tablename,policyname,roles,cmd,qual,with_check
FROM pg_policies WHERE schemaname='public';

SELECT p.proname,p.prosecdef AS security_definer,
       has_function_privilege('anon',p.oid,'EXECUTE') AS anon_execute,
       has_function_privilege('authenticated',p.oid,'EXECUTE') AS auth_execute,
       has_function_privilege('service_role',p.oid,'EXECUTE') AS server_execute
FROM pg_proc p JOIN pg_namespace n ON n.oid=p.pronamespace
WHERE n.nspname='public';
```

Resultado mínimo requerido:

- `candles`, `predictions`, `live_evaluations`: RLS activo, lecturas aprobadas por políticas, ninguna escritura pública.
- Las dos vistas de auditoría: SELECT `false` para anon y authenticated.
- `academic_demo_state`: RLS activo y sin SELECT/escritura para anon/authenticated.
- `academic_demo_state_read()` y `academic_demo_state_cas(bigint,jsonb)`: `security_definer=false`; EXECUTE público `false`, servidor `true`. El servidor tiene SELECT/UPDATE sobre el estado; no necesita INSERT/DELETE.
- Revisa **todas** las funciones públicas devueltas, especialmente SECURITY DEFINER. Un permiso heredado o dado a PUBLIC también cuenta. No basta con ocultar un botón.

Revisa también las columnas/datos que serán públicos: sólo información de mercado/predicción/evaluación aprobada, sin personas, trazas internas ni credenciales. Si no puedes interpretar un permiso, pide revisión antes de continuar. No crees políticas `true` amplias ni apruebes datos por el simple hecho de ver un 200. Lee [RLS de Supabase](https://supabase.com/docs/guides/database/postgres/row-level-security).

- [ ] Once SQL aplicados en tu proyecto nuevo, Cron inactivo y permisos públicos revisados. Conservas `NORTHFLANK_PUBLIC_DATA_APPROVED=false` mientras quede cualquier duda.

## 5. Desplegar el backend del pipeline

1. En [Render Dashboard](https://dashboard.render.com/), pulsa **New → Web Service** y elige construir desde repositorio Git.
2. Conecta GitHub y autoriza el acceso a **tu fork**, no a repositorios ajenos. Selecciona tu `alpha-engine-academico`, rama `main`.
3. Asigna un nombre propio disponible; por ejemplo `mi-alpha-pipeline`. La URL final la asigna Render: no asumas que el nombre sugerido estará libre.
4. Elige runtime **Python**, región y **Free**. Deja Root Directory vacío (raíz del repositorio).
5. Configura lo siguiente. **Build y Start se pegan en Render, no se ejecutan en PowerShell**; `$PORT` debe conservarse literalmente:

```text
Build Command: pip install -r deploy/render/requirements.txt
Start Command: uvicorn deploy.render.app:app --host 0.0.0.0 --port $PORT --workers 1
Health Check Path: /health
```

6. En **Environment**, usa **Add Environment Variable** para cada fila siguiente. Si la pantalla inicial lo sitúa bajo Advanced, expándelo. La [guía de variables de Render](https://render.com/docs/configure-environment-variables) explica las opciones actuales de guardado/despliegue.

| KEY del pipeline | Valor que debes configurar |
|---|---|
| `PYTHON_VERSION` | `3.12.14` |
| `APP_ENV` | `production` |
| `STORAGE_PROVIDER` | `supabase` |
| `SUPABASE_URL` | URL de tu proyecto |
| `SUPABASE_KEY` | Clave **de servidor** de tu proyecto |
| `INTERNAL_CRON_SECRET` | Secreto nuevo A, generado abajo |
| `RECOMMENDATION_API_SECRET` | Otro secreto nuevo B, diferente de A |
| `SYMBOL` | `BTCUSDT` |
| `INTERVAL` | `1m` |
| `MODEL_PATH` | `models/xgboost_v1.joblib` |
| `MODEL_CONFIDENCE_THRESHOLD` | `0.60` |
| `LOCAL_DEMO_MODE` | `false` |
| `PRESCRIPTIVE_NEWS_ENABLED` | `false` |
| `V3_SHADOW_ENABLED` | `false` |

Render admite fijar una versión completa con [`PYTHON_VERSION`](https://render.com/docs/python-version); no uses su versión predeterminada para estas dependencias. No necesitas claves NVIDIA ni Marketaux en el pipeline. No elijas el `Dockerfile.backend` para este servicio: ese archivo inicia el **DEMO**, no el pipeline.

### Generar secretos sin mostrarlos

En tu PowerShell, ejecuta **una vez por cada secreto distinto**:

```powershell
python -c "import secrets; print(secrets.token_hex(32))" | Set-Clipboard
```

La salida va al portapapeles, no a la pantalla. Pega con Ctrl+V en el campo privado correspondiente de Render y guarda una copia en tu gestor. **No lo pegues en un editor público, archivo del repositorio o chat.** Genera otra vez para B; conserva A porque debe coincidir con Vault. Al terminar de usar el portapapeles:

```powershell
Set-Clipboard -Value ''
```

Si usas historial del portapapeles de Windows, **Win+V → borrar historial**; vaciar el portapapeles actual no borra por sí solo ese historial. No grabes pantalla al configurar secretos.

7. Confirma de nuevo **Free**, despliega y espera **Live**. Revisa Events/Logs; no exportes variables ni headers. Desactiva Auto-Deploy si quieres revisar cada actualización manualmente (la ubicación varía según la [documentación de Web Services](https://render.com/docs/web-services)).
8. Copia la URL HTTPS que muestra Render. Comprueba en PowerShell, introduciendo tu URL real sin barra final:

```powershell
$PipelineUrl = (Read-Host 'URL HTTPS de TU pipeline').TrimEnd('/')
(Invoke-WebRequest -UseBasicParsing -Uri "$PipelineUrl/health" -TimeoutSec 120).StatusCode
```

Esperado **200**. Un servicio Free dormido puede tardar en responder. No pulses repetidamente acciones operativas para despertarlo. Aún no habrá predicciones cloud si Cron sigue inactivo.

- [ ] Tu pipeline está Live y `/health` responde 200. Guardaste su URL y los dos secretos por separado.

## 6. Configurar Vault y activar Cron

Vault guarda secretos en la base sin ponerlos en el SQL versionado. Cron ejecuta el job; pg_net envía la solicitud al pipeline. No se utiliza un Render Cron Job de pago ni el scheduler de Compose en esta arquitectura.

1. Dentro de **tu Supabase**, busca **Vault** (puede estar en Project Settings o Integrations). La [documentación oficial confirma la administración mediante Dashboard](https://supabase.com/docs/guides/database/vault), pero no fija todos los rótulos actuales. Abre el formulario para añadir un secreto; no uses consultas que impriman `vault.decrypted_secrets`.
2. Crea **exactamente** estos dos nombres, con valores privados:

| Nombre en Vault | Valor |
|---|---|
| `alpha_engine_render_cycle_url` | Tu URL del pipeline más `/internal/run-cycle` |
| `alpha_engine_internal_cron_secret` | El mismo secreto A de `INTERNAL_CRON_SECRET` del pipeline |

La URL debe tener formato `https://nombre-real.onrender.com/internal/run-cycle`; la migración valida ese formato, no dominios personalizados. No pongas la URL del frontend ni la del DEMO. El secreto debe tener al menos 32 caracteres; el generador anterior entrega 64.

3. Abre **Cron → Jobs**, identifica `alpha-engine-live-cycle` y cámbialo a **Active**. Su programación debe ser `* * * * *`: una vez por minuto. No crees un job adicional.
4. Espera dos o tres cierres de minuto y consulta en SQL Editor:

```sql
SELECT jobname,schedule,active FROM cron.job
WHERE jobname='alpha-engine-live-cycle';
SELECT status,start_time,end_time FROM cron.job_run_details
ORDER BY start_time DESC LIMIT 5;
SELECT status,started_at,new_prediction FROM public.pipeline_runs
ORDER BY started_at DESC LIMIT 5;
SELECT count(*) AS velas,max(close_time) AS ultima_vela FROM public.candles;
SELECT count(*) AS predicciones,max(generated_at) AS ultima_prediccion FROM public.predictions;
SELECT count(*) AS evaluaciones,max(evaluated_at) AS ultima_evaluacion FROM public.live_evaluations;
```

Esperado: job activo, ejecuciones nuevas y filas/fechas que avanzan. La primera ejecución obtiene contexto y calcula una predicción; la evaluación requiere el cierre siguiente. Un job «succeeded» puede indicar sólo que se **encoló** un HTTP; confirma también `pipeline_runs` y datos. Un 401 en logs del pipeline suele indicar secretos A distintos; no lo arregles quitando autenticación.

Este montaje cloud empieza vacío y crece con los ciclos. No crea automáticamente las 241 predicciones del bootstrap histórico local ni reproduce el dataset privado. Consulta UTC y la vigencia de cada predicción.

Cron y las trazas ocupan almacenamiento; revisa Usage en Supabase y define una retención propia revisada si lo necesitas. Esta guía no ejecuta limpiezas, borra logs ni instala otra tarea Cron de retención.

- [ ] Hay predicciones nuevas y evaluaciones después del cierre siguiente, no sólo un healthcheck sano.

## 7. Crear el backend DEMO y tus usuarios

Este servicio **es independiente del pipeline**. No habilita carteras persistentes, trading ni acceso público a NVIDIA. Necesita el SQL 11 ya aplicado y cuotas persistentes en Supabase; no lo reemplaces por archivos efímeros de Render.

### 7.1 Crear contraseñas y hashes propios

En el clon `alpha-cloud-profesor`, crea un entorno Python sólo para el asistente. No hace falta activar scripts ni cambiar la política de ejecución de Windows:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install "httpx>=0.27,<1"
.\.venv\Scripts\python.exe scripts/configure_academic_users.py --users administrador profesor companero_01 companero_02
```

Introduce y confirma una contraseña **nueva y diferente para cada persona**, de 12 a 256 caracteres. No aparecerán letras al escribir; es normal. El comando crea `.academic-demo-users.json` con hashes salados, no contraseñas. Se niega a sobrescribir un archivo existente. Son tus usuarios, no los de la presentación.

Comprueba sin mostrar el contenido:

```powershell
Test-Path .\.academic-demo-users.json
git check-ignore .academic-demo-users.json
```

Esperado `True` y el nombre del archivo ignorado. No lo publiques aunque sólo contenga hashes. Entrega cada contraseña por un canal privado a su destinatario, sin capturas colectivas.

### 7.2 Crear el servicio y configurar variables

Repite **Render → New → Web Service → tu fork → Python → Free**, con un nombre disponible para el DEMO. Root Directory vacío, mismo build, **otro Start**:

```text
Build Command: pip install -r deploy/render/requirements.txt
Start Command: uvicorn deploy.render.academic_only:app --host 0.0.0.0 --port $PORT --workers 1
Health Check Path: /health
```

| KEY del DEMO | Valor |
|---|---|
| `PYTHON_VERSION` | `3.12.14` |
| `APP_ENV` | `academic_demo` |
| `STORAGE_PROVIDER` | `supabase` |
| `SUPABASE_URL` | Tu proyecto |
| `SUPABASE_KEY` | Clave **de servidor**, nunca publicable |
| `INTERNAL_CRON_SECRET` | Nuevo secreto C, distinto del pipeline |
| `RECOMMENDATION_API_SECRET` | Nuevo secreto D, distinto de A/B/C |
| `ACADEMIC_DEMO_ENABLED` | `true` cuando usuarios, estado y ventana estén listos |
| `ACADEMIC_DEMO_EXPIRES_AT` | Tu fecha ISO-8601 con zona, no más de 48 horas por delante |
| `ACADEMIC_DEMO_USERS_JSON` | JSON de tus hashes mediante pegado privado |
| `ACADEMIC_DEMO_SESSION_SECRET` | Nuevo secreto E, diferente de los anteriores, al menos 48 caracteres |
| `ACADEMIC_DEMO_LLM_ENABLED` | `false` inicialmente |
| `ACADEMIC_DEMO_NEWS_ENABLED` | `false` |
| `ACADEMIC_DEMO_NEWS_RIGHTS_APPROVED` | `false` |
| `LOCAL_DEMO_MODE` | `false` |
| `PRESCRIPTIVE_NEWS_ENABLED` | `false` |

Genera C, D y E separadamente con el comando `secrets.token_hex(32)` de la etapa 5. No reutilices el secreto Cron del pipeline. Aunque el DEMO exige estos secretos para su configuración, su entrypoint **no expone** la ruta de Cron del pipeline.

Para copiar el JSON directamente al campo privado `ACADEMIC_DEMO_USERS_JSON`:

```powershell
Get-Content -Raw -Encoding UTF8 .\.academic-demo-users.json | Set-Clipboard
```

Para producir una fecha de expiración **propia**, 24 horas después del momento actual:

```powershell
python -c "from datetime import datetime,timedelta,timezone; print((datetime.now(timezone.utc)+timedelta(hours=24)).isoformat())" | Set-Clipboard
```

Pega en `ACADEMIC_DEMO_EXPIRES_AT` después de haber pegado el JSON en su campo. No uses una fecha del autor, una fecha pasada ni 2099. Registra la hora UTC y tu hora local; un reinicio después de expirar no extiende la ventana. Al terminar vacía portapapeles e historial.

Sesiones de **15 minutos** y logout con revocación. NVIDIA tiene **10 intentos globales/día UTC y 3 por usuario/día**; los fallos también consumen cuota. Marketaux conserva máximo 48/día y separación de 30 minutos. No aumentes cuotas ni crees sesiones para eludirlas.

Despliega, espera Live y guarda la URL real:

```powershell
$DemoUrl = (Read-Host 'URL HTTPS de TU backend DEMO').TrimEnd('/')
(Invoke-WebRequest -UseBasicParsing -Uri "$DemoUrl/health" -TimeoutSec 120).StatusCode
curl.exe --silent --output NUL --write-out '%{http_code}\n' --request POST --header 'Content-Type: application/json' --data '{}' "$DemoUrl/api/academic/explain"
curl.exe --silent --output NUL --write-out '%{http_code}\n' --request POST --header 'Content-Type: application/json' --data '{}' "$DemoUrl/api/academic/refresh-news"
```

Esperado **200, 401, 401**: las llamadas anónimas deben rechazarse antes de consultar proveedores. No gastan generaciones. Si recibes 200 en una acción protegida sin sesión, detente y no publiques el enlace DEMO como seguro.

- [ ] DEMO independiente Live, usuarios propios, expiración explícita y rechazo anónimo confirmado.

## 8. Construir y publicar tu imagen privada

Necesitas Docker Desktop encendido en Linux. GHCR es el registro de imágenes de GitHub; no es el repositorio de código. Tu fork público no vuelve automáticamente pública tu imagen ni te da acceso a imágenes privadas del autor.

### 8.1 Token de publicación y login

1. En GitHub, avatar → **Settings → Developer settings → Personal access tokens → Tokens (classic) → Generate new token (classic)**. Estos rótulos pueden cambiar: consulta [autenticación GHCR oficial](https://docs.github.com/en/packages/working-with-a-github-packages-registry/working-with-the-container-registry).
2. Crea un token de duración corta con **`write:packages`**, sin permisos innecesarios de repositorios ni borrado. Si la pantalla agrega `repo`, revisa la selección según la documentación; no concedas acceso amplio por comodidad.
3. Guarda el token en tu gestor. No lo escribas como parte de un comando ni lo metas en `.env`, Git o capturas.
4. Desde el clon y en PowerShell:

```powershell
$GitHubUser = (Read-Host 'Tu usuario GitHub').Trim().ToLowerInvariant()
docker login ghcr.io --username $GitHubUser
```

En **Password**, pega el token **classic**, no la contraseña de GitHub. La entrada no se muestra. Esperado `Login Succeeded`. Docker puede guardarlo en el gestor de credenciales del equipo; no copies su configuración al proyecto.

### 8.2 Construir, verificar y subir una etiqueta nueva

Usa una etiqueta nueva cada vez. Los siguientes comandos incorporan SHA y fecha para no sobrescribir versiones:

```powershell
$Version = 'manual-' + (git rev-parse --short HEAD) + '-' + (Get-Date -Format 'yyyyMMddHHmmss')
$Image = "ghcr.io/${GitHubUser}/alpha-engine-frontend:${Version}"
docker build --platform linux/amd64 -f .\Dockerfile.frontend.northflank -t $Image .
docker image inspect $Image --format '{{.Os}}/{{.Architecture}}'
docker push $Image
docker buildx imagetools inspect $Image
```

Esperado: construcción satisfactoria, **linux/amd64**, push sin error y manifiesto remoto. Contexto de build: la raíz `.`; Dockerfile: `Dockerfile.frontend.northflank`, con su `.dockerignore` específico. No uses `--build-arg` para claves ni agregues archivos privados al contexto. Las credenciales se configuran después en runtime.

En GitHub abre tu perfil → **Packages** → `alpha-engine-frontend` → configuración del paquete. **Verifica visibilidad PRIVATE** antes de continuar; no basta con suponerla. Revisa los [permisos y visibilidad oficiales](https://docs.github.com/en/packages/learn-github-packages/configuring-a-packages-access-control-and-visibility) si no aparece el indicador. Si no puedes confirmar Private, no continúes ni cambies a público para resolver el acceso.

En la salida de `imagetools inspect`:

- Si es un manifiesto simple, guarda el `Digest: sha256:...` que corresponde a esa imagen.
- Si muestra varios **Manifests**, copia el digest de la entrada **Platform: linux/amd64**, no el de `unknown/unknown` de una atestación. El digest del índice completo puede ser diferente.

Construye tu referencia fija y comprueba que existe en el registro:

```powershell
$Digest = (Read-Host 'Digest sha256 del manifiesto linux/amd64').Trim()
if ($Digest -notmatch '^sha256:[a-f0-9]{64}$') { throw 'Digest incompleto: vuelve a comprobar el manifiesto' }
$PinnedImage = "ghcr.io/${GitHubUser}/alpha-engine-frontend@${Digest}"
docker buildx imagetools inspect $PinnedImage
$PinnedImage
```

Guarda esa referencia junto con SHA y etiqueta. Un digest identifica contenido; una etiqueta puede moverse, por eso se despliega por digest.

### 8.3 Credencial de sólo lectura para Render

Crea **otro** token classic con **`read:packages`**, sin `write:packages` ni borrado. Lo pegarás sólo en el formulario privado de registro de Render, con tu usuario GitHub. No lo pongas en variables de Streamlit, README ni URL de imagen. Revoca tokens que ya no necesites desde GitHub; no revoques el de lectura mientras Render dependa de él para tirar de la imagen.

- [ ] Paquete Private confirmado, plataforma linux/amd64 y referencia `@sha256:...` remota guardada.

## 9. Desplegar el frontend público

1. En Render: **New → Web Service → Existing Image**. Pega tu `$PinnedImage` de GHCR y configura una **Registry Credential** con tu usuario y token de sólo lectura. Sigue [Deploying an image](https://render.com/docs/deploying-an-image) si varían los campos.
2. Selecciona un nombre propio, región y **Free**. No uses el servicio del autor ni crees un Static Site: Streamlit requiere un Web Service.
3. Health Check Path: **`/_stcore/health`**. No agregues un Start Command que reemplace el entrypoint de la imagen.
4. Configura únicamente estas variables del frontend:

| KEY del frontend | Valor |
|---|---|
| `PORT` | `8501` |
| `STORAGE_PROVIDER` | `supabase` |
| `SUPABASE_URL` | URL HTTPS de tu proyecto |
| `SUPABASE_KEY` | Tu clave **publishable/anon**, nunca secret/service_role |
| `FRONTEND_PUBLIC_HOST` | El hostname real que te asigna Render, sin `https://`, ruta ni puerto |
| `NORTHFLANK_PUBLIC_READ_ONLY` | `true` siempre |
| `NORTHFLANK_PUBLIC_DATA_APPROVED` | `false` hasta aprobar la etapa 4; después `true` sólo en tu proyecto revisado |
| `SYMBOL` | `BTCUSDT` |
| `INTERVAL` | `1m` |
| `MODEL_CONFIDENCE_THRESHOLD` | `0.60` |
| `PRESCRIPTIVE_NEWS_ENABLED` | `false` |
| `DASHBOARD_FALLBACK_ENABLED` | `false` |
| `LOCAL_DEMO_MODE` | `false` |
| `V3_SHADOW_ENABLED` | `false` |
| `ACADEMIC_DEMO_ENABLED` | `true` sólo si tu DEMO de la etapa 7 está verificado; de lo contrario `false` |
| `ACADEMIC_DEMO_API_URL` | Tu URL base HTTPS DEMO, sin rutas, sólo si habilitas DEMO |

`NORTHFLANK_...` conserva su nombre histórico y funciona en Render sin cambiar su seguridad. No agregues `DATABASE_URL`, secretos de sesión/usuarios, claves NVIDIA/Marketaux o secretos del pipeline aquí: el frontend los rechaza. Tampoco copies toda la configuración del backend por comodidad.

**Primer despliegue y dominio:** si Render no muestra la URL asignada antes de crear el servicio, créalo manteniendo aprobación de datos en false. Ese primer arranque se bloqueará deliberadamente: no es Live sano. Cuando el panel muestre tu URL, copia sólo el hostname en `FRONTEND_PUBLIC_HOST`; después de aprobar RLS/GRANT/datos, cambia tu `NORTHFLANK_PUBLIC_DATA_APPROVED` a true, guarda y vuelve a desplegar. No inventes el dominio ni elimines el guard para que arranque.

Espera **Live**, copia tu URL HTTPS y comprueba:

```powershell
$FrontendUrl = (Read-Host 'URL HTTPS de TU frontend').TrimEnd('/')
(Invoke-WebRequest -UseBasicParsing -Uri "$FrontendUrl/_stcore/health" -TimeoutSec 120).StatusCode
```

Esperado **200**. Abre la URL en el navegador; la principal debe mostrar Home/Alpha Engine. No pongas `/descriptivo` como base de un proxy: la aplicación está en la raíz. El canal WebSocket es `/_stcore/stream`; algunos 404 transitorios de detección de rutas no prueban por sí solos una sesión rota.

- [ ] Frontend Live por digest, lectura pública segura y sin secretos de servidor en sus variables.

## 10. Probar pantallas, autenticación y NVIDIA

### 10.1 Pantallas y datos sin iniciar sesión

Abre una ventana privada del navegador para simular un visitante. Recorre:

| Pantalla | Verificación |
|---|---|
| Home | Encabezado, menú y acceso a las otras cinco |
| Descriptivo | OHLC, volumen, indicadores y fecha de última vela |
| Predictivo | XGBoost V1, UP/DOWN, probabilidades y fecha/vigencia |
| Evaluación | Registros tras el siguiente cierre y métricas de la muestra; ninguna cifra implica precisión global |
| Prescriptivo | Reglas, gate 0.60 y acción determinista; sin generación IA anónima |
| Simulación | Replay en memoria, avance de pasos y gráficos; sin escrituras de cartera pública |

Comprueba que las fechas avanzan con tu Cron; un gráfico visible puede ser antiguo. Si tu base nueva todavía tiene pocos registros, espera a obtener más. No presentes bootstrap/replay o predicciones vencidas como señales LIVE.

Para revisar la conexión del navegador, pulsa **F12 → Network/Red**, recarga y filtra por **WS**: `/_stcore/stream` debería establecer conexión **101** y mantenerse mientras navegas. También revisa errores 500 y los logs de tus servicios. No copies mensajes con tokens o credenciales. Un healthcheck 200 no sustituye este recorrido.

### 10.2 Login, contexto, logout y expiración

1. En **Prescriptivo**, busca **DEMO académico autorizado**.
2. Prueba una contraseña incorrecta una sola vez: debe rechazarla. No agotes los límites de login.
3. Escribe tu usuario `profesor` y su contraseña propia → **Ingresar al DEMO**. Esperado «Sesión DEMO autorizada».
4. Mantén el ID de predicción propuesto si existe. Si no, en tu SQL Editor obtén uno con `SELECT prediction_id FROM public.predictions ORDER BY generated_at DESC LIMIT 1;` y pégalo en **ID de predicción persistida**. No inventes el ID.
5. Pulsa **Consultar contexto DEMO**. Comprueba predicción, decisión determinista y sus motivos. Esto no llama al proveedor.
6. Si estás trabajando con una predicción ya vencida, marca **Evaluar reglas en el momento histórico (no LIVE)**. El resultado debe identificarse como histórico, sin noticias actuales. No cambies el reloj ni la vigencia guardada.
7. Pulsa **Cerrar sesión DEMO**: desaparecen acciones autorizadas. Abre una ventana privada nueva y confirma que pide login.
8. Para comprobar expiración, inicia sesión, deja pasar más de 15 minutos sin cerrarla y confirma que solicita reingreso. No amplíes la ventana global para pasar una prueba.

Las pruebas de agotamiento de cuotas, respuestas contradictorias, sesiones cruzadas y fallos de API se ejecutan con mocks, no gastando créditos. Con un entorno Python **3.12** y las dependencias del repositorio instaladas, los archivos exactos son:

```powershell
python -m pytest tests/test_academic_demo.py tests/test_academic_provider_reliability.py -q
```

Estos tests no prueban que tus claves reales funcionen ni reemplazan el login en tu despliegue. Para preparar ese entorno completo, usa [04 · Pruebas](04_PRUEBAS_DE_RECONSTRUCCION.md); el pequeño entorno `httpx` de la etapa 7 no incluye pytest ni todas las dependencias. Las cuotas usan el estado privado en Supabase: no lo edites ni lo borres para reiniciar contadores. Conservarlo tras redeploy es requisito del DEMO.

### 10.3 NVIDIA Muse opcional, sólo en el DEMO

1. Crea/inicia sesión en tu cuenta de [NVIDIA Build: Muse Glimmer](https://build.nvidia.com/meta/muse-glimmer-30b).
2. Comprueba términos, créditos disponibles, modelo y ausencia de cargos adicionales **en tu cuenta**. El botón «free API endpoint» no garantiza crédito ilimitado. No actives un plan de pago. Si no puedes verificar el coste o el modelo, conserva LLM apagado.
3. La página del modelo ofrece **Generate API Key**. Genera tu propia clave, guárdala privadamente y revisa la [referencia oficial del modelo](https://docs.api.nvidia.com/nim/reference/meta-muse-glimmer-30b). No pruebes desde el playground si quieres controlar el número de llamadas.
4. En **Environment de tu DEMO**, no en frontend/pipeline, configura:

| KEY | Valor |
|---|---|
| `LLM_PROVIDER` | `nvidia` |
| `NVIDIA_API_KEY` | Tu clave privada |
| `NVIDIA_MODEL` | `meta/muse-glimmer-30b` |
| `NVIDIA_BASE_URL` | `https://integrate.api.nvidia.com/v1` |
| `NVIDIA_REASONING_EFFORT` | `low` |
| `ACADEMIC_DEMO_LLM_ENABLED` | `true` sólo después de comprobar coste, autorización y cuota |

5. Guarda y despliega **sólo tu DEMO**, comprueba `/health` y vuelve a iniciar sesión si hace falta.
6. Consulta primero contexto para un ID real y verifica la decisión. Pulsa **Generar explicación con IA una sola vez**. No recargues ni pulses otra vez mientras espera. El cliente permite como máximo 115 s; el transporte proveedor tiene límites de conexión/lectura/total de 5/60/85 s y no reintenta automáticamente.
7. Esperado: explicación breve, modelo identificado y la **misma decisión** que las reglas. Si falla, la decisión permanece visible; el intento consume cuota igualmente. No llames «LIVE» a una respuesta mock o histórica.
8. Anota hora, estado, latencia si se informa y cuota, sin copiar claves. Si no necesitas más generaciones, vuelve a `ACADEMIC_DEMO_LLM_ENABLED=false` en tu DEMO.

NVIDIA nunca es requisito para mostrar las pantallas ni para calcular XGBoost. No hay un modelo alternativo silencioso: si Muse deja de estar disponible, documenta el bloqueo y no cambies evidencias históricas. Una explicación no es una promesa de rentabilidad.

- [ ] Seis pantallas, login/logout/expiración comprobados; IA sólo si autorizaste y verificaste una generación real propia.

## 11. Marketaux permanece desactivado

**No prometas sentimiento Marketaux LIVE.** El código acepta Bitcoin `BTCUSD`, relevancia finita no negativa incluso mayor a 100 y sentimiento entre −1 y 1. Normaliza hasta 24 h, pero el agregado actual sólo considera noticias BTC elegibles de hasta una hora; no convierte ausencia de noticias en neutral. Tiene deduplicación, caché, diagnósticos saneados, cuota persistente de 48/día y separación mínima de 30 minutos.

Sin embargo, una prueba técnica interna HTTP 200 no obtuvo un sentimiento BTC LIVE validado y los derechos para almacenar/exponer scores o agregados ante terceros siguen sin aprobarse. Una API key Free **no equivale** a permiso de redistribución. Consulta [documentación](https://www.marketaux.com/documentation), [precios](https://www.marketaux.com/pricing) y [términos](https://www.marketaux.com/tos), y solicita aclaración al proveedor si corresponde.

En esta reconstrucción conserva:

```text
PRESCRIPTIVE_NEWS_ENABLED=false
ACADEMIC_DEMO_NEWS_ENABLED=false
ACADEMIC_DEMO_NEWS_RIGHTS_APPROVED=false
```

No necesitas configurar `MARKETAUX_API_TOKEN`. No pulses **Actualizar sentimiento (controlado)** para intentar conseguir datos: el bloqueo es intencional. Mostrar N/A o un resultado histórico correctamente fechado es válido; inventar un score LIVE no lo es. No envíes noticias a NVIDIA ni agregues consultas al Cron de predicción. Activar proveedor, revisar licencia y autorizar una prueba son etapas separadas de esta guía. FinBERT y Redis quedan fuera de esta versión.

- [ ] Noticias apagadas; N/A, STALE e HISTÓRICO se distinguen de neutral y LIVE.

## 12. Detener consumo y recuperar una versión

### Al terminar la práctica

1. En **tu Supabase → Cron → Jobs**, coloca **tu** `alpha-engine-live-cycle` en **Inactive**. Si sólo cierras el navegador, Cron continúa llamando al pipeline y puede gastar horas.
2. En tu DEMO desactiva `ACADEMIC_DEMO_LLM_ENABLED`; conserva noticias apagadas. Cierra sesión en el frontend. Si el ejercicio terminó, desactiva tu DEMO o deja vencer su ventana; no la renueves sin autorización del dueño del entorno.
3. En **cada servicio propio de Render**, busca la opción de suspender el servicio en Settings/controles del servicio. La ubicación exacta **no se verificó en un panel autenticado**: consulta la [documentación de administración](https://render.com/docs) o el soporte si no aparece. **No elijas Delete ni cambies a un plan de pago.** Si no lo suspendes manualmente, Free puede dormir por inactividad, pero no mientras siga recibiendo tráfico.
4. Revisa Billing/Usage para confirmar que no siguen consumiéndose horas o transferencia. No crees monitores que mantengan despiertos los servicios.
5. Supabase puede pausar proyectos Free inactivos: [pausa y recuperación oficiales](https://supabase.com/docs/guides/platform/free-project-pausing). No es un respaldo recuperable garantizado de tu trabajo. Antes de retirar un proyecto con datos, prepara y prueba tu propio respaldo con asistencia si la necesitas. No borres base/tablas como receta de ahorro.
6. En tu computadora, detén también la práctica local siguiendo el tutorial 06 y cierra Docker Desktop si ya no lo usas.

Para reanudar: verifica planes/cuotas, reanuda primero Supabase, después pipeline y DEMO, comprueba healthchecks; revisa la ventana DEMO propia, luego frontend y por último reactiva **el mismo** Cron cuando Vault y destino sean correctos. No crees un segundo job.

### Actualizar o volver atrás

Antes de cada cambio guarda privadamente: SHA del código, digest/etiqueta del frontend, revisión desplegada de cada backend y configuración de sus variables. No compartas capturas de secretos. Los archivos locales de Render son efímeros; sesiones/cuotas deben seguir en Supabase.

Si falla un proveedor, desactiva **sólo su flag del DEMO**. Si falla una nueva imagen frontend, usa el digest anterior de **tu propio paquete**, conservado en GHCR, y vuelve a desplegar sólo ese frontend. Para un backend, abre su página **Deploys**, localiza una revisión satisfactoria conservada y usa **Rollback**, si está disponible en tu plan; consulta [rollback de Render](https://render.com/docs/rollbacks). Verifica salud y acceso tras recuperar. El rollback puede reutilizar variables de la revisión anterior: revisa especialmente flags de proveedores y expiración antes de confirmar.

No uses digests privados del autor, no restaures toda la base, no borres el estado DEMO y no reapliques migraciones para deshacer una imagen. Un rollback de aplicación no revierte automáticamente cambios de esquema; esta guía no incluye ampliaciones de esquema posteriores.

- [ ] Sabes detener tu Cron/servicios, conservar datos y volver a una versión propia anterior.

## 13. Errores y comprobación final

| Problema | Comprobación y acción segura |
|---|---|
| SQL: objeto ya existe | Confirma proyecto y migraciones ya ejecutadas. Detente; no borres tablas ni vuelvas a aplicar todo |
| No aparece Vault/Cron | Revisa extensiones del SQL 7 y guías oficiales. No simules Vault guardando claves en SQL/Git |
| Job succeeded pero sin predicciones | Confirma Vault, HTTP del pipeline, `pipeline_runs` y acceso a Binance. Encolar HTTP no prueba éxito de inferencia |
| Render build falla con dependencias | Runtime Python y `PYTHON_VERSION=3.12.14`, build en raíz y archivo exacto. No actualices XGBoost ni sustituyas el modelo |
| API tarda al despertar | Free puede arrancar en frío. Espera healthcheck; no dupliques generaciones NVIDIA |
| Frontend bloqueado al inicio | Revisa hostname real, clave publicable, flags y aprobación de RLS. No apagues el perfil de sólo lectura |
| GHCR unauthorized | Token classic correcto y propietario en minúsculas. Render requiere token separado read:packages con acceso al paquete Private |
| Render no puede descargar imagen | Comprueba digest existente, plataforma y credencial de registro. No hagas público el paquete como arreglo |
| DEMO 401 | Cuenta/contraseña, sesión y fecha; no sustituir por secreto compartido ni compartir tokens |
| DEMO 503 | Readiness, estado privado, claves de servidor, ventana/modelo y logs saneados; `/health` no prueba una generación real |
| DEMO 429 | Esperar límites; los fallos cuentan. No borrar cuotas ni abrir sesiones para eludirlas |
| Noticias 403 / N/A | Esperado con derechos/NEWS apagados; no marcar aprobación sin fundamento |
| Muse 429/502/timeout | Guardar código y latencia, no reintentar sin presupuesto/autorización. Las reglas siguen disponibles |
| WebSocket cae o hay 500 persistentes | Revisar Network/WS y logs de frontend; conservar CORS/XSRF. No añadir proxies por un 404 transitorio aislado |

Lista de cierre:

- [ ] Sólo utilizaste cuentas/recursos propios y no publicaste claves, contraseñas, hashes ni archivos `.env`.
- [ ] Aplicaste/revisaste esquema y RLS en tu proyecto; RPC DEMO y vistas internas sin acceso público.
- [ ] Pipeline y Cron producen datos con fechas nuevas; las evaluaciones corresponden a velas cerradas.
- [ ] Los tres servicios tienen healthchecks correctos y planes revisados; no hay recursos de pago añadidos.
- [ ] Frontend de sólo lectura; usuarios anónimos no pueden llamar proveedores ni persistir carteras.
- [ ] Tus usuarios DEMO pueden consultar contexto; logout y expiración funcionan.
- [ ] El modelo sigue siendo XGBoost V1, con ocho features y gate 0.60.
- [ ] Si probaste NVIDIA, registraste el resultado real y mantuviste las cuotas. No llamas LIVE a mocks.
- [ ] Marketaux sigue desactivado hasta resolver permisos y pruebas autorizadas.
- [ ] Guardaste tus referencias de rollback y sabes detener Cron y servicios.

Una reconstrucción no está comprobada sólo por terminar de leer la guía. Marca únicamente lo que realmente ejecutaste. Si te bloqueas, comparte versión, paso y error saneado; nunca compartas secretos para recibir ayuda.
