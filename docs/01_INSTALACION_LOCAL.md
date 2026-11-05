# 01 · Instalación local independiente

Esta ruta levanta **PostgreSQL local + API ASGI/Uvicorn + Streamlit completo + bootstrap + scheduler**. No requiere Supabase, Render, GHCR, contraseñas del autor ni claves de proveedores. Necesita Internet para descargar imágenes/dependencias y obtener velas públicas nuevas de Binance. No reproduce los registros históricos privados originales ni vuelve a entrenar XGBoost.

La instalación cloud es diferente: usa Supabase, Cron, un lector público y un backend DEMO autenticado separado. Véase [02](02_DESPLIEGUE_NUBE.md). En local, las acciones académicas quedan limitadas a `127.0.0.1`; no publique este Compose en Internet ni active túneles sobre él.

## 1. Requisitos

- Windows 10/11 de 64 bits compatible con la versión instalada de Docker Desktop, virtualización y WSL 2; consulte los [requisitos vigentes de Docker](https://docs.docker.com/desktop/setup/install/windows-install/). No todas las ediciones/versiones de Windows 10 siguen soportadas. Seleccione contenedores **Linux**.
- Linux: Docker Engine y plugin Compose v2; en Debian/Ubuntu, instálelos siguiendo la documentación oficial de Docker para su distribución. Permita acceso a Docker a su usuario siguiendo la política de su equipo.
- Git; Python 3.12 o superior para los helpers. Si no instala Python en el host, puede ejecutar los helpers con una imagen pública Python, como se indica abajo.
- Espacio libre de varios GB: las dependencias XGBoost/scikit-learn y las imágenes son grandes. El primer build puede tardar varios minutos. Mantenga Docker iniciado.

```powershell
git --version
docker version
docker compose version
python --version
```

`docker version` debe mostrar **Client y Server**, con motor Linux disponible. Tener sólo el comando instalado no significa que el daemon esté iniciado. No necesita Node, pnpm ni Supabase CLI para esta ruta.

## 2. Clonar fuera de cualquier instalación existente

En PowerShell o bash, desde una carpeta propia:

```text
git clone https://github.com/ErickArtica/alpha-engine-academico.git alpha-profesor
cd alpha-profesor
git rev-parse HEAD
```

Conserve el SHA para identificar la versión evaluada. No copie `.env`, bases de datos, volúmenes Docker ni archivos personales de otra instalación. Todo comando posterior se ejecuta en este clon.

## 3. Crear configuración y secretos NUEVOS

El `.env.example` de la distribución pública sirve para el **frontend cloud**. Compose utiliza el ejemplo separado **`.env.local.example`**. El siguiente helper lo toma como plantilla, crea `.env` y sustituye tres placeholders por secretos aleatorios diferentes de 64 caracteres hexadecimales:

```text
python scripts/configure_local.py --project alpha-profesor --api-port 18000 --frontend-port 18501
```

No imprime los secretos, no lee credenciales externas y rechaza sobrescribir un `.env` existente. El archivo queda excluido de Git. En Windows conserve también las restricciones de acceso de su cuenta. Los puertos deben estar libres; si están ocupados, elija otros antes de crear la configuración. Nunca detenga un servicio ajeno para liberar un puerto.

Alternativa sin Python instalado, PowerShell:

```powershell
docker run --rm --network none --mount "type=bind,source=$($PWD.Path),target=/repo" -w /repo python:3.12-slim python scripts/configure_local.py --project alpha-profesor --api-port 18000 --frontend-port 18501
```

Linux/macOS:

```bash
docker run --rm --network none --mount "type=bind,source=$PWD,target=/repo" -w /repo python:3.12-slim python scripts/configure_local.py --project alpha-profesor --api-port 18000 --frontend-port 18501
```

Docker puede necesitar descargar primero esa imagen pública. En Linux, el archivo generado por un contenedor puede quedar perteneciendo a root; utilice preferentemente Python del host o añada `--user "$(id -u):$(id -g)"` al comando anterior.

La configuración deja noticias apagadas y claves NVIDIA/Alpha Vantage vacías. No hacen falta para mercado, inferencia V1, evaluación ni replay. No ejecute `docker compose config` sin `--quiet` en una captura: su salida expandida contiene secretos.

## 4. Construir y arrancar

```text
docker compose --project-name alpha-profesor --env-file .env config --quiet
docker compose --project-name alpha-profesor --env-file .env build
docker compose --project-name alpha-profesor --env-file .env up -d
docker compose --project-name alpha-profesor --env-file .env ps --all
python scripts/verify_local.py --observe-scheduler
```

El verificador espera healthchecks hasta 180 segundos y observa un ciclo nuevo hasta 160 segundos adicionales. Devuelve JSON con `status: PASS`, HTTP 200, cantidades de filas, comprobación del modelo y rechazo 401 del disparador sin autenticación. No llama NVIDIA, Marketaux ni un backend remoto. Este verificador requiere Python y Docker CLI en el host. Sin Python del host, puede hacer manualmente las comprobaciones de HTTP, `compose ps`, logs y SQL descritas abajo; no monte el socket Docker en contenedores ajenos para sortear este requisito.

| Servicio | Resultado esperado |
|---|---|
| postgres | running / healthy; datos en volumen propio; sin puerto PostgreSQL publicado |
| paper_migrate | exited (0); tablas académicas locales creadas idempotentemente |
| api | running / healthy; `http://127.0.0.1:18000/health` responde 200 |
| bootstrap | exited (0); comprobar además filas: el proceso puede terminar sin datos si Binance falla |
| scheduler | running / healthy; un POST local autenticado por minuto |
| streamlit | running / healthy; `http://127.0.0.1:18501/_stcore/health` responde 200 |

`up -d` devuelve el control antes de que todos los healthchecks estén listos. Un reset de conexión inmediatamente después no demuestra un fallo del sistema; espere al verificador. Sólo un exit 0 de bootstrap no basta para declarar PASS.

## 5. Datos propios y causalidad

Bootstrap consulta aproximadamente seis horas de velas cerradas BTCUSDT de Binance. Usa ventanas causales de 120 velas y el artefacto XGBoost V1 incluido: ocho features, sin entrenamiento. Una ejecución con 360 velas continuas suele producir 241 predicciones y 240 evaluaciones; las cifras dependen del momento y de la disponibilidad del mercado.

Estas primeras evaluaciones son **replay histórico local**, registrado en `local_bootstrap_runs`, no una nueva evaluación prospectiva. Después el scheduler llama a la API local y crea nuevas predicciones/evaluaciones a medida que cierran velas. No llama un LLM. Una nueva ejecución de bootstrap no duplica el replay ya terminado.

```text
docker compose --project-name alpha-profesor --env-file .env exec -T postgres psql -U alpha_engine -d alpha_engine -c "SELECT count(*) AS candles FROM candles; SELECT count(*) AS predictions, max(generated_at) AS latest FROM predictions; SELECT count(*) AS evaluations FROM live_evaluations; SELECT kind, candle_count, prediction_count FROM local_bootstrap_runs;"
docker compose --project-name alpha-profesor --env-file .env logs --tail 30 scheduler bootstrap api
```

Debe observar `LOCAL_BOOTSTRAP` y, posteriormente, `LOCAL_SCHEDULER run-cycle HTTP 200`. Los valores nuevos pertenecen a su instalación; no se prometen las métricas ni los registros privados del autor.

## 6. Abrir las seis pantallas

Abra `http://127.0.0.1:18501/`: Inicio/Alpha Engine, Descriptivo, Predictivo, Prescriptivo, Evaluación y Simulación. La navegación local usa `streamlit_app/`; el lector cloud usa `deploy/northflank/` y muestra **Home**. No son el mismo entrypoint.

- Descriptivo: OHLC, volumen e indicadores sobre velas locales.
- Predictivo: predicciones guardadas; modelo V1 y su fecha.
- Evaluación: distinguir replay local, nuevas observaciones y TEST histórico incluido.
- Prescriptivo: conserva el gate 0.60. Sin claves externas, un estado no disponible del LLM es correcto; no es un error del predictor.
- Simulación: avanzar un minuto en el replay antes de esperar gráficos de equity/operaciones. No opera dinero real. No confundir replay en memoria con las rutas privadas de cartera local.

El demo local heredado y el DEMO cloud autenticado no son equivalentes. El segundo necesita un Supabase propio y las sesiones/cuotas de [02](02_DESPLIEGUE_NUBE.md). Las pruebas con mocks del backend DEMO funcionan sin Supabase; véase [04](04_PRUEBAS_DE_RECONSTRUCCION.md).

## 7. Reiniciar, detener y limpiar SOLO este proyecto

Compruebe primero carpeta y proyecto. Mantenga siempre `--project-name alpha-profesor` y `--env-file .env`; si cambió el nombre, sustituya ambos comandos consistentemente.

```text
docker compose --project-name alpha-profesor --env-file .env restart api streamlit scheduler
docker compose --project-name alpha-profesor --env-file .env stop
docker compose --project-name alpha-profesor --env-file .env start
docker compose --project-name alpha-profesor --env-file .env down
```

`down` conserva el volumen local. Para borrar explícitamente los datos de **esta instalación de práctica**:

```text
docker compose --project-name alpha-profesor --env-file .env down --volumes --remove-orphans
```

Esto elimina su PostgreSQL local; el siguiente arranque vuelve a descargar datos y ejecutar bootstrap. No lo ejecute contra una instalación compartida. No use `docker system prune`, comodines ni borrados de volúmenes globales. Conserve `.env` privado si quiere reutilizar sus claves locales.

## 8. Problemas frecuentes

| Síntoma | Acción concreta |
|---|---|
| No existe `dockerDesktopLinuxEngine` | Iniciar Docker Desktop, seleccionar Linux y esperar Server en `docker version`. Revisar WSL/virtualización; no instalar herramientas dentro del proyecto original. |
| Falta POSTGRES_PASSWORD | Se copió el ejemplo cloud: usar `configure_local.py` / `.env.local.example`, no el ejemplo cloud. |
| Puerto ocupado | Elegir puertos libres en una instalación nueva; no detener el puerto original. |
| API 503 | Revisar `logs api`; secretos diferentes de al menos 32 caracteres, PostgreSQL healthy y checksum del modelo. |
| Bootstrap sin filas | Revisar `logs bootstrap`; Internet, DNS, hora UTC y respuesta de Binance. Corregir conectividad y ejecutar `docker compose --project-name alpha-profesor --env-file .env run --rm bootstrap`. No declarar PASS con tablas vacías. |
| Binance 451/403 | Restricción regional/proveedor. No usar VPN ni evasión automática; comprobar una fuente pública oficial permitida y configurar `BINANCE_BASE_URL` explícitamente antes de recrear los servicios. Si no hay fuente accesible, la reconstrucción de datos LIVE queda bloqueada. |
| Scheduler sin crecimiento | Esperar una vela cerrada y comprobar `logs scheduler api`; revisar reloj y acceso a Binance. No pulsar repetidamente acciones de proveedores. |
| Paquetes no descargan | Revisar red, espacio y repositorios públicos. No sustituir versiones del modelo para eludir un error de instalación. |
| LLM/noticias N/A | Esperado sin claves propias o autorización; las funciones básicas siguen operativas. |

Siguiente: [variables y credenciales](03_VARIABLES_Y_CREDENCIALES.md) y [pruebas reproducibles](04_PRUEBAS_DE_RECONSTRUCCION.md).
