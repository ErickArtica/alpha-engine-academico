# 06 · ALPHA ENGINE en tu computadora, desde cero

Esta guía está dirigida a quien nunca ha usado programación ni Docker. Primero instala la versión **local**, sin cuentas de Supabase, Render, NVIDIA o Marketaux. Necesitas Internet para descargar programas, contenedores y velas públicas de Binance. No necesitas una cuenta de Binance ni depositar dinero.

Trabaja en una carpeta nueva. Los comandos sólo administran el proyecto Docker `alpha-profesor`, con API en el puerto **18000** y pantalla en el **18501**, accesibles desde tu computadora. No apuntes a servicios del autor ni ejecutes estos pasos sobre otra instalación.

Revisión: 5 de noviembre de 2026. Los comandos se contrastaron con los archivos actuales del repositorio. No se repitió un despliegue completo al redactar este tutorial; la instalación independiente ejecutada anteriormente y su SHA están en [04 · Pruebas de reconstrucción](04_PRUEBAS_DE_RECONSTRUCCION.md). No se promete recuperar datos privados ni el historial de la investigación.

## Índice

1. [Qué vas a instalar](#1-qué-vas-a-instalar)
2. [Preparar Windows 11](#2-preparar-windows-11)
3. [Descargar el proyecto](#3-descargar-el-proyecto)
4. [Crear tu configuración privada](#4-crear-tu-configuración-privada)
5. [Construir e iniciar](#5-construir-e-iniciar)
6. [Comprobar datos y automatización](#6-comprobar-datos-y-automatización)
7. [Recorrer las seis pantallas](#7-recorrer-las-seis-pantallas)
8. [Autenticación e inteligencia artificial](#8-autenticación-e-inteligencia-artificial)
9. [Detener, reanudar o borrar tu práctica](#9-detener-reanudar-o-borrar-tu-práctica)
10. [Solucionar problemas](#10-solucionar-problemas)
11. [Linux y siguiente paso](#11-linux-y-siguiente-paso)

## 1. Qué vas a instalar

- [ ] Una computadora Windows 11 de 64 bits con virtualización habilitada e Internet. Recomendación práctica: 8 GB de RAM o más y unos 15 GB de espacio libre; no es una garantía de rendimiento.
- [ ] Permiso para instalar programas y reiniciar Windows.
- [ ] Git: descarga el código. No es una cuenta ni una contraseña.
- [ ] Python 3.13: ejecuta los asistentes locales. El Python del contenedor se instala por separado; no hay que cambiarlo.
- [ ] Docker Desktop con contenedores Linux: ejecuta PostgreSQL, la API, la pantalla y los procesos automáticos.

**No instales PostgreSQL, Node.js, Supabase CLI, CUDA, Redis ni FinBERT para este recorrido.** Docker instala la base local; el modelo XGBoost V1 ya está incluido. No se vuelve a entrenar.

Palabras útiles: una **terminal** recibe comandos; un **repositorio** contiene archivos del proyecto; una **imagen** es un paquete de ejecución; un **contenedor** es ese paquete funcionando; un **volumen** conserva la base de datos aunque detengas el contenedor; una **API** recibe solicitudes de otros componentes.

## 2. Preparar Windows 11

Los nombres de botones siguientes son orientativos: no se inspeccionaron tus instaladores. Si cambian, sigue las páginas oficiales enlazadas; no descargues instaladores de anuncios ni páginas de terceros.

### 2.1 Abrir PowerShell

1. Haz clic derecho en **Inicio → Terminal** y abre una pestaña **Windows PowerShell** o **PowerShell**.
2. Para los comandos normales no necesitas «Administrador».
3. Copia sólo el contenido de cada bloque, sin los caracteres de las líneas que lo rodean. Pulsa **Enter**. Espera a que reaparezca la línea del cursor antes del siguiente bloque.

Puedes usar Windows PowerShell 5.1 incluido en Windows; PowerShell 7 es opcional ([instalación oficial](https://learn.microsoft.com/en-us/powershell/scripting/install/install-powershell-on-windows)). Estos comandos no son para CMD.

### 2.2 WSL y Docker Desktop

1. Abre **Inicio**, escribe `Terminal`, pulsa botón derecho → **Ejecutar como administrador**.
2. Ejecuta:

```powershell
wsl --install
```

3. Reinicia si Windows lo pide. Si Ubuntu solicita usuario y contraseña, crea los tuyos; al escribir la contraseña no aparecen caracteres. No son contraseñas de ALPHA ENGINE.
4. Cierra la terminal de administrador. Abre una terminal normal y comprueba:

```powershell
wsl --status
wsl --version
```

Debe mostrarse información de WSL, sin errores de virtualización. Si WSL ya estaba instalado, no hace falta repetir la instalación. Consulta [WSL oficial](https://learn.microsoft.com/en-us/windows/wsl/install) si requiere actualización o habilitar virtualización.

5. Desde [Docker Desktop para Windows](https://docs.docker.com/desktop/setup/install/windows-install/), descarga el instalador adecuado a tu equipo. Sigue el asistente y utiliza el motor **WSL 2** cuando se ofrezca.
6. Abre **Docker Desktop** desde Inicio, revisa sus condiciones de uso y espera a que el motor esté funcionando. Usa **contenedores Linux**, no Windows. No necesitas publicar nada ni iniciar sesión en un registro para construir este entorno.
7. Abre otra PowerShell y ejecuta:

```powershell
docker --version
docker compose version
docker info --format '{{.OSType}}'
```

Debes ver versiones de Docker/Compose y finalmente **`linux`**. Un error que dice que no puede conectarse al motor significa que Docker Desktop no terminó de iniciar.

### 2.3 Git y Python

1. Descarga **Git for Windows**, instalador x64 para equipos Intel/AMD habituales, desde [Git oficial](https://git-scm.com/install/windows). Instala con las opciones predeterminadas que permitan usar Git desde la terminal.
2. En [Python para Windows](https://www.python.org/downloads/windows/), busca una versión **3.13** con **Windows installer (64-bit)**. No descargues el paquete «embeddable» ni una versión preliminar. En el instalador clásico, marca **Add python.exe to PATH** si aparece y continúa con la instalación.
3. Cierra y vuelve a abrir PowerShell para actualizar las rutas. Ejecuta:

```powershell
git --version
python --version
python -m pip --version
```

Esperado: Git instalado y Python **3.13.x**; los asistentes requieren al menos 3.12. Si `python` abre la tienda, corrige instalación/PATH siguiendo la [documentación oficial de Python en Windows](https://docs.python.org/3/using/windows.html). No instales paquetes globales para intentar resolverlo.

- [ ] Docker responde `linux`; Git, Python y pip muestran sus versiones.

## 3. Descargar el proyecto

No necesitas cuenta GitHub para descargar este repositorio público. Abre PowerShell normal y ejecuta:

```powershell
New-Item -ItemType Directory -Path "$env:USERPROFILE\Proyectos" -Force | Out-Null
Set-Location "$env:USERPROFILE\Proyectos"
git clone https://github.com/ErickArtica/alpha-engine-academico.git alpha-profesor
Set-Location .\alpha-profesor
git rev-parse HEAD
Get-ChildItem -Name
```

Verás un identificador largo de versión y archivos como `docker-compose.yml`, `README.md`, `.env.local.example` y carpetas `scripts`, `models` y `docs`. Guarda el identificador para comunicar errores. **No copies carpetas privadas del autor.**

Si ya existe `alpha-profesor`, no la sobrescribas: elige otra carpeta para una práctica nueva. En las próximas terminales vuelve a esta carpeta con:

```powershell
Set-Location "$env:USERPROFILE\Proyectos\alpha-profesor"
```

Comprueba que los puertos sugeridos estén libres:

```powershell
Get-NetTCPConnection -LocalPort 18000,18501 -State Listen -ErrorAction SilentlyContinue
```

Lo normal para una práctica nueva es que no aparezcan filas. Si aparecen, **no detengas el proceso**: en la siguiente etapa utiliza otros dos puertos libres, por ejemplo 18010 y 18511, y usa esos mismos números en las URLs del tutorial.

- [ ] Estás dentro del clon nuevo y conoces su versión.

## 4. Crear tu configuración privada

Un archivo `.env` guarda la configuración. Hay dos ejemplos diferentes: **`.env.local.example` es para Compose local**; `.env.example` es para el lector cloud y no sirve para este paso.

El asistente siguiente toma el ejemplo local y crea `.env` con tres secretos nuevos y distintos, sin mostrarlos. No necesita claves de proveedores.

```powershell
python scripts/configure_local.py --project alpha-profesor --api-port 18000 --frontend-port 18501
Test-Path .\.env
git check-ignore .env
```

Esperado: mensaje `Created local .env ... values suppressed`, luego `True` y `.env`. Si te avisa que `.env` ya existe, se negó a sobrescribirlo para proteger tus claves; conserva ese archivo y continúa sólo si corresponde a esta práctica.

No pegues el archivo en chats, GitHub, capturas o informes. No necesitas abrirlo para ejecutar el sistema. Puedes consultar **los nombres**, sin valores, en [03 · Variables](03_VARIABLES_Y_CREDENCIALES.md).

Valida la configuración sin imprimir secretos:

```powershell
docker compose --project-name alpha-profesor --env-file .env config --quiet
```

Resultado esperado: vuelve al cursor sin errores. No quites `--quiet`: la configuración expandida puede revelar claves. Mantén `PRESCRIPTIVE_NEWS_ENABLED=false`, la clave NVIDIA vacía y el gate **0.60**.

- [ ] Existe `.env`, está ignorado por Git y la validación termina sin error.

## 5. Construir e iniciar

Deja Docker Desktop abierto. Ejecuta uno por uno:

```powershell
docker compose --project-name alpha-profesor --env-file .env build
docker compose --project-name alpha-profesor --env-file .env up -d
docker compose --project-name alpha-profesor --env-file .env ps --all
```

La primera construcción puede tardar varios minutos y descargar varios GB. No cierres la terminal mientras construye. `up -d` deja los servicios ejecutándose en segundo plano.

| Servicio en `ps --all` | Qué hace | Resultado normal tras arrancar |
|---|---|---|
| `postgres` | Base PostgreSQL local | Up, healthy |
| `paper_migrate` | Crea estructuras locales de simulación | Exited (0), es una tarea única |
| `api` | API local ASGI/Uvicorn y modelo | Up, healthy |
| `bootstrap` | Descarga velas y calcula el historial inicial | Exited (0) al terminar; confirmar filas en el paso 6 |
| `scheduler` | Solicita un ciclo por minuto | Up |
| `streamlit` | Dashboard privado local | Up, healthy cuando termine de iniciar |

«Exited (0)» en las dos tareas únicas no es un fallo. «Restarting», «unhealthy» o un código de salida distinto de cero requiere revisar logs. Un bootstrap sin errores de proceso no garantiza que Binance entregara datos.

Consulta las últimas líneas de tus propios servicios:

```powershell
docker compose --project-name alpha-profesor --env-file .env logs --tail 40 bootstrap scheduler
docker compose --project-name alpha-profesor --env-file .env logs --tail 40 api streamlit
```

Para seguir logs, añade `--follow` antes del nombre del servicio; **Ctrl+C** deja de verlos, pero no detiene los contenedores. No publiques logs sin revisarlos.

- [ ] Los servicios permanentes están en marcha y las tareas únicas terminaron correctamente.

## 6. Comprobar datos y automatización

Primero, ejecuta el verificador incluido:

```powershell
python scripts/verify_local.py --observe-scheduler
```

Espera varios minutos: comprueba arranque, base, modelo, ocho features, gate 0.60 y un avance del scheduler. No llama a NVIDIA/Marketaux. Debe devolver JSON con **`PASS`**; consulta los campos de recuento antes/después. No tiene que coincidir exactamente con un número del autor.

Comprueba también ambos healthchecks:

```powershell
(Invoke-WebRequest -UseBasicParsing -Uri 'http://127.0.0.1:18000/health').StatusCode
(Invoke-WebRequest -UseBasicParsing -Uri 'http://127.0.0.1:18501/_stcore/health').StatusCode
```

Resultado: **200** en los dos. Esto confirma procesos, no todas las funcionalidades. Para consultar tus datos:

```powershell
docker compose --project-name alpha-profesor --env-file .env exec -T postgres psql -U alpha_engine -d alpha_engine -c "SELECT count(*) AS velas FROM candles; SELECT count(*) AS predicciones FROM predictions; SELECT count(*) AS evaluaciones FROM live_evaluations; SELECT max(generated_at) AS ultima_prediccion FROM predictions;"
```

Debe haber velas, predicciones y evaluaciones tras terminar el bootstrap. Una ejecución independiente anterior obtuvo 360/241/240 al inicio, pero **esas cifras son evidencia histórica, no un requisito fijo ni una precisión del modelo**. La siembra reproduce velas cerradas recientes de Binance; el scheduler añade predicciones prospectivas después. Si bloquean Binance en tu ubicación, no se inventan datos: consulta la sección de errores.

- [ ] Hay filas reales y el verificador observa nuevas predicciones/evaluaciones.

## 7. Recorrer las seis pantallas

Abre en tu navegador **http://127.0.0.1:18501/**. La barra lateral permite cambiar de pantalla. Si está plegada, abre el control de expansión de Streamlit en la esquina superior izquierda.

| Pantalla | Qué debes hacer | Resultado que debes comprobar |
|---|---|---|
| Home / Inicio | Abrir la principal | Encabezado Alpha Engine y resumen; el nombre del menú puede diferir del lector cloud |
| Descriptivo | Abrir y usar actualización de datos | Velas OHLC, volumen e indicadores; comprobar fechas, no sólo que haya gráficos |
| Predictivo | Consultar predicción e historial | UP/DOWN, probabilidades, XGBoost V1 y fecha; una fila vieja no es una señal vigente |
| Evaluación | Abrir cuando haya evaluaciones | Filas y métricas de la muestra seleccionada; no llamar «precisión global» al porcentaje mostrado |
| Prescriptivo | Revisar la política | BUY/SELL/HOLD/WAIT según datos y posición; gate 0.60; la ausencia de explicación IA no impide mostrar reglas |
| Simulación | Elegir **Replay Histórico** en «Modo de simulación», después **Siguiente minuto >** | Evolución, gráficos y cartera virtual en memoria; al paso cero puede no haber gráfico todavía |

No necesitas pulsar «Crear cartera virtual», «Aplicar operación simulada», generar explicaciones ni otras acciones persistentes para validar el replay. El modo local también contiene funciones privadas que no están disponibles para un visitante del frontend público.

El replay es histórico y académico. No conecta con una cuenta de exchange ni realiza compraventas reales. El estado en memoria puede perderse al reiniciar la sesión del navegador. No esperes ver los registros privados originales del autor.

- [ ] Visitaste las seis pantallas sin excepción y avanzaste al menos un paso en el replay.

## 8. Autenticación e inteligencia artificial

**Esta instalación Compose es un dashboard privado en tu computadora, no la arquitectura pública con login DEMO.** Los puertos están vinculados a `127.0.0.1`; no abras firewall, túneles ni puertos del router para publicarla. No agregues una contraseña sólo a Streamlit como sustituto de autorización backend.

La API local rechaza invocaciones operativas sin su secreto; el verificador ya comprueba un rechazo **401** en `/internal/run-cycle`. No escribas el secreto en un comando para hacer pruebas manuales.

Para comprobar inicio/cierre de sesión, expiración y explicación NVIDIA con cuentas propias, sigue el [tutorial cloud, etapas 7 y 10](07_TUTORIAL_CLOUD_PRINCIPIANTES.md#7-crear-el-backend-demo-y-tus-usuarios). No uses las cuentas de presentación del autor: no se publican contraseñas ni hashes.

Las pruebas offline no requieren claves ni gastan créditos. Si quieres ejecutarlas, crea un entorno Python separado; para las dependencias completas usa Python **3.12**, como los contenedores, y sigue los comandos de [04 · Pruebas](04_PRUEBAS_DE_RECONSTRUCCION.md). No cambies versiones del modelo para hacer desaparecer avisos.

Marketaux permanece **desactivado**: la normalización y los diagnósticos tienen pruebas, pero no hay sentimiento BTC LIVE validado para terceros y faltan derechos de exposición. N/A no significa neutral. NVIDIA es opcional, necesita cuenta/créditos propios y autorización antes de una generación; no es un requisito para demostrar predicción, reglas o replay.

- [ ] Entiendes qué funciones locales no requieren autenticación DEMO y cuáles necesitan el backend protegido.

## 9. Detener, reanudar o borrar tu práctica

Ejecuta siempre dentro de **tu clon** y conserva `--project-name alpha-profesor`. No uses comandos globales de limpieza Docker.

**Pausar sin perder datos:**

```powershell
docker compose --project-name alpha-profesor --env-file .env stop
docker compose --project-name alpha-profesor --env-file .env ps --all
```

**Reanudar lo que pausaste:**

```powershell
docker compose --project-name alpha-profesor --env-file .env start
python scripts/verify_local.py --observe-scheduler
```

**Retirar contenedores y red, conservando la base:**

```powershell
docker compose --project-name alpha-profesor --env-file .env down
```

Después de `down`, reanuda con `up -d`, no con `start`. La base persiste en el volumen. Si cambias tu `.env` o el código, usa `up -d --build` para recrear con esos cambios; `restart` no aplica variables nuevas.

**Borrado irreversible sólo de los datos de esta práctica:** si quieres empezar desde cero y no necesitas conservar su historial, comprueba primero el nombre de proyecto en `ps --all`. Únicamente entonces:

```powershell
docker compose --project-name alpha-profesor --env-file .env down --volumes --remove-orphans
```

No ejecutes esa última orden sobre otra instalación. No es necesaria para terminar una sesión. Las imágenes permanecen para reconstruir más rápido; no hace falta borrarlas. Detener el stack evita CPU, RAM y nuevas consultas públicas desde tu equipo.

- [ ] Al terminar, tus contenedores están detenidos y sabes si conservaste o eliminaste los datos.

## 10. Solucionar problemas

| Síntoma | Qué hacer |
|---|---|
| `git` o `python` no se reconoce | Reabrir PowerShell; comprobar instalador/PATH oficial. No pegar comandos en CMD |
| Docker no conecta o devuelve `windows` | Abrir Docker Desktop; esperar al motor y seleccionar contenedores Linux. Revisar WSL/virtualización según Docker oficial |
| `no configuration file provided` | Volver al clon con `Set-Location`; debe existir `docker-compose.yml` |
| `.env already exists` | El asistente protege el archivo. No lo regeneres ni compartas. Para una práctica distinta usa un clon y nombre de proyecto nuevos |
| Puerto ocupado | No matar procesos. Editar **sólo tu `.env`** (`notepad .env`), cambiar `LOCAL_API_PORT`/`LOCAL_STREAMLIT_PORT` a puertos libres, guardar y recrear con `up -d`. Usar las nuevas URLs |
| Pantalla no abre | Revisar `ps --all`, healthchecks y logs propios. Dar tiempo al primer arranque; no confundir 18501 con 8501 interno |
| Bootstrap termina pero tablas vacías | Leer su log: Binance puede bloquear por región, devolver error o estar caído. Comprobar acceso y condiciones del proveedor. No pegar datos inventados ni usar un proxy desconocido para eludir restricciones |
| Evaluaciones vacías | Esperar bootstrap y cierre de nuevas velas. Verificar scheduler. No borrar volumen como primera solución |
| Replay no dibuja | Seleccionar Replay Histórico y avanzar un minuto; esperar a tener filas |
| Explicación/sentimiento no disponible | Esperado sin proveedores autorizados. Las reglas y el replay deben seguir funcionando |
| Error de descarga/poco espacio | Comprobar conexión y espacio de Docker. No limpiar todos los volúmenes ni alterar otros proyectos |

Para pedir ayuda, indica el SHA, el servicio, el código de error y un fragmento saneado del log. Nunca incluyas `.env`, contraseñas, tokens, headers Authorization ni capturas de variables.

## 11. Linux y siguiente paso

En Linux instala Git, Python y **Docker Engine con el plugin Compose** siguiendo [Docker oficial por distribución](https://docs.docker.com/engine/install/) y [Compose](https://docs.docker.com/compose/install/linux/). Comprueba `docker --version`, `docker compose version` y `python3 --version`. No pegues órdenes PowerShell como `Set-Location` en Bash.

Desde tu clon Linux, los comandos equivalentes básicos son:

```bash
python3 scripts/configure_local.py --project alpha-profesor --api-port 18000 --frontend-port 18501
docker compose --project-name alpha-profesor --env-file .env config --quiet
docker compose --project-name alpha-profesor --env-file .env up -d --build
python3 scripts/verify_local.py --observe-scheduler
```

Revisa los permisos Docker de tu usuario con la guía oficial; pertenecer al grupo Docker otorga privilegios elevados. La configuración local sigue siendo privada. Para publicar una instalación propia con permisos, usuarios y servicios separados, continúa con [07 · Tutorial cloud para principiantes](07_TUTORIAL_CLOUD_PRINCIPIANTES.md). Las guías técnicas [01](01_INSTALACION_LOCAL.md) y [04](04_PRUEBAS_DE_RECONSTRUCCION.md) se conservan como referencia.
