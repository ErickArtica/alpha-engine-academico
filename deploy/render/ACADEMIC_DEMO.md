# DEMO académico: contrato y límites

Estado 05/11/2026: backend DEMO y frontend desplegados; seis pantallas verificadas.
Muse respondió HTTP 200 en una generación real controlada: 16.361 s, 1179 tokens,
acción WAIT inalterada y explicación comprobada en el frontend DEMO. NVIDIA está
habilitado solo para sesiones autorizadas con cuotas persistentes. Marketaux sigue
desactivado para terceros, pendiente de cuenta/token y derechos de exposición.

El lector público conserva sus controles. El panel adicional de Prescriptivo requiere
activación explícita y sesión autorizada. No habilita paper persistente ni Cron.
El backend actual es ASGI/Uvicorn, no FastAPI; reutiliza los servicios existentes.

## Contrato y límites implementados

- POST `/api/academic/login`: usuario individual y contraseña; hashes PBKDF2-SHA256
  con sal aleatoria y 600000 iteraciones, únicamente en configuración del backend.
  No existe autenticación individual anterior reutilizable en esta aplicación.
- POST `context`, `explain`, `news`, `refresh-news`, `logout` bajo el mismo prefijo:
  bearer de sesión firmado, audiencia fija, identidad aún autorizada, caducidad
  máxima de 15 minutos y revocación al salir. Ventana DEMO explícita de máximo 48 horas.
- Backend valida cada acción; la clave de servicio existente no autentica el DEMO.
  La sesión DEMO tampoco abre las rutas legacy de cartera/Cron/recomendaciones.
- NVIDIA: diez intentos globales/día UTC, tres por usuario/día y sesión/día,
  uno por usuario/minuto; una solicitud global concurrente. Fallos consumen cuota.
  Sin reintentos; streaming SSE validado completo, máximo 2048 tokens y caché privada
  de dos minutos. Conexión 5 s, lectura 60 s y límite total cancelable 85 s; el cliente
  Streamlit termina en 115 s. No se devuelve razonamiento parcial. Los límites opcionales
  se configuran mediante `ACADEMIC_NVIDIA_CONNECT_SECONDS`, `ACADEMIC_NVIDIA_READ_SECONDS`,
  `ACADEMIC_NVIDIA_TOTAL_SECONDS` y `ACADEMIC_NVIDIA_MAX_TOKENS`.
- Marketaux: una consulta global cada 30 minutos, hasta 48/día, diez por usuario/día,
  cinco por sesión/día y una concurrente. Login: cinco intentos globales/minuto,
  cuarenta/día y cinco por usuario/hora. Las cuotas son de ventanas fijas.
  Un límite adicional en memoria rechaza más de sesenta solicitudes/minuto/proceso
  antes del acceso a PostgreSQL; las reservas denegadas sin cambios no escriben estado.
- Estado y cuotas se confirman con compare-and-swap antes de I/O externo. Si falla
  el estado compartido, no se llama al proveedor. No se usa SQLite efímero en Render.
- El servidor lee una predicción V1 persistida y sus velas causales. Reutiliza política
  paper, guardas de riesgo/frescura y adaptador NVIDIA. No infiere, entrena ni escribe
  recomendaciones/carteras de producción. `has_long` siempre es hipotético.
  Una selección explícita `historical` permite evaluar reglas en el instante real de
  generación de una predicción persistida que fue oportuna; se etiqueta HISTORICO,
  no añade noticias actuales y no convierte la decisión histórica en una acción vigente.
- Respuesta LLM: acción, hash de contexto y riesgo deben coincidir; JSON inválido,
  duplicados y contradicciones estructurales se rechazan. También se rechazan cifras del
  texto generado; los valores verificados se muestran por separado. Esto no garantiza la verdad
  de cada frase: presentar la explicación como texto generado, nunca como autoridad.
- La explicación no refresca noticias. El frontend lee la última caché y actualiza
  solo bajo una acción explícita autorizada. Se almacenan scores y timestamps derivados,
  no títulos, URLs, textos, payloads crudos, contraseñas ni tokens.

## Marketaux Free y bloqueo de derechos

Comprobado el 04/11/2026: [Free](https://www.marketaux.com/pricing) anuncia 100
solicitudes/día y tres artículos/consulta. [News/all](https://www.marketaux.com/documentation)
está disponible en todos los planes; las series temporales de Entity Stats requieren
Standard o superior y no se utilizan aquí. Se utiliza `symbols=BTCUSD` (entidad Bitcoin USD, exchange CC), búsqueda
Bitcoin, entidades obligatorias/filtradas, agrupación de similares, tres artículos,
una página y ventana de 24 horas. La media usa solo scores válidos de noticias
recientes de hasta 60 minutos. Sin scores es N/A; positivo ≥ +0.05, negativo ≤ −0.05 y, entre ambos, NEUTRAL.
La normalización exige entidad Bitcoin, deduplica y filtra relevancia mínima diez en DEMO.

Los [términos publicados](https://www.marketaux.com/tos) no ofrecen una autorización
inequívoca para redistribuir contenido o derivados en este dashboard público.
No asumir que acceso Free o finalidad académica resuelve esos derechos. Obtener
confirmación de las condiciones aplicables antes de aprobar visualización/caché
derivada, incluso en DEMO. Hasta entonces, noticias N/A y flags desactivados.
La API real, cuota de la cuenta y símbolos aceptados siguen sin probarse.

No se detectaron claves de proveedores en entorno/local `.env` del worktree.
Los nombres y presencia de configuración del backend DEMO se verificaron sin registrar valores.
NVIDIA_API_KEY está presente; MARKETAUX_API_TOKEN no está configurado en ese servicio. El adaptador Alpha Vantage sigue
existiendo, pero no se activa como fallback silencioso ni se consume su API.
Los experimentos históricos de sentimiento no modifican las ocho features de V1.

El camino Marketaux anterior depende de `claim_marketaux_news`/`marketaux_payload`;
esos objetos no se necesitan para este DEMO: utiliza el estado privado
`academic_demo_state` y las dos RPC ya instaladas, restringidas a service_role.
No se crean columnas/RPC Marketaux potencialmente duplicadas.

## Instalación reproducible en infraestructura propia, tras aprobación

1. Ejecutar únicamente la inspección dirigida `sql/academic_dependencies_inspect.sql`
   con conexión de lectura. No repetir la auditoría ya aprobada de las tres tablas públicas.
   Confirmar que los nombres `academic_demo_*` están libres o corresponden exactamente
   a este contrato. Revisar propietarios, ACL y dependencia de `service_role` para estos objetos.
2. Revisar y autorizar `sql/academic_demo_state.sql`, aplicar manualmente, verificar
   denegación a anon/authenticated y ambos RPC `SECURITY INVOKER`. Solo modifica objetos
   nuevos; no cambia permisos de FastAPI/ASGI, Cron, velas ni recomendaciones existentes.
   `academic_demo_rollback.sql` retira solo el estado DEMO; desactivar DEMO antes de usarlo.
3. Generar credenciales individuales fuertes mediante entrada interactiva, fuera de Git/chat.
   `deploy.render.academic_demo.password_hash` produce el formato requerido con sal aleatoria;
   guardar únicamente su hash en el JSON de usuarios del backend. La contraseña original se
   entrega al usuario autorizado por un canal privado. Generar una clave de sesión independiente.
   No hay contraseñas/usuarios operativos predeterminados ni credenciales en la imagen.
4. Backend: conservar su configuración existente y agregar únicamente los nombres de la tabla
   siguiente. Configurar ventana y usuarios; comenzar con ambos proveedores desactivados.
   La clave Supabase del backend debe poder usar los dos RPC privados; nunca trasladarla al frontend.
5. Exportar fuentes saneadas con `python deploy/render/build_academic_bundle.py`.
   Usar el ZIP inventariado para una fuente de distribución privada separada y revisada,
   sin `.git`, historial, `.env`, pruebas o SQL. No hacer push al repositorio original.
   Crear un Web Service Free NUEVO para esa distribución: no cambiar el backend original.
   Fijar `PYTHON_VERSION` a `3.12.14`. Build: `pip install -r deploy/render/requirements.txt`;
   start: `uvicorn deploy.render.academic_only:app --host 0.0.0.0 --port $PORT --workers 1`.
   Esta entrada excluye Cron, paper, recomendaciones legacy y diagnóstico de proveedor;
   solo admite salud y la API académica autenticada. Los secretos internos exigidos
   por readiness serán nuevos y distintos, sin conectar tareas Cron.
   Validar `/health`, login, lectura y rechazo anónimo antes de habilitar proveedores.
6. Publicar la nueva imagen frontend por un tag distinto (ver `validation/academic-implementation/candidate.json`):

   ```powershell
   $academicTag = (Get-Content deploy/northflank/validation/academic-implementation/candidate.json | ConvertFrom-Json).tag
   $registryTag = 'ghcr.io/erickartica/alpha-engine-frontend:' + ($academicTag -split ':')[-1]
   docker tag $academicTag $registryTag
   docker push $registryTag
   docker buildx imagetools inspect $registryTag
   ```

   Confirmar paquete PRIVATE y digest remoto linux/amd64; no interpretar un índice
   multi-arquitectura como el manifiesto de plataforma. No reutilizar tags anteriores.
   Con autorización, actualizar solo el frontend existente de Render a la referencia por digest.
   Puerto 8501, health `/_stcore/health`; conservar CORS/XSRF y configuración pública actual.
   Rollback: `ghcr.io/erickartica/alpha-engine-frontend@sha256:58e9ae8451d7fb768b6d6560bc72041089cd1138fe3eda53361aac5e093271ea`.
7. Habilitar el panel frontend contra el backend probado. Profesor: Prescriptivo →
   Ingresar al DEMO → Consultar contexto → comprobar ID/vigencia/posición hipotética
   y selección explícita de momento histórico, si procede →
   Generar explicación con IA o leer sentimiento guardado → Cerrar sesión. El replay sigue
   en Simulación; no confundirlo con una cartera persistente.
8. Proveedores: autorización separada y presupuesto inicial propuesto de UNA llamada NVIDIA
   y UNA Marketaux. La prueba técnica interna no autoriza redistribución. Aprobar derechos Marketaux
   antes de habilitar sus flags,
   probar y registrar fecha/modelo/estado/coste permitido. Los límites de código son máximos,
   no autorización para gastar. La validación del 05/11 consumió una llamada nueva Muse
   satisfactoria, además de la llamada anterior fallida. No hay reintentos automáticos.

## Variables adicionales (sin valores operativos)

| Ubicación | Nombres |
|---|---|
| Frontend | `ACADEMIC_DEMO_ENABLED`, `ACADEMIC_DEMO_API_URL` |
| Backend | `ACADEMIC_DEMO_ENABLED`, `ACADEMIC_DEMO_EXPIRES_AT`, `ACADEMIC_DEMO_USERS_JSON`, `ACADEMIC_DEMO_SESSION_SECRET` |
| Backend, aprobación separada | `ACADEMIC_DEMO_LLM_ENABLED`, `ACADEMIC_DEMO_NEWS_ENABLED`, `ACADEMIC_DEMO_NEWS_RIGHTS_APPROVED` |
| Backend NVIDIA existente | `LLM_PROVIDER`, `NVIDIA_API_KEY`, `NVIDIA_BASE_URL`, `NVIDIA_MODEL`, `NVIDIA_REASONING_EFFORT` |
| Backend Marketaux existente | `MARKETAUX_API_TOKEN` |
| Backend almacenamiento existente | `STORAGE_PROVIDER`, `SUPABASE_URL`, `SUPABASE_KEY` |

El frontend rechaza cualquier configuración DEMO exclusiva del servidor. No necesita
clave NVIDIA, token Marketaux, secreto de servicio ni clave administrativa Supabase.
No hay dependencias runtime nuevas. Muse sigue en el [catálogo oficial](https://docs.api.nvidia.com/nim/reference/meta-muse-glimmer-30b);
la integración exige ese modelo explícitamente y no sustituye el modelo histórico.

## Declaraciones académicas honestas

- LIVE observado: seis pantallas, datos/predicciones persistidas y métricas sobre muestra acotada.
- DEMO LIVE comprobado: inicio de sesión individual, contexto histórico determinista y una
  explicación real Muse WAIT visible en el frontend; cuotas privadas persistentes.
- Offline comprobado: BUY/SELL/HOLD/WAIT, errores, streaming, límites y sentimiento simulado.
- HISTÓRICO: TEST, experimento de sentimiento y replay; nunca presentar esos resultados como LIVE.
- Referencia G7 original leída sin modificar/copiar: el HTML documenta BUY/LOW,
  SELL/LOW y HOLD/MEDIUM como tres casos PASS. Se conservan como evidencia histórica
  declarada por ese material; no se reejecutaron llamadas. Las nuevas pruebas reproducen
  esas combinaciones con mocks y añaden WAIT, sin afirmar disponibilidad LIVE del proveedor.
- PENDIENTE: cuenta/token Marketaux y derechos de exposición. La activación del backend,
  frontend DEMO y una explicación real Muse ya están verificadas.
- No implementado en esta entrega: carteras persistentes multiusuario ni trading real.

Un eventual repositorio académico público requiere una exportación nueva revisada de código y
evidencias sintéticas; no convertir el repositorio original ni publicar su historial automáticamente.
