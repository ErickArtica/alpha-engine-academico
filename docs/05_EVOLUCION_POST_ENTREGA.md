# Evolución técnica posterior a la entrega académica

Fecha: 5 de noviembre de 2026. Este registro describe trabajo posterior; **no modifica, sustituye ni reescribe el cuestionario universitario entregado**. La evidencia de reconstrucción anterior de la guía 04 conserva su fecha y su SHA.

## Decisión de arquitectura

Marketaux permanece como proveedor elegido para sentimiento BTC. FinBERT, Redis y el procesamiento propio quedan pospuestos. Se conservan NVIDIA Muse Glimmer, XGBoost V1, sus ocho features y el gate determinista 0.60. No se añaden dependencias ni se modifican Cron o tablas de mercado.

## Corrección técnica desplegada

La imagen privada del backend DEMO `marketaux-20261105-aaba6b8b5a4c` incorpora relevancia finita no negativa sin techo de 100, diagnóstico saneado por motivos, caché y cuotas privadas. Está desplegada únicamente en `alpha-engine-academic-api`; el frontend y el backend del pipeline no se actualizan en esta evolución.

- Backend nuevo: `sha256:622510db9462d20aa25249f4a703c2b5b8bc9b84c028e05a6cacf902406b6f30`.
- Rollback conservado: `sha256:f140b51b2260e61567dcf149dc7bcfb81f3f9ab500635476191d0f083f36bd6a`.
- Salud 200; endpoints protegidos rechazan anónimos con 401; imagen Linux/amd64 no root, sin credenciales operativas detectadas en capas/metadatos.
- 89 pruebas offline backend/noticias/helpers y 36 frontend: **125 PASS**. Dentro de la imagen: 58 pruebas runtime PASS; no son 58 adicionales al total anterior.
- Navegación real de seis pantallas, evaluación guardada, replay en memoria y WebSocket estable durante 35 segundos. Ninguna llamada NVIDIA nueva.

La prueba integrada con mocks recorre BTCUSD/CC/cryptocurrency, relevancia 112.536514 y sentimiento 0.13115, caché persistente, agregado y contexto para Muse. Probabilidades y decisión determinista permanecen iguales. Fallos de proveedor conservan el contexto y consumen cuota; no se reintentan. Los ejemplos de prueba no son sentimiento LIVE.

## Autorización y límites de uso

El plan Free publica acceso a `/v1/news/all`, 100 solicitudes diarias y tres noticias por petición. Los términos publicados no establecen inequívocamente derechos para ingesta periódica, almacenamiento de scores y exposición de agregados a terceros. Tener una API key no demuestra ese alcance. La condición personal/no comercial y la cláusula general de acceso automatizado requieren aclarar el alcance concreto de la API.

Por ello siguen en `false` **ACADEMIC_DEMO_NEWS_ENABLED** y **ACADEMIC_DEMO_NEWS_RIGHTS_APPROVED**. El profesor es un tercero incluso si inicia sesión. No se publican artículos, snippets, scores ni agregados. La key no se instala en el frontend. El backend DEMO mantiene su flag NVIDIA y sus usuarios/expiración anteriores, sin ampliación de permisos.

El procesamiento interno y la exposición se separan operacionalmente: una prueba puntual del propietario, expresamente autorizada, puede usar el cliente en memoria y reservar las cuotas privadas existentes; eso **no habilita** el endpoint de actualización del profesor ni un scheduler. No se crea un endpoint alternativo para eludir la licencia. Antes de activar ingesta persistente o exposición se necesita una confirmación verificable del proveedor que cubra ese uso y, si corresponde, el envío de contexto a NVIDIA.

La única prueba LIVE interna autorizada se ejecutó el 05/11/2026 a las 17:04 UTC: **HTTP 401 de Marketaux (AUTHENTICATION_FAILED), 1.123 segundos**. La autenticación DEMO previa sí pasó; el rechazo fue del proveedor. No se recibieron noticias, no se ejecutó su normalización y no hubo agregado ni scores que almacenar. No se guardaron tokens ni contenido completo. El intento consume cuota interna y cooldown; el proveedor no informó consumo facturable. No hubo reintentos ni una segunda solicitud. La evidencia no distingue key incorrecta, revocada, de otro proveedor o copiada con un formato no aceptado. Se necesita comprobar la key en el Dashboard propio y una nueva autorización para otro intento, siempre después del cooldown. Por tanto: IMPLEMENTADO sí; PROBADO LIVE con datos BTC no; DESPLEGADO sí, desactivado; HABILITADO PARA EL PROFESOR no.

Fuentes revisadas: [documentación API](https://www.marketaux.com/documentation), [plan Free](https://www.marketaux.com/pricing), [FAQ](https://www.marketaux.com/faq) y [términos](https://www.marketaux.com/tos). Acceso técnico, autorización para consumir cuota y derechos de redistribución son controles independientes.
