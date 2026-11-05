"""Independent DEMO service: health and academic API only, no pipeline/paper routes."""
from deploy.render.app import RenderApp, _send_json


class AcademicOnly:
    def __init__(self, application=None):
        self.application = application or RenderApp()

    async def __call__(self, scope, receive, send):
        if scope['type'] == 'http':
            path = scope.get('path', '')
            if path != '/health' and not path.startswith('/api/academic/'):
                await _send_json(send, 404, {'status': 'NOT_FOUND'})
                return
        elif scope['type'] == 'websocket':
            await send({'type': 'websocket.close', 'code': 1008})
            return
        await self.application(scope, receive, send)


app = AcademicOnly()
