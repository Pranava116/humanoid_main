"""Web dashboard (FastAPI + websocket, one page): live IMU and joints, state, camera, commands, e-stop, log tail.

Connects to a robot process over the network link, or runs in the robot process itself:

  python -m robot.interface.dashboard                      # robot on this machine (robot.run already running)
  python -m robot.interface.dashboard --connect 192.168.1.20
  python -m robot.run --backend sim --dashboard             # robot + dashboard in one process

Then open http://localhost:8000. There is no login: anyone who can reach the port can drive the robot.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import sys
import threading
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse, Response, StreamingResponse

from robot.behavior.commands import KINDS, Command
from robot.config import Config, load_config
from robot.interface.bus import Link

log = logging.getLogger(__name__)

STATIC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")
PUSH_HZ = 10.0


def meta(cfg: Config) -> Dict[str, Any]:
    """Static info the page needs: joint names and limits, gestures, command kinds."""
    return {
        "joints": [{"name": j.name, "min": j.min_rad, "max": j.max_rad} for j in cfg.joints],
        "gestures": sorted((cfg.get("behavior.gestures") or {}).keys()),
        "kinds": list(KINDS),
        "default_speed": float(cfg.get("behavior.walking.default_speed", 0.02)),
    }


def parse_command(body: Dict[str, Any]) -> Command:
    """JSON from the page -> Command (raises ValueError on bad input)."""
    if not isinstance(body, dict) or "kind" not in body:
        raise ValueError("expected {kind, args}")
    args = body.get("args") or {}
    if not isinstance(args, dict):
        raise ValueError("args must be an object")
    return Command(str(body["kind"]), dict(args), "dashboard")


def create_app(link: Link, cfg: Config) -> FastAPI:
    """FastAPI app bound to one Link."""
    app = FastAPI(title="humanoid dashboard")
    info = meta(cfg)

    def status() -> Dict[str, Any]:
        s = dict(link.status())
        s["connected"] = link.connected and bool(s)
        return s

    @app.get("/")
    def index() -> FileResponse:
        return FileResponse(os.path.join(STATIC, "index.html"))

    @app.get("/api/meta")
    def get_meta() -> Dict[str, Any]:
        return info

    @app.get("/api/status")
    def get_status() -> Dict[str, Any]:
        return status()

    @app.get("/api/logs")
    def get_logs(since: int = 0) -> List[Any]:
        return [list(x) for x in link.logs(since)]

    @app.post("/api/command")
    def post_command(body: Dict[str, Any]) -> Dict[str, Any]:
        try:
            cmd = parse_command(body)
        except ValueError as e:
            raise HTTPException(400, str(e))
        link.send(cmd)
        return {"ok": True, "id": cmd.id, "command": str(cmd)}

    @app.get("/camera.jpg")
    def camera() -> Response:
        jpg = link.frame()
        if jpg is None:
            return Response(status_code=204)
        return Response(jpg, media_type="image/jpeg", headers={"Cache-Control": "no-store"})

    @app.get("/camera.mjpg")
    async def camera_stream() -> StreamingResponse:
        async def frames():
            last = None
            while True:
                jpg = link.frame()
                if jpg is not None and jpg is not last:
                    last = jpg
                    yield (b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: "
                           + str(len(jpg)).encode() + b"\r\n\r\n" + jpg + b"\r\n")
                await asyncio.sleep(0.1)
        return StreamingResponse(frames(), media_type="multipart/x-mixed-replace; boundary=frame")

    @app.websocket("/ws")
    async def ws(sock: WebSocket) -> None:
        """Pushes {status, logs} at PUSH_HZ; accepts {kind, args} commands."""
        await sock.accept()
        seq = 0

        async def reader() -> None:
            while True:
                body = await sock.receive_json()
                try:
                    link.send(parse_command(body))
                except ValueError as e:
                    await sock.send_json({"error": str(e)})

        task = asyncio.ensure_future(reader())
        try:
            while not task.done():
                lines = link.logs(seq)
                if lines:
                    seq = lines[-1][0]
                await sock.send_json({"status": status(), "logs": [list(x) for x in lines]})
                await asyncio.sleep(1.0 / PUSH_HZ)
        except (WebSocketDisconnect, RuntimeError):
            pass
        finally:
            task.cancel()

    @app.exception_handler(ValueError)
    def bad_value(_req, exc: ValueError) -> JSONResponse:
        return JSONResponse({"detail": str(exc)}, status_code=400)

    return app


class DashboardServer:
    """uvicorn in a background thread (the control loop keeps the main thread; uvicorn skips signals there)."""

    def __init__(self, app: Any, host: str, port: int) -> None:
        import uvicorn
        self.server = uvicorn.Server(uvicorn.Config(app, host=host, port=port, log_level="warning"))
        self.host, self.port = host, port
        self._thread: Optional[threading.Thread] = None

    def start(self) -> "DashboardServer":
        self._thread = threading.Thread(target=self.server.run, name="dashboard", daemon=True)
        self._thread.start()
        shown = "localhost" if self.host in ("0.0.0.0", "127.0.0.1") else self.host
        log.info("dashboard: http://%s:%d", shown, self.port)
        return self

    def stop(self) -> None:
        self.server.should_exit = True
        if self._thread is not None:
            self._thread.join(timeout=3.0)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="python -m robot.interface.dashboard", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--connect", metavar="HOST", help="robot address (default network.robot_host)")
    p.add_argument("--host", help="address to serve on (default network.dashboard.host)")
    p.add_argument("--port", type=int, help="port (default network.dashboard.port)")
    p.add_argument("--config-dir", help="config directory (default config/stack)")
    p.add_argument("--local", action="store_true",
                   help="run the robot in this process too (same as robot.run --dashboard); "
                        "remaining options go to robot.run, e.g. --backend sim --viewer")
    p.add_argument("--log-level", default="INFO")
    return p


def main(argv: Optional[List[str]] = None) -> int:
    args, rest = build_parser().parse_known_args(argv)
    if args.local:
        from robot import run
        extra = ["--config-dir", args.config_dir] if args.config_dir else []
        if args.port:
            extra += ["--dashboard-port", str(args.port)]
        return run.main(rest + extra + ["--dashboard", "--log-level", args.log_level])
    if rest:
        build_parser().error(f"unrecognized arguments: {' '.join(rest)}")
    from robot.cli import setup_logging
    from robot.interface.bus import ZmqLink
    setup_logging(args.log_level)
    cfg = load_config(args.config_dir)
    link = ZmqLink(cfg, args.connect, source="dashboard")
    if not link.wait_connected(3.0):
        log.warning("no telemetry from %s yet; is robot.run running? (the page will show 'disconnected')",
                    link.tel_addr)
    import uvicorn
    host = args.host or str(cfg.get("network.dashboard.host", "0.0.0.0"))
    port = args.port or int(cfg.get("network.dashboard.port", 8000))
    log.info("dashboard: http://localhost:%d", port)
    try:
        uvicorn.run(create_app(link, cfg), host=host, port=port, log_level="warning")
    finally:
        link.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
