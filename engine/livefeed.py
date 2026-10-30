"""Real-time trade stream from Finnhub (free key at https://finnhub.io).

A background thread keeps a websocket open, receives every US trade for the subscribed symbols
and remembers the latest price and trade time per symbol. It reconnects on its own.
"""
import json
import threading
import time

import websocket

URL = "wss://ws.finnhub.io?token={key}"


class FinnhubStream:
    def __init__(self, api_key: str):
        self._key = api_key
        self._lock = threading.Lock()
        self._latest: dict[str, tuple[float, float]] = {}   # symbol -> (price, unix time)
        self._symbols: set[str] = set()
        self._ws = None
        self.status = "connecting"
        self.error = ""
        self.trades_received = 0
        threading.Thread(target=self._run, daemon=True, name="finnhub-stream").start()

    # ---- public ------------------------------------------------------------------
    def subscribe(self, symbols) -> None:
        new = set(symbols) - self._symbols
        self._symbols |= new
        if self.status == "live":
            for s in new:
                self._send({"type": "subscribe", "symbol": s})

    def latest(self, symbol: str) -> tuple[float, float] | None:
        with self._lock:
            return self._latest.get(symbol)

    # ---- internals ---------------------------------------------------------------
    def _send(self, msg: dict) -> None:
        try:
            self._ws.send(json.dumps(msg))
        except Exception as e:  # socket closed mid-send; the reconnect loop resubscribes
            self.error = str(e)

    def _on_open(self, ws):
        self.status = "live"
        self.error = ""
        self._backoff = 2
        for s in sorted(self._symbols):
            self._send({"type": "subscribe", "symbol": s})

    def _on_message(self, ws, raw):
        try:
            msg = json.loads(raw)
        except ValueError:
            return
        if msg.get("type") == "trade":
            with self._lock:
                for tr in msg.get("data") or []:
                    sym, price, ts = tr.get("s"), tr.get("p"), (tr.get("t") or 0) / 1000
                    if not sym or not price:
                        continue
                    cur = self._latest.get(sym)
                    if cur is None or ts >= cur[1]:
                        self._latest[sym] = (float(price), ts)
                    self.trades_received += 1
        elif msg.get("type") == "error":
            self.error = msg.get("msg", "unknown error")

    def _on_error(self, ws, err):
        self.error = str(err)

    def _on_close(self, ws, code, reason):
        self.status = "reconnecting"
        if reason:
            self.error = str(reason)

    def _run(self):
        self._backoff = 2
        while True:
            self._ws = websocket.WebSocketApp(URL.format(key=self._key), on_open=self._on_open,
                                              on_message=self._on_message, on_error=self._on_error,
                                              on_close=self._on_close)
            self._ws.run_forever(ping_interval=30, ping_timeout=10)
            self.status = "reconnecting"
            time.sleep(self._backoff)
            self._backoff = min(self._backoff * 2, 60)
