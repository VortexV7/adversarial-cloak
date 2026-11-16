import { useCallback, useEffect, useRef, useState } from "react";
import type { FrameMessage } from "../lib";
import { WS_URL } from "../lib";

/** WebSocket hook: sends webcam JPEG frames, receives protected frames + metrics. */
export function useCloakSocket() {
  const wsRef = useRef<WebSocket | null>(null);
  const [connected, setConnected] = useState(false);
  const [frame, setFrame] = useState<FrameMessage | null>(null);
  const [error, setError] = useState<string | null>(null);

  const connect = useCallback(() => {
    if (wsRef.current?.readyState === WebSocket.OPEN) return;
    const ws = new WebSocket(WS_URL);
    wsRef.current = ws;
    ws.onopen = () => { setConnected(true); setError(null); };
    ws.onclose = () => setConnected(false);
    ws.onerror = () => setError("Could not reach the backend at " + WS_URL + ". Is `make dev` running?");
    ws.onmessage = (ev) => {
      try {
        const msg = JSON.parse(ev.data);
        if (msg.type === "frame") setFrame(msg as FrameMessage);
        else if (msg.type === "error") setError(msg.message);
      } catch { /* ignore non-JSON frames */ }
    };
  }, []);

  const disconnect = useCallback(() => { wsRef.current?.close(); wsRef.current = null; }, []);

  const sendFrame = useCallback((jpeg: Blob) => {
    if (wsRef.current?.readyState === WebSocket.OPEN) wsRef.current.send(jpeg);
  }, []);

  useEffect(() => () => wsRef.current?.close(), []);
  return { connected, frame, error, connect, disconnect, sendFrame, clearError: () => setError(null) };
}
