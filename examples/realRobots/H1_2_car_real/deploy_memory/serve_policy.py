"""Websocket server for the bundled car policy (deploy_infer.CarPolicy), on the starVLA websocket
protocol (deployment/model_server/tools: msgpack-numpy frames, metadata at the handshake). The memory lives in the
server: one robot / episode stream at a time.

    python serve_policy.py [--port 10093] [--update_interval_s 1.0]

Client (deployment/model_server/tools/websocket_policy_client.py):
    client = WebsocketClientPolicy(host, 10093)
    r = client.predict_action({"type": "infer", "payload": {
            "reset": True,                       # first call of every episode (with its instruction)
            "instruction": "Activate the left turn signal and inspect its operation.",
            "images": [head_rgb, left_wrist_rgb, right_wrist_rgb],   # uint8 (H, W, 3), this order
            "state63": state63,                  # recorded observation.state layout (or "state42")
            "t": 0.0}})                          # seconds since the episode start
    r["data"]["actions"]   # (50, 42) absolute @ 30 Hz; r["data"]["progress"], ["memory_updated"], ["latency_ms"]
"""
import argparse
import logging

from deploy_infer import CAMERAS, IMAGE_SIZE, CarPolicy


class _Stream:
    """predict_action(**payload) for WebsocketPolicyServer: reset + one streamed step."""

    def __init__(self, policy):
        self.policy = policy

    def predict_action(self, images, t, state63=None, state42=None, instruction=None, reset=False, **_):
        if reset:
            self.policy.reset(instruction)
        elif instruction is not None and instruction != self.policy.instruction:
            raise ValueError("instruction changed without reset=True (the memory belongs to one episode)")
        return self.policy.predict(images, state63=state63, t=float(t), state42=state42)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=10093)
    ap.add_argument("--ckpt", default=None, help="default: the bundle's model/checkpoints/*.pt")
    ap.add_argument("--update_interval_s", type=float, default=1.0)
    ap.add_argument("--idle_timeout", type=int, default=-1)
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, force=True)

    from deployment.model_server.tools.websocket_policy_server import WebsocketPolicyServer

    policy = CarPolicy(ckpt=args.ckpt, update_interval_s=args.update_interval_s)
    meta = {"env": "starvla_car_memory_policy", "cameras": list(CAMERAS), "image_size": list(IMAGE_SIZE),
            "action_chunk_size": policy.horizon, "action_rate_hz": 30, "state_keys": policy.proc.state_keys,
            "action_keys": policy.proc.action_keys, "has_memory": policy.has_memory,
            "memory_update_interval_s": policy.update_interval_s if policy.has_memory else None,
            "tasks": policy.tasks}
    logging.info("serving on :%d  %s", args.port, meta)
    WebsocketPolicyServer(policy=_Stream(policy), host="0.0.0.0", port=args.port, idle_timeout=args.idle_timeout,
                          metadata=meta).serve_forever()


if __name__ == "__main__":
    main()
