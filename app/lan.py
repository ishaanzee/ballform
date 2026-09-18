from __future__ import annotations

import argparse
import ipaddress
import json
import os
import secrets
import shutil
import socket
import subprocess
from dataclasses import dataclass

import qrcode
import uvicorn


@dataclass(frozen=True)
class ShareTransport:
    name: str
    address: str
    description: str


def local_ip() -> str:
    """Return the interface address macOS would use for outbound traffic."""
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        # UDP connect selects an interface without sending application data.
        sock.connect(("8.8.8.8", 80))
        return str(sock.getsockname()[0])
    except OSError:
        try:
            return socket.gethostbyname(socket.gethostname())
        except OSError:
            return "127.0.0.1"
    finally:
        sock.close()


def _tailscale_command() -> str | None:
    discovered = shutil.which("tailscale")
    if discovered:
        return discovered
    for candidate in (
        "/Applications/Tailscale.app/Contents/MacOS/Tailscale",
        "/Applications/Tailscale.app/Contents/MacOS/tailscale",
    ):
        if os.path.isfile(candidate) and os.access(candidate, os.X_OK):
            return candidate
    return None


def tailscale_ip() -> tuple[str | None, str]:
    """Return an active Tailscale IPv4 and a human-readable failure reason."""
    command = _tailscale_command()
    if not command:
        return None, "Tailscale is not installed on this Mac."
    try:
        completed = subprocess.run(
            [command, "status", "--json"], capture_output=True, text=True, timeout=5
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return None, f"Could not read Tailscale status: {exc}"
    if completed.returncode != 0:
        detail = completed.stderr.strip() or "Tailscale status failed."
        return None, detail
    try:
        status = json.loads(completed.stdout)
    except json.JSONDecodeError:
        return None, "Tailscale returned an unreadable status response."
    if status.get("BackendState") != "Running":
        return None, f"Tailscale is {str(status.get('BackendState') or 'not connected').lower()}."
    candidates = status.get("TailscaleIPs") or status.get("Self", {}).get("TailscaleIPs") or []
    for candidate in candidates:
        try:
            address = ipaddress.ip_address(candidate)
        except ValueError:
            continue
        if address.version == 4:
            return str(address), "connected"
    return None, "Tailscale is running but has no IPv4 address yet."


def choose_transport(network: str) -> tuple[ShareTransport, str | None]:
    tail_ip, reason = tailscale_ip()
    if network == "tailscale":
        if not tail_ip:
            raise RuntimeError(
                f"{reason} Open Tailscale on the Mac and iPhone, sign into the same account, "
                "and connect both devices before trying again."
            )
        return ShareTransport("tailscale", tail_ip, "encrypted Tailscale connection"), None
    if network == "auto" and tail_ip:
        return ShareTransport("tailscale", tail_ip, "encrypted Tailscale connection"), None
    warning = reason if network == "auto" and _tailscale_command() else None
    return ShareTransport("lan", local_ip(), "local Wi-Fi connection"), warning


def print_qr(url: str) -> None:
    qr = qrcode.QRCode(border=1)
    qr.add_data(url)
    qr.make(fit=True)
    qr.print_ascii(invert=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Share Ballform privately with a phone")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument(
        "--network", choices=("auto", "tailscale", "lan"), default="auto",
        help="connection to advertise (default: prefer Tailscale, otherwise local Wi-Fi)",
    )
    parser.add_argument("--no-qr", action="store_true", help="Do not print the terminal QR code")
    args = parser.parse_args()
    if not 1024 <= args.port <= 65535:
        parser.error("port must be between 1024 and 65535")

    token = secrets.token_urlsafe(18)
    os.environ["BALLFORM_ACCESS_TOKEN"] = token
    try:
        transport, warning = choose_transport(args.network)
    except RuntimeError as exc:
        parser.error(str(exc))
    url = f"http://{transport.address}:{args.port}/?token={token}&transport={transport.name}"
    print("\nBALLFORM PHONE UPLOAD")
    print(f"Using {transport.description} at {transport.address}.")
    if warning:
        print(f"Warning: {warning} Falling back to local Wi-Fi.")
    if transport.name == "tailscale":
        print("Keep Tailscale connected on both devices and leave this terminal open.\n")
    else:
        print("Keep this terminal open and connect the phone to the same Wi-Fi.\n")
    if not args.no_qr:
        print_qr(url)
    print(f"\nOpen or scan:\n{url}\n")
    print("Only devices with this temporary pairing link can access jobs.")
    print("Press Ctrl+C to stop phone access.\n")
    uvicorn.run("app.main:app", host="0.0.0.0", port=args.port, log_level="info")


if __name__ == "__main__":
    main()
