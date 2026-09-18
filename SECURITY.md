# Security policy

## Reporting a vulnerability

Do not open a public issue for a vulnerability involving authentication bypass, arbitrary file access, command execution, or disclosure of uploaded footage. Use GitHub's private vulnerability reporting feature for the repository. If that feature is unavailable, contact the repository owner privately.

## Deployment boundary

Ballform is designed for a single trusted operator. The temporary QR token protects job APIs from other devices, but it is not a multi-user identity system. Do not expose the local server directly to the public internet.

The recommended remote transport is Tailscale. Ordinary LAN mode uses HTTP without application-layer encryption and should only be used on a trusted private network. Stop the sharing command with `Ctrl+C` when analysis is finished.

Uploaded videos may contain biometric and location-revealing information. They remain in `data/jobs/` until manually removed. Do not sync that directory to a public repository or shared cloud folder.
