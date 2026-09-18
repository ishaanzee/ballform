# Security policy

## Reporting a vulnerability

Do not open a public issue for a vulnerability involving authentication bypass, arbitrary file access, command execution, or disclosure of uploaded footage. Use GitHub's private vulnerability reporting feature for the repository. If that feature is unavailable, contact the repository owner privately.

## Deployment boundary

Ballform is designed for a trusted individual or team. Its shared access token is not a multi-user identity system: token holders can access all jobs. Never expose an unauthenticated local server to the public internet.

For the Vercel frontend, host the Python service behind HTTPS, set `BALLFORM_REQUIRE_TOKEN=1` and a private `BALLFORM_ACCESS_TOKEN`, and allow only your exact frontend origins in `BALLFORM_CORS_ORIGINS`. The Docker image requires a token at startup. CORS is not authentication. The public frontend environment contains only the backend address, never the token. Users enter the token in the interface; it stays in session storage. Video playback includes the token in its URL, so backend/proxy logs must omit query strings (the Docker worker disables Uvicorn access logs).

The recommended remote transport is Tailscale. Ordinary LAN mode uses HTTP without application-layer encryption and should only be used on a trusted private network. Stop the sharing command with `Ctrl+C` when analysis is finished.

Uploaded videos may contain biometric and location-revealing information. In local mode they remain in `data/jobs/`; in hosted mode they are sent directly to the configured Python service and remain in `BALLFORM_JOBS_DIR` until manually removed. Do not sync that directory to a public repository or shared cloud folder. Add per-user authentication, job ownership, quotas, and retention controls before offering public multi-tenant access.
