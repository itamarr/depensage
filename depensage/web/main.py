"""
Entry point for the DepenSage web server.

Usage:
    python -m depensage.web.main [--host HOST] [--port PORT] [--no-ssl]
"""

import argparse
import logging
import os
import sys
from datetime import datetime

import uvicorn


def _setup_logging():
    """Configure logging to both console and dated log file."""
    log_dir = os.path.join(os.path.dirname(__file__), "..", "..", "logs")
    log_dir = os.path.abspath(log_dir)
    os.makedirs(log_dir, exist_ok=True)

    date_str = datetime.now().strftime("%Y-%m-%d")
    log_file = os.path.join(log_dir, f"depensage_{date_str}.log")

    # Root logger: file + console
    root = logging.getLogger()
    root.setLevel(logging.INFO)

    fmt = logging.Formatter(
        "%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    # File handler (append, one file per day)
    fh = logging.FileHandler(log_file, encoding="utf-8")
    fh.setLevel(logging.INFO)
    fh.setFormatter(fmt)
    root.addHandler(fh)

    # Console handler
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(fmt)
    root.addHandler(ch)

    print(f"Logging to {log_file}")
    return log_file


def main():
    parser = argparse.ArgumentParser(description="DepenSage Web Server")
    parser.add_argument("--host", default="0.0.0.0", help="Bind host")
    parser.add_argument("--port", type=int, default=8000, help="Bind port")
    parser.add_argument("--no-ssl", action="store_true", help="Disable HTTPS")
    args = parser.parse_args()

    _setup_logging()

    ssl_kwargs = {}
    if not args.no_ssl:
        cert_dir = os.path.abspath(".secrets")
        cert_file = os.path.join(cert_dir, "cert.pem")
        key_file = os.path.join(cert_dir, "key.pem")

        if os.path.exists(cert_file) and os.path.exists(key_file):
            ssl_kwargs["ssl_certfile"] = cert_file
            ssl_kwargs["ssl_keyfile"] = key_file
            protocol = "https"
        else:
            print(
                f"SSL certificates not found at {cert_dir}/cert.pem and key.pem\n"
                f"Generate with:\n"
                f"  openssl req -x509 -newkey rsa:2048 -keyout {key_file} "
                f"-out {cert_file} -days 365 -nodes "
                f'-subj "/CN=depensage"\n'
                f"Or run with --no-ssl for HTTP (not recommended)."
            )
            sys.exit(1)
    else:
        protocol = "http"

    print(f"Starting DepenSage at {protocol}://{args.host}:{args.port}")
    if args.host == "0.0.0.0":
        print("Accessible from any device on the local network")

    uvicorn.run(
        "depensage.web.app:create_app",
        factory=True,
        host=args.host,
        port=args.port,
        log_level="info",
        **ssl_kwargs,
    )


if __name__ == "__main__":
    main()
