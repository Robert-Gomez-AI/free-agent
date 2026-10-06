"""Cybersecurity tools — defensive analysis, threat intel and triage.

Everything here is passive or single-target inspection (CVE intel, hashing,
decoding, header/TLS audits of a given URL). No scanning or exploitation.
"""
from __future__ import annotations

import base64
import binascii
import codecs
import hashlib
import ipaddress
import json
import math
import re
import socket
import ssl
import time
import urllib.parse
import urllib.request

from langchain_core.tools import tool

from free_agent.tools.domains._common import http_json, net_error, resolve_path, truncate


@tool
def cve_lookup(cve_id: str) -> str:
    """Fetch a CVE from the NIST NVD: description, CVSS score/vector, CWE, dates and references.

    Args:
      cve_id: e.g. `CVE-2021-44228`.
    """
    cve = cve_id.strip().upper()
    if not re.fullmatch(r"CVE-\d{4}-\d{4,}", cve):
        return f"[{cve_id!r} is not a valid CVE id (CVE-YYYY-NNNN)]"
    try:
        data = http_json("https://services.nvd.nist.gov/rest/json/cves/2.0", {"cveId": cve}, timeout=30)
    except Exception as exc:
        return net_error(exc)
    vulns = data.get("vulnerabilities") or []
    if not vulns:
        return f"[{cve} not found in NVD]"
    c = vulns[0]["cve"]
    desc = next((d["value"] for d in c.get("descriptions", []) if d.get("lang") == "en"), "")
    lines = [f"{cve} — status: {c.get('vulnStatus', '?')}",
             f"published {c.get('published', '')[:10]} · modified {c.get('lastModified', '')[:10]}",
             f"description: {desc}"]
    metrics = c.get("metrics", {})
    for key in ("cvssMetricV40", "cvssMetricV31", "cvssMetricV30", "cvssMetricV2"):
        if metrics.get(key):
            cd = metrics[key][0]["cvssData"]
            sev = cd.get("baseSeverity") or metrics[key][0].get("baseSeverity", "")
            lines.append(f"CVSS {cd.get('version')}: {cd.get('baseScore')} {sev} — {cd.get('vectorString')}")
            break
    cwes = {d["value"] for w in c.get("weaknesses", []) for d in w.get("description", [])}
    if cwes:
        lines.append(f"weaknesses: {', '.join(sorted(cwes))}")
    if c.get("cisaExploitAdd"):
        lines.append(f"CISA KEV: known exploited (added {c['cisaExploitAdd']}) — action: {c.get('cisaRequiredAction', '')}")
    refs = [r["url"] for r in c.get("references", [])][:8]
    if refs:
        lines.append("references:\n" + "\n".join(f"  - {u}" for u in refs))
    return truncate("\n".join(lines))


@tool
def hash_data(text: str = "", file_path: str = "") -> str:
    """Compute MD5, SHA-1, SHA-256 and SHA-512 of a string or a file (for IOC matching / integrity).

    Args:
      text: String to hash (UTF-8). Ignored when file_path is given.
      file_path: Path of a file to hash.
    """
    algos = ("md5", "sha1", "sha256", "sha512")
    hashers = {a: hashlib.new(a) for a in algos}
    if file_path:
        try:
            with resolve_path(file_path).open("rb") as fh:
                for chunk in iter(lambda: fh.read(1 << 20), b""):
                    for h in hashers.values():
                        h.update(chunk)
        except OSError as exc:
            return f"[cannot read {file_path}: {exc}]"
    else:
        for h in hashers.values():
            h.update(text.encode("utf-8"))
    return "\n".join(f"{a}: {h.hexdigest()}" for a, h in hashers.items())


_CODECS = ("base64_encode", "base64_decode", "hex_encode", "hex_decode", "url_encode",
           "url_decode", "rot13", "base32_decode", "base64url_decode")


@tool
def codec_transform(data: str, operation: str) -> str:
    """Encode/decode data for payload and artifact analysis.

    Args:
      data: Input string.
      operation: One of base64_encode, base64_decode, base64url_decode, base32_decode,
                 hex_encode, hex_decode, url_encode, url_decode, rot13.
    """
    op = operation.strip().lower()
    try:
        if op == "base64_encode":
            return base64.b64encode(data.encode()).decode()
        if op in ("base64_decode", "base64url_decode"):
            raw = data.strip()
            raw += "=" * (-len(raw) % 4)
            out = base64.urlsafe_b64decode(raw) if op == "base64url_decode" else base64.b64decode(raw)
        elif op == "base32_decode":
            out = base64.b32decode(data.strip().upper())
        elif op == "hex_encode":
            return data.encode().hex()
        elif op == "hex_decode":
            out = bytes.fromhex(re.sub(r"[^0-9a-fA-F]", "", data))
        elif op == "url_encode":
            return urllib.parse.quote(data, safe="")
        elif op == "url_decode":
            return urllib.parse.unquote(data)
        elif op == "rot13":
            return codecs.encode(data, "rot13")
        else:
            return f"[unknown operation {operation!r} — use one of {', '.join(_CODECS)}]"
    except (binascii.Error, ValueError) as exc:
        return f"[decode failed: {exc}]"
    try:
        return out.decode("utf-8")
    except UnicodeDecodeError:
        return f"[binary output, hex] {out.hex()}"


@tool
def jwt_inspect(token: str) -> str:
    """Decode a JWT WITHOUT verifying it and flag risky claims (alg=none, missing/expired exp, …).

    Args:
      token: The compact JWT (`header.payload.signature`).
    """
    parts = token.strip().split(".")
    if len(parts) < 2:
        return "[not a JWT — expected header.payload.signature]"

    def dec(seg: str) -> dict:
        seg += "=" * (-len(seg) % 4)
        return json.loads(base64.urlsafe_b64decode(seg))

    try:
        header, payload = dec(parts[0]), dec(parts[1])
    except (ValueError, binascii.Error) as exc:
        return f"[could not decode: {exc}]"
    findings = []
    alg = str(header.get("alg", "")).lower()
    if alg in ("none", ""):
        findings.append("CRITICAL: alg=none — signature not enforced")
    elif alg.startswith("hs"):
        findings.append("info: HMAC (shared secret) — check secret strength; beware RS/HS confusion")
    if "jku" in header or "x5u" in header:
        findings.append("warn: header references remote keys (jku/x5u) — validate allowlist")
    if "kid" in header and re.search(r"[./\\]|'|--", str(header["kid"])):
        findings.append("warn: suspicious kid value (path traversal / injection?)")
    now = time.time()
    if "exp" not in payload:
        findings.append("warn: no exp claim — token never expires")
    elif isinstance(payload["exp"], (int, float)) and payload["exp"] < now:
        findings.append(f"info: expired at {time.strftime('%Y-%m-%d %H:%M:%S', time.gmtime(payload['exp']))} UTC")
    elif isinstance(payload["exp"], (int, float)) and payload["exp"] - now > 86400 * 30:
        findings.append("warn: lifetime > 30 days")
    if len(parts) < 3 or not parts[2]:
        findings.append("warn: empty signature segment")
    return (
        f"header: {json.dumps(header, indent=2)}\npayload: {json.dumps(payload, indent=2)}\n"
        f"findings:\n" + "\n".join(f"  - {f}" for f in findings or ["none"])
    )


_SEC_HEADERS = {
    "strict-transport-security": "HSTS — forces HTTPS",
    "content-security-policy": "CSP — mitigates XSS/injection",
    "x-content-type-options": "should be `nosniff`",
    "x-frame-options": "clickjacking protection (or CSP frame-ancestors)",
    "referrer-policy": "limits referrer leakage",
    "permissions-policy": "restricts browser features",
    "cross-origin-opener-policy": "process isolation",
}


@tool
def http_security_headers(url: str) -> str:
    """Audit a URL's HTTP response security headers, cookies flags and server disclosure.

    Only use on systems the user owns or is authorized to assess.

    Args:
      url: Full URL, e.g. `https://example.com`.
    """
    if not url.startswith(("http://", "https://")):
        url = "https://" + url
    req = urllib.request.Request(url, method="GET", headers={"User-Agent": "free-agent-header-audit"})
    try:
        with urllib.request.urlopen(req, timeout=20) as resp:
            status, headers = resp.status, resp.headers
    except urllib.error.HTTPError as exc:
        status, headers = exc.code, exc.headers
    except Exception as exc:
        return net_error(exc)
    lower = {k.lower(): v for k, v in headers.items()}
    lines = [f"{url} → HTTP {status}", "present:"]
    missing = []
    for h, why in _SEC_HEADERS.items():
        if h in lower:
            lines.append(f"  ✓ {h}: {lower[h][:160]}")
        else:
            missing.append(f"  ✗ {h} — {why}")
    lines += ["missing:", *(missing or ["  (none)"])]
    leaks = [f"{h}: {lower[h]}" for h in ("server", "x-powered-by", "x-aspnet-version") if h in lower]
    if leaks:
        lines.append("disclosure: " + "; ".join(leaks))
    for cookie in headers.get_all("Set-Cookie") or []:
        flags = cookie.lower()
        bad = [f for f in ("secure", "httponly", "samesite") if f not in flags]
        name = cookie.split("=", 1)[0]
        lines.append(f"cookie {name}: " + (f"missing {', '.join(bad)}" if bad else "flags ok"))
    return "\n".join(lines)


@tool
def tls_certificate_info(host: str, port: int = 443) -> str:
    """Inspect a server's TLS certificate and negotiated protocol (issuer, SANs, validity, cipher).

    Args:
      host: Hostname, e.g. `example.com`.
      port: TLS port (default 443).
    """
    host = host.strip().removeprefix("https://").split("/")[0]
    ctx = ssl.create_default_context()
    try:
        with socket.create_connection((host, int(port)), timeout=15) as sock, \
                ctx.wrap_socket(sock, server_hostname=host) as tls:
            cert = tls.getpeercert()
            proto, cipher = tls.version(), tls.cipher()
    except ssl.SSLCertVerificationError as exc:
        return f"[certificate verification FAILED: {exc.verify_message}]"
    except OSError as exc:
        return f"[connection failed: {exc}]"
    subject = dict(x[0] for x in cert.get("subject", ()))
    issuer = dict(x[0] for x in cert.get("issuer", ()))
    sans = [v for k, v in cert.get("subjectAltName", ()) if k == "DNS"]
    not_after = ssl.cert_time_to_seconds(cert["notAfter"])
    days_left = int((not_after - time.time()) // 86400)
    return (
        f"subject: {subject.get('commonName', '')}\nissuer: {issuer.get('organizationName', '')} "
        f"({issuer.get('commonName', '')})\nvalid: {cert.get('notBefore')} → {cert.get('notAfter')} "
        f"({days_left} days left{' — EXPIRING SOON' if days_left < 30 else ''})\n"
        f"SANs: {', '.join(sans[:20])}\nprotocol: {proto}\ncipher: {cipher[0] if cipher else '?'}"
    )


@tool
def dns_lookup(hostname: str) -> str:
    """Resolve a hostname to its IPv4/IPv6 addresses and reverse-DNS names.

    Args:
      hostname: Domain to resolve.
    """
    try:
        infos = socket.getaddrinfo(hostname.strip(), None)
    except socket.gaierror as exc:
        return f"[resolution failed: {exc}]"
    addrs = sorted({i[4][0] for i in infos})
    lines = []
    for a in addrs:
        try:
            rev = socket.gethostbyaddr(a)[0]
        except OSError:
            rev = "-"
        kind = "private" if ipaddress.ip_address(a).is_private else "public"
        lines.append(f"{a}  ({kind}, ptr: {rev})")
    return "\n".join(lines)


_IOC_PATTERNS = {
    "ipv4": r"\b(?:(?:25[0-5]|2[0-4]\d|1?\d?\d)\.){3}(?:25[0-5]|2[0-4]\d|1?\d?\d)\b",
    "url": r"\bhxxps?://[^\s\"'<>]+|\bhttps?://[^\s\"'<>]+",
    "domain": r"\b(?:[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?(?:\.|\[\.\]))+(?:com|net|org|io|ru|cn|info|biz|xyz|top|gov|edu|co|me|us|uk|de|onion)\b",
    "email": r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b",
    "md5": r"\b[a-f0-9]{32}\b",
    "sha1": r"\b[a-f0-9]{40}\b",
    "sha256": r"\b[a-f0-9]{64}\b",
    "cve": r"\bCVE-\d{4}-\d{4,}\b",
}


@tool
def extract_iocs(text: str) -> str:
    """Extract indicators of compromise (IPs, URLs, domains, emails, hashes, CVEs) from logs or reports.

    Handles common defanging like `hxxp://` and `example[.]com`.

    Args:
      text: Raw text — log lines, email, threat report.
    """
    found: dict[str, list[str]] = {}
    for kind, pat in _IOC_PATTERNS.items():
        hits = sorted({m.group(0) for m in re.finditer(pat, text, re.IGNORECASE)})
        if kind == "domain":
            hits = [h for h in hits if not re.fullmatch(_IOC_PATTERNS["ipv4"], h)]
        if hits:
            found[kind] = hits
    if not found:
        return "[no IOCs found]"
    return truncate("\n".join(f"{k} ({len(v)}):\n  " + "\n  ".join(v[:100]) for k, v in found.items()))


@tool
def password_strength(password: str) -> str:
    """Estimate password entropy and flag weaknesses (length, charset, patterns). Nothing is stored or sent.

    Args:
      password: The password to evaluate.
    """
    pools = [(r"[a-z]", 26), (r"[A-Z]", 26), (r"\d", 10), (r"[^a-zA-Z0-9]", 33)]
    pool = sum(size for pat, size in pools if re.search(pat, password))
    bits = len(password) * math.log2(pool) if pool else 0.0
    issues = []
    if len(password) < 12:
        issues.append("shorter than 12 chars")
    if re.search(r"(.)\1{2,}", password):
        issues.append("repeated characters")
    if re.search(r"(?:0123|1234|2345|3456|4567|5678|6789|abcd|qwer|asdf)", password.lower()):
        issues.append("sequential pattern")
    if password.lower() in {"password", "123456", "qwerty", "letmein", "admin", "welcome", "iloveyou"}:
        issues.append("in common-password list")
    rating = "weak" if bits < 50 or issues else "moderate" if bits < 80 else "strong"
    return f"length={len(password)} charset≈{pool} entropy≈{bits:.0f} bits → {rating}\nissues: {', '.join(issues) or 'none'}"


SECURITY_TOOLS = [
    cve_lookup, hash_data, codec_transform, jwt_inspect, http_security_headers,
    tls_certificate_info, dns_lookup, extract_iocs, password_strength,
]
