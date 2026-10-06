from __future__ import annotations

import json
import logging
import os
import stat
from pathlib import Path
from typing import Any, Literal

from pydantic import Field, SecretStr, model_validator
from pydantic_settings import BaseSettings, PydanticBaseSettingsSource, SettingsConfigDict

Provider = Literal["fireworks", "ollama", "anthropic"]
PROVIDERS: tuple[str, ...] = ("fireworks", "ollama", "anthropic")

FIREWORKS_MODEL_PREFIX = "accounts/fireworks/models/"
DEFAULT_FIREWORKS_MODEL = FIREWORKS_MODEL_PREFIX + "kimi-k3"

log = logging.getLogger(__name__)

# ─── persistence locations ──────────────────────────────────────────────────
#
# Two files split on sensitivity:
#   settings.json — non-secret user preferences (provider, models, temp, …)
#   secrets.json  — API keys; written with mode 0600
#
# Both live under ~/.config/free-agent/ to be siblings of state.json and
# the workspaces/ directory the rest of the app already uses.

CONFIG_ROOT = Path.home() / ".config" / "free-agent"
SETTINGS_FILE = CONFIG_ROOT / "settings.json"
SECRETS_FILE = CONFIG_ROOT / "secrets.json"

# Field names that belong in secrets.json instead of settings.json. The
# settings panel and persistence helpers route them accordingly.
_SECRET_FIELDS = frozenset({"anthropic_api_key", "fireworks_api_key"})


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except (OSError, json.JSONDecodeError) as exc:
        log.warning("could not read %s: %s", path, exc)
        return {}


def _atomic_write_json(path: Path, data: dict[str, Any], *, mode: int | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")
    if mode is not None:
        try:
            os.chmod(tmp, mode)
        except OSError as exc:
            log.warning("could not chmod %s: %s", tmp, exc)
    os.replace(tmp, path)


class _JsonFileSource(PydanticBaseSettingsSource):
    """Pydantic settings source that reads a JSON file as field values.

    Lower priority than env/dotenv so an ANTHROPIC_API_KEY in the shell
    still wins over a stale value persisted to disk.
    """

    def __init__(self, settings_cls: type[BaseSettings], path_getter: Any) -> None:
        super().__init__(settings_cls)
        # `path_getter` is a zero-arg callable that resolves the path at the
        # moment the source is consumed — keeps tests that monkeypatch the
        # module-level constants honest, and avoids stale paths if the user
        # ever moves the config root mid-process.
        self._data = _read_json(path_getter())

    def get_field_value(self, field: Any, field_name: str) -> tuple[Any, str, bool]:
        if field_name in self._data:
            return self._data[field_name], field_name, False
        return None, field_name, False

    def __call__(self) -> dict[str, Any]:
        # IMPORTANT: emit each value under the field's ALIAS, not its name.
        # The dotenv / env sources also emit by alias (e.g. FREE_AGENT_PROVIDER),
        # so using the same key lets `deep_update` collapse duplicates and the
        # source-priority order alone decides the winner. If we emitted field
        # names here, the final state would carry both `provider` and
        # `FREE_AGENT_PROVIDER` and pydantic would silently prefer the alias
        # entry — letting a stale .env override a freshly-saved settings.json.
        out: dict[str, Any] = {}
        for name, field in self.settings_cls.model_fields.items():
            value, key, is_complex = self.get_field_value(field, name)
            if value is None:
                continue
            prepared = self.prepare_field_value(name, field, value, is_complex)
            out[field.alias or name] = prepared
        return out


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        # JSON files persist values keyed by field name, not env-var alias —
        # without this flag pydantic would only honor the alias key and
        # silently fall back to defaults for the field-name keys our
        # _JsonFileSource emits.
        populate_by_name=True,
    )

    # Fireworks is the standard provider: hosted open-weight models with
    # tool calling, no local GPU needed. Ollama and Anthropic stay available.
    provider: Provider = Field("fireworks", alias="FREE_AGENT_PROVIDER")

    fireworks_model: str = Field(DEFAULT_FIREWORKS_MODEL, alias="FREE_AGENT_FIREWORKS_MODEL")
    fireworks_api_key: SecretStr | None = Field(None, alias="FIREWORKS_API_KEY")
    fireworks_base_url: str = Field(
        "https://api.fireworks.ai/inference", alias="FREE_AGENT_FIREWORKS_BASE_URL"
    )

    ollama_model: str = Field("qwen2.5:7b", alias="FREE_AGENT_OLLAMA_MODEL")
    ollama_base_url: str = Field("http://localhost:11434", alias="FREE_AGENT_OLLAMA_BASE_URL")

    anthropic_model: str = Field("claude-sonnet-4-6", alias="FREE_AGENT_ANTHROPIC_MODEL")
    anthropic_api_key: SecretStr | None = Field(None, alias="ANTHROPIC_API_KEY")

    temperature: float = Field(0.7, alias="FREE_AGENT_TEMPERATURE")
    max_tokens: int = Field(4096, alias="FREE_AGENT_MAX_TOKENS")
    log_level: str = Field("WARNING", alias="FREE_AGENT_LOG_LEVEL")
    history_file: Path = Field(
        default_factory=lambda: Path.home() / ".free_agent_history",
        alias="FREE_AGENT_HISTORY_FILE",
    )

    # When True, the agent's read_file/write_file/ls/edit_file tools operate on
    # the real filesystem rooted at cwd (with virtual_mode guardrails — paths
    # cannot escape via .. / ~ / absolute outside-root). Defaults to False, in
    # which case files live in the in-memory deepagents StateBackend.
    writable: bool = Field(False, alias="FREE_AGENT_WRITABLE")

    # Preset mode (see free_agent.modes): "" = plain agent, or one of
    # code / science / security / finance.
    mode: str = Field("", alias="FREE_AGENT_MODE")

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        # Priority (high → low):
        #   1. explicit init kwargs        — code in this process wants this exact value
        #   2. real env vars (os.environ)  — `EXPORT FOO=...; free-agent` is an explicit override
        #   3. ~/.config/secrets.json      — saved API keys
        #   4. ~/.config/settings.json     — saved user prefs (/settings, /model use)
        #   5. .env in cwd                 — directory-local DEFAULTS, beats only hardcoded code defaults
        #   6. /run/secrets                — docker secret style fallback
        #   7. field defaults
        #
        # Why settings.json beats .env: a click-to-save in /settings or
        # /model use should never be silently undone by a stale .env that's
        # been sitting in the repo since first checkout. Real env vars (which
        # the user must export each time) still win — that's the explicit
        # one-off override path.
        return (
            init_settings,
            env_settings,
            _JsonFileSource(settings_cls, secrets_file_path),
            _JsonFileSource(settings_cls, settings_file_path),
            dotenv_settings,
            file_secret_settings,
        )

    @property
    def active_model(self) -> str:
        if self.provider == "fireworks":
            return self.fireworks_model
        return self.ollama_model if self.provider == "ollama" else self.anthropic_model

    @model_validator(mode="after")
    def _check_provider_credentials(self) -> Settings:
        self.fireworks_model = normalize_fireworks_model(self.fireworks_model)
        if self.provider == "fireworks":
            key = self.fireworks_api_key
            if key is None or not key.get_secret_value().strip():
                raise ValueError(
                    "FIREWORKS_API_KEY is required when FREE_AGENT_PROVIDER=fireworks "
                    "(the default). Get one at https://fireworks.ai/account/api-keys and set "
                    f"it via /settings, the FIREWORKS_API_KEY env var / .env, or {SECRETS_FILE}. "
                    "To run fully local instead, set FREE_AGENT_PROVIDER=ollama."
                )
        if self.provider == "anthropic":
            key = self.anthropic_api_key
            if key is None or not key.get_secret_value().strip():
                raise ValueError(
                    "ANTHROPIC_API_KEY is required when FREE_AGENT_PROVIDER=anthropic. "
                    "Set it via /settings, the ANTHROPIC_API_KEY env var, or "
                    f"{SECRETS_FILE}."
                )
        return self


def normalize_fireworks_model(name: str) -> str:
    """Expand short Fireworks ids: `kimi-k3` → `accounts/fireworks/models/kimi-k3`.

    Full ids (`accounts/<acct>/models/<m>`, `accounts/<acct>/routers/<r>`,
    `accounts/<acct>/deployedModels/<d>`) and a `fireworks:` prefix are accepted.
    """
    n = name.strip().removeprefix("fireworks:").strip()
    if not n or n.startswith("accounts/"):
        return n
    return FIREWORKS_MODEL_PREFIX + n


# ─── persistence helpers (used by the settings panel) ───────────────────────


# Subset of fields the panel is allowed to persist. Excludes derived /
# environment-y fields like history_file and ollama_base_url that don't
# belong in a per-user prefs file.
_PERSISTABLE_FIELDS = (
    "provider",
    "fireworks_model",
    "ollama_model",
    "anthropic_model",
    "temperature",
    "max_tokens",
    "writable",
    "mode",
)


def save_user_settings(settings: Settings) -> Path:
    """Write non-secret prefs to settings.json. Returns the path written."""
    payload = {name: getattr(settings, name) for name in _PERSISTABLE_FIELDS}
    # `provider` is a Literal — coerce to plain str so json.dumps is happy.
    payload["provider"] = str(payload["provider"])
    _atomic_write_json(SETTINGS_FILE, payload)
    return SETTINGS_FILE


def save_secret_api_key(key: str | None, field: str = "anthropic_api_key") -> Path:
    """Write (or clear) an API key in secrets.json with mode 0600.

    `field` is the settings field name (`anthropic_api_key` or
    `fireworks_api_key`). Pass `None` or an empty string to remove it.
    """
    if field not in _SECRET_FIELDS:
        raise ValueError(f"not a secret field: {field!r}")
    existing = _read_json(SECRETS_FILE)
    if key is None or not key.strip():
        existing.pop(field, None)
    else:
        existing[field] = key.strip()
    _atomic_write_json(SECRETS_FILE, existing, mode=stat.S_IRUSR | stat.S_IWUSR)
    return SECRETS_FILE


def settings_file_path() -> Path:
    return SETTINGS_FILE


def secrets_file_path() -> Path:
    return SECRETS_FILE


def is_secret_field(name: str) -> bool:
    return name in _SECRET_FIELDS
