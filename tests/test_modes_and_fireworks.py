from __future__ import annotations

from pathlib import Path

import pytest

from free_agent import config
from free_agent.agent.profile import AgentProfile, SubAgentProfile
from free_agent.config import DEFAULT_FIREWORKS_MODEL, Settings, normalize_fireworks_model
from free_agent.modes import MODES, get_mode


@pytest.fixture(autouse=True)
def _isolated_config(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(config, "SETTINGS_FILE", tmp_path / "settings.json")
    monkeypatch.setattr(config, "SECRETS_FILE", tmp_path / "secrets.json")
    for var in ("FIREWORKS_API_KEY", "FREE_AGENT_PROVIDER", "FREE_AGENT_MODE", "ANTHROPIC_API_KEY"):
        monkeypatch.delenv(var, raising=False)


def _settings(**kw: object) -> Settings:
    return Settings(_env_file=None, **kw)  # type: ignore[call-arg]


# ─── config ─────────────────────────────────────────────────────────────────


def test_fireworks_is_default_provider() -> None:
    s = _settings(fireworks_api_key="fw_test")
    assert s.provider == "fireworks"
    assert s.active_model == DEFAULT_FIREWORKS_MODEL


def test_fireworks_requires_key() -> None:
    with pytest.raises(ValueError, match="FIREWORKS_API_KEY"):
        _settings()


def test_ollama_needs_no_key() -> None:
    assert _settings(provider="ollama").active_model == "qwen2.5:7b"


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("glm-5p3", "accounts/fireworks/models/glm-5p3"),
        ("fireworks:kimi-k3", "accounts/fireworks/models/kimi-k3"),
        ("accounts/me/deployedModels/x", "accounts/me/deployedModels/x"),
        ("accounts/fireworks/routers/kimi-k3-fast", "accounts/fireworks/routers/kimi-k3-fast"),
    ],
)
def test_normalize_fireworks_model(raw: str, expected: str) -> None:
    assert normalize_fireworks_model(raw) == expected


def test_short_model_id_normalized_on_load() -> None:
    s = _settings(fireworks_api_key="k", fireworks_model="glm-5p3")
    assert s.fireworks_model == "accounts/fireworks/models/glm-5p3"


def test_fireworks_key_persisted_as_secret() -> None:
    config.save_secret_api_key("fw_secret", "fireworks_api_key")
    s = _settings()
    assert s.fireworks_api_key is not None
    assert s.fireworks_api_key.get_secret_value() == "fw_secret"
    assert oct(config.SECRETS_FILE.stat().st_mode & 0o777) == "0o600"


def test_mode_persisted_in_settings() -> None:
    s = _settings(fireworks_api_key="k", mode="finance")
    config.save_user_settings(s)
    assert _settings(fireworks_api_key="k").mode == "finance"


# ─── modes ──────────────────────────────────────────────────────────────────


def test_four_modes_exist_with_tools() -> None:
    assert set(MODES) == {"code", "science", "security", "finance"}
    for m in MODES.values():
        assert m.tools, m.name
        assert m.system_prompt.strip()


@pytest.mark.parametrize(("alias", "name"), [("finanzas", "finance"), ("ciberseguridad", "security"),
                                             ("ciencia", "science"), ("CODE", "code")])
def test_mode_aliases(alias: str, name: str) -> None:
    mode = get_mode(alias)
    assert mode is not None and mode.name == name


@pytest.mark.parametrize("off", [None, "", "off", "none"])
def test_mode_off(off: str | None) -> None:
    assert get_mode(off) is None


def test_unknown_mode() -> None:
    with pytest.raises(ValueError, match="unknown mode"):
        get_mode("astrology")


def test_mode_apply_merges_profile() -> None:
    profile = AgentProfile(
        system_prompt="Always answer in Spanish.",
        subagents=[SubAgentProfile(name="mine", description="d", system_prompt="p")],
    )
    merged = MODES["code"].apply(profile, ["current_time", "shell"])
    assert merged.system_prompt is not None
    assert "CODE mode" in merged.system_prompt
    assert "Always answer in Spanish." in merged.system_prompt
    assert merged.tools is not None
    assert {"shell", "str_replace", "grep_code"} <= set(merged.tools)
    assert [sa.name for sa in merged.subagents] == ["mine", "code-explorer", "code-reviewer"]


def test_assemble_agent_with_each_mode() -> None:
    from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
    from langchain_core.messages import AIMessage

    from free_agent.agent.builder import assemble_agent

    model = FakeMessagesListChatModel(responses=[AIMessage(content="ok")])
    for name in [*MODES, None]:
        assert assemble_agent(model, AgentProfile(), mode=name) is not None


# ─── CLI helpers ────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("name", "current", "expected"),
    [
        ("glm-5p3", "fireworks", ("fireworks", "glm-5p3")),
        ("fireworks:glm-5p3", "ollama", ("fireworks", "glm-5p3")),
        ("accounts/fireworks/models/kimi-k3", "ollama", ("fireworks", "accounts/fireworks/models/kimi-k3")),
        ("qwen3:8b", "fireworks", ("ollama", "qwen3:8b")),
        ("ollama:llama3", "fireworks", ("ollama", "llama3")),
        ("claude-sonnet-4-6", "fireworks", ("anthropic", "claude-sonnet-4-6")),
        ("llama3", "ollama", ("ollama", "llama3")),
    ],
)
def test_infer_provider(name: str, current: str, expected: tuple[str, str]) -> None:
    from free_agent.cli.commands import _infer_provider

    assert _infer_provider(name, current) == expected


def test_settings_panel_parses_fireworks_and_mode() -> None:
    from free_agent.cli.settings_panel import _KEY_KEEP_SENTINEL, _parse_yaml, _Snapshot

    snap = _Snapshot(
        workspace_name="default", provider="ollama", fireworks_model=DEFAULT_FIREWORKS_MODEL,
        fireworks_api_key=None, mode="", ollama_model="qwen2.5:7b",
        anthropic_model="claude-sonnet-4-6", anthropic_api_key=None,
        temperature=0.7, max_tokens=4096, writable_root=None,
    )
    out = _parse_yaml(
        {"provider": "fireworks", "fireworks_model": "glm-5p3", "fireworks_api_key": "fw_new", "mode": "security"},
        snap,
    )
    assert (out.provider, out.fireworks_model, out.fireworks_api_key, out.mode) == (
        "fireworks", "accounts/fireworks/models/glm-5p3", "fw_new", "security",
    )
    # bare `off` in YAML parses as False; keep-sentinel keeps the key
    out2 = _parse_yaml({"mode": False, "fireworks_api_key": _KEY_KEEP_SENTINEL}, out)
    assert out2.mode == "" and out2.fireworks_api_key == "fw_new"
    with pytest.raises(ValueError, match="fireworks"):
        _parse_yaml({"provider": "fireworks"}, snap)
