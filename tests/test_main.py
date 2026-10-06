"""
Testes do entrypoint CLI (src/main.py): garante que toda falha (config ausente,
step não mapeado, módulo sem main(), exceção na etapa) termina o processo com
código de saída != 0 em vez de continuar silenciosamente.
"""
import sys
import types

import pytest

import src.main as main_module


def _set_step(monkeypatch, step):
    monkeypatch.setattr(sys, "argv", ["main.py", "--step", step])


def _register_fake_step_module(monkeypatch, main_fn=None):
    """Registra um módulo falso em sys.modules, resolvido por importlib.import_module
    sem tocar no mecanismo de import de verdade (evita quebrar outros imports do processo)."""
    fake_module = types.ModuleType("tests._fake_step_module")
    if main_fn is not None:
        fake_module.main = main_fn
    monkeypatch.setitem(sys.modules, "tests._fake_step_module", fake_module)
    return fake_module


def _config_with_step(step="training", module_path="tests/_fake_step_module.py"):
    return {"name": "fake_project", "pipelines": {step: module_path}}


def test_step_argument_is_required(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["main.py"])
    with pytest.raises(SystemExit):
        main_module.main()


def test_missing_project_yml_exits_nonzero(monkeypatch, tmp_path):
    _set_step(monkeypatch, "training")
    monkeypatch.chdir(tmp_path)  # diretório vazio, sem project.yml
    with pytest.raises(SystemExit) as exc:
        main_module.main()
    assert exc.value.code == 1


def test_step_not_mapped_in_pipelines_exits_nonzero(monkeypatch):
    _set_step(monkeypatch, "training")
    monkeypatch.setattr(main_module, "load_config", lambda: {"name": "x", "pipelines": {}})
    with pytest.raises(SystemExit) as exc:
        main_module.main()
    assert exc.value.code == 1


def test_module_without_main_function_exits_nonzero(monkeypatch):
    _set_step(monkeypatch, "training")
    monkeypatch.setattr(main_module, "load_config", lambda: _config_with_step())
    _register_fake_step_module(monkeypatch, main_fn=None)

    with pytest.raises(SystemExit) as exc:
        main_module.main()
    assert exc.value.code == 1


def test_exception_in_step_main_exits_nonzero_not_silently(monkeypatch):
    """A etapa falha (ex: erro no treino) e isso NÃO pode ser engolido - precisa
    terminar o processo com erro, não seguir em frente como se tivesse dado certo."""
    _set_step(monkeypatch, "training")
    monkeypatch.setattr(main_module, "load_config", lambda: _config_with_step())

    def boom():
        raise RuntimeError("falha simulada na etapa")

    _register_fake_step_module(monkeypatch, main_fn=boom)

    with pytest.raises(SystemExit) as exc:
        main_module.main()
    assert exc.value.code == 1


def test_happy_path_calls_step_main_exactly_once(monkeypatch):
    _set_step(monkeypatch, "training")
    monkeypatch.setattr(main_module, "load_config", lambda: _config_with_step())
    calls = []
    _register_fake_step_module(monkeypatch, main_fn=lambda: calls.append(1))

    main_module.main()  # não deve lançar

    assert calls == [1]
