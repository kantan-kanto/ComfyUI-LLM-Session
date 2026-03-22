from __future__ import annotations

import importlib
import sys
import types


def _load_nodes_module(monkeypatch):
    fake_folder_paths = types.SimpleNamespace(
        models_dir="C:/models",
        get_folder_paths=lambda _key: [],
        get_filename_list=lambda _key: [],
        get_output_directory=lambda: "C:/output",
    )
    monkeypatch.setitem(sys.modules, "folder_paths", fake_folder_paths)
    sys.modules.pop("llm_session_nodes", None)
    return importlib.import_module("llm_session_nodes")


def test_turn_execution_dependencies_use_injected_runtime_container(monkeypatch):
    module = _load_nodes_module(monkeypatch)
    container = module.RuntimeContainer(model_manager=None, mem_kv_state={"sid": {"k": "v"}})

    deps = module._build_turn_execution_dependencies(runtime_container=container)

    assert deps["mem_kv_state"] is container.mem_kv_state
    deps["clear_kv_state_for_session"]("sid")
    assert "sid" not in container.mem_kv_state


def test_session_chat_dependencies_reuse_injected_model_manager(monkeypatch):
    module = _load_nodes_module(monkeypatch)

    class DummyManager:
        pass

    manager = DummyManager()
    container = module.RuntimeContainer(model_manager=manager, mem_kv_state={})

    deps = module._build_session_chat_node_execution_dependencies(runtime_container=container)

    assert deps.get_or_create_model_manager() is manager


def test_cleanup_unloads_runtime_container_model_manager(monkeypatch):
    module = _load_nodes_module(monkeypatch)

    class DummyManager:
        def __init__(self) -> None:
            self.unloaded = False

        def unload_model(self):
            self.unloaded = True

    manager = DummyManager()
    container = module.RuntimeContainer(model_manager=manager, mem_kv_state={})
    monkeypatch.setattr(module, "_runtime_container", container)

    module.cleanup()

    assert manager.unloaded is True
    assert container.model_manager is None

def test_resolve_runtime_container_lazy_initializes_default(monkeypatch):
    module = _load_nodes_module(monkeypatch)

    class DummyManager:
        pass

    monkeypatch.setattr(module, "GGUFModelManager", DummyManager)
    monkeypatch.setattr(module, "_runtime_container", None)

    container = module._resolve_runtime_container()

    assert isinstance(container.model_manager, DummyManager)
    assert module._runtime_container is container


def test_cleanup_skips_when_default_runtime_container_not_initialized(monkeypatch):
    module = _load_nodes_module(monkeypatch)
    monkeypatch.setattr(module, "_runtime_container", None)

    module.cleanup()

    assert module._runtime_container is None

def test_unload_node_uses_runtime_container_manager(monkeypatch):
    module = _load_nodes_module(monkeypatch)

    class DummyManager:
        def __init__(self) -> None:
            self.unload_calls = 0

        def unload_model(self):
            self.unload_calls += 1

    manager = DummyManager()
    container = module.RuntimeContainer(model_manager=manager, mem_kv_state={})
    monkeypatch.setattr(module, "_runtime_container", container)
    node = module.UnloadLLMModelNode()

    out = node.unload_model(unload_now=True, trigger="tick")

    assert out == ("tick",)
    assert manager.unload_calls == 1


def test_unload_model_clears_runtime_container_mem_kv_state(monkeypatch):
    module = _load_nodes_module(monkeypatch)
    manager = module.GGUFModelManager()
    manager.model = object()
    manager.chat_handler = object()
    manager.current_model_path = "model.gguf"
    manager.current_mmproj_path = "mmproj.gguf"
    manager._signature = ("sig",)

    calls: list[tuple[object, bool]] = []

    def _invalidate_cache(llm, remove_disk_data=False):
        calls.append((llm, bool(remove_disk_data)))

    monkeypatch.setattr(manager, "invalidate_cache", _invalidate_cache)
    container = module.RuntimeContainer(model_manager=manager, mem_kv_state={"sid": {"state": 1}})
    monkeypatch.setattr(module, "_runtime_container", container)

    manager.unload_model()

    assert len(calls) == 1
    assert calls[0][1] is False
    assert manager.model is None
    assert manager.chat_handler is None
    assert manager.current_model_path is None
    assert manager.current_mmproj_path is None
    assert manager._signature is None
    assert container.mem_kv_state == {}

def test_dialogue_cycle_model_manager_reuse_in_runtime_container(monkeypatch):
    module = _load_nodes_module(monkeypatch)
    container = module.RuntimeContainer(model_manager=None, mem_kv_state={})

    manager_a_1 = module._get_or_create_dialogue_cycle_model_manager("A", runtime_container=container)
    manager_a_2 = module._get_or_create_dialogue_cycle_model_manager("A", runtime_container=container)
    manager_b_1 = module._get_or_create_dialogue_cycle_model_manager("B", runtime_container=container)

    assert manager_a_1 is manager_a_2
    assert manager_a_1 is not manager_b_1
    assert set(container.dialogue_model_managers.keys()) == {"A", "B"}


def test_unload_node_unloads_dialogue_cycle_managers(monkeypatch):
    module = _load_nodes_module(monkeypatch)

    class DummyManager:
        def __init__(self) -> None:
            self.unload_calls = 0

        def unload_model(self):
            self.unload_calls += 1

    shared = DummyManager()
    manager_a = DummyManager()
    manager_b = DummyManager()
    container = module.RuntimeContainer(
        model_manager=shared,
        mem_kv_state={},
        dialogue_model_managers={"A": manager_a, "B": manager_b},
    )
    monkeypatch.setattr(module, "_runtime_container", container)

    node = module.UnloadLLMModelNode()
    out = node.unload_model(unload_now=True, trigger="tick")

    assert out == ("tick",)
    assert shared.unload_calls == 1
    assert manager_a.unload_calls == 1
    assert manager_b.unload_calls == 1
    assert container.model_manager is None
    assert container.dialogue_model_managers == {}

def test_cleanup_unloads_dialogue_cycle_managers(monkeypatch):
    module = _load_nodes_module(monkeypatch)

    class DummyManager:
        def __init__(self) -> None:
            self.unloaded = False

        def unload_model(self):
            self.unloaded = True

    shared = DummyManager()
    manager_a = DummyManager()
    manager_b = DummyManager()
    container = module.RuntimeContainer(
        model_manager=shared,
        mem_kv_state={},
        dialogue_model_managers={"A": manager_a, "B": manager_b},
    )
    monkeypatch.setattr(module, "_runtime_container", container)

    module.cleanup()

    assert shared.unloaded is True
    assert manager_a.unloaded is True
    assert manager_b.unloaded is True
    assert container.model_manager is None
    assert container.dialogue_model_managers == {}
