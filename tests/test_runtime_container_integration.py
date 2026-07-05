from __future__ import annotations


class DummyManager:
    pass


class UnloadTrackingManager:
    def __init__(self) -> None:
        self.unload_calls = 0
        self.unloaded = False

    def unload_model(self):
        self.unload_calls += 1
        self.unloaded = True



def test_turn_execution_dependencies_use_injected_runtime_container(load_nodes_module, monkeypatch):
    module = load_nodes_module()
    container = module.RuntimeContainer(model_manager=None, mem_kv_state={"sid": {"k": "v"}})

    deps = module._build_turn_execution_dependencies(runtime_container=container)

    assert deps["mem_kv_state"] is container.mem_kv_state
    deps["clear_kv_state_for_session"]("sid")
    assert "sid" not in container.mem_kv_state


def test_session_chat_dependencies_reuse_injected_model_manager(load_nodes_module, monkeypatch):
    module = load_nodes_module()

    manager = DummyManager()
    container = module.RuntimeContainer(model_manager=manager, mem_kv_state={})

    deps = module._build_session_chat_node_execution_dependencies(runtime_container=container)

    assert deps.get_or_create_model_manager() is manager


def test_cleanup_unloads_runtime_container_model_manager(load_nodes_module, monkeypatch):
    module = load_nodes_module()

    manager = UnloadTrackingManager()
    container = module.RuntimeContainer(model_manager=manager, mem_kv_state={})
    monkeypatch.setattr(module, "_runtime_container", container)

    module.cleanup()

    assert manager.unloaded is True
    assert container.model_manager is None

def test_resolve_runtime_container_lazy_initializes_default(load_nodes_module, monkeypatch):
    module = load_nodes_module()

    monkeypatch.setattr(module, "GGUFModelManager", DummyManager)
    monkeypatch.setattr(module, "_runtime_container", None)

    container = module._resolve_runtime_container()

    assert isinstance(container.model_manager, DummyManager)
    assert module._runtime_container is container


def test_cleanup_skips_when_default_runtime_container_not_initialized(load_nodes_module, monkeypatch):
    module = load_nodes_module()
    monkeypatch.setattr(module, "_runtime_container", None)

    module.cleanup()

    assert module._runtime_container is None

def test_unload_node_uses_runtime_container_manager(load_nodes_module, monkeypatch):
    module = load_nodes_module()

    manager = UnloadTrackingManager()
    container = module.RuntimeContainer(model_manager=manager, mem_kv_state={})
    monkeypatch.setattr(module, "_runtime_container", container)
    node = module.UnloadLLMModelNode()

    out = node.unload_model(unload_now=True, trigger="tick")

    assert out == ("tick",)
    assert manager.unload_calls == 1


def test_unload_model_clears_runtime_container_mem_kv_state(load_nodes_module, monkeypatch):
    module = load_nodes_module()
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

def test_dialogue_cycle_model_manager_reuse_in_runtime_container(load_nodes_module, monkeypatch):
    module = load_nodes_module()
    container = module.RuntimeContainer(model_manager=None, mem_kv_state={})

    manager_a_1 = module._get_or_create_dialogue_cycle_model_manager("A", runtime_container=container)
    manager_a_2 = module._get_or_create_dialogue_cycle_model_manager("A", runtime_container=container)
    manager_b_1 = module._get_or_create_dialogue_cycle_model_manager("B", runtime_container=container)

    assert manager_a_1 is manager_a_2
    assert manager_a_1 is not manager_b_1
    assert set(container.dialogue_model_managers.keys()) == {"A", "B"}


def test_chat_one_turn_forwards_dialogue_log_prefix_override(load_nodes_module, monkeypatch):
    module = load_nodes_module()
    captured: dict[str, object] = {}

    def _execute_dialogue_cycle_turn(**kwargs):
        captured.update(kwargs)
        return module.TurnExecutionResult(
            assistant_text="ok",
            generation_succeeded=True,
        )

    monkeypatch.setattr(module, "_require_llama_cpp_available", lambda: None)
    monkeypatch.setattr(module, "_get_or_create_model_manager", lambda _manager=None: DummyManager())
    monkeypatch.setattr(module, "_execute_dialogue_cycle_turn", _execute_dialogue_cycle_turn)

    result = module._chat_one_turn(
        user_text="hello",
        session_id="sid_A",
        model="model.gguf",
        mmproj="(Auto detect)",
        system_prompt="sys",
        max_tokens=64,
        temperature=0.7,
        top_p=0.9,
        n_gpu_layers=0,
        tensor_split=None,
        n_ctx=1024,
        log_prefix_override="[LLM Dialogue Cycle A/1]",
    )

    assert result == "ok"
    assert captured["log_prefix_override"] == "[LLM Dialogue Cycle A/1]"


def test_unload_node_unloads_dialogue_cycle_managers(load_nodes_module, monkeypatch):
    module = load_nodes_module()

    shared = UnloadTrackingManager()
    manager_a = UnloadTrackingManager()
    manager_b = UnloadTrackingManager()
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

def test_cleanup_unloads_dialogue_cycle_managers(load_nodes_module, monkeypatch):
    module = load_nodes_module()

    shared = UnloadTrackingManager()
    manager_a = UnloadTrackingManager()
    manager_b = UnloadTrackingManager()
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
