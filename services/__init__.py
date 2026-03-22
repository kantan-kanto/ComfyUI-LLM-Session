# services/: orchestration layer that coordinates multi-step application flows.
# - chat_turn_service.py:
#   - ChatTurnService for A<->B dialogue-cycle runtime orchestration.
#   - DialogueCycleNodeExecutionService for node-input request/dependency orchestration.
# - turn_execution_service.py:
#   - TurnExecutionService for shared single-turn runtime execution.
#   - SessionChatNodeExecutionService for node-input request/dependency orchestration.
# - generation_execution_service.py:
#   - GenerationExecutionService for adaptive generation orchestration and output normalization.
# - kv_state_service.py:
#   - KvStateService for KV cache restore/save and mismatch recovery.
# - history_persistence_service.py:
#   - HistoryPersistenceService for turn append, summarization, and history write.



