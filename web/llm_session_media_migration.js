import { app } from "../../scripts/app.js";

const SESSION_CHAT_NODE_TYPES = new Set([
  "LLMSessionChatNode",
  "LLMSessionChatSimpleNode",
]);
const LITEGRAPH_INPUT = 1;
const AUTOGROW_MEDIA_INPUT_RE = /^media_inputs\.(media_\d+)$/;

function hasLink(input) {
  if (!input) {
    return false;
  }
  if (input.link != null) {
    return true;
  }
  return Array.isArray(input.links) && input.links.length > 0;
}

function renameInput(input, name, targetInput = null, localizedName = name) {
  input.name = name;
  input.label = localizedName;
  input.localized_name = targetInput?.localized_name ?? name;
  input.type = targetInput?.type ?? "*";
}

function normalizeAutogrowMediaLabels(node) {
  let changed = false;
  for (const input of node.inputs) {
    const match = AUTOGROW_MEDIA_INPUT_RE.exec(input?.name ?? "");
    if (!match || input.label === match[1]) {
      continue;
    }
    input.label = match[1];
    changed = true;
  }
  return changed;
}

function removeInput(node, slot) {
  if (slot < 0 || !node.inputs || slot >= node.inputs.length) {
    return;
  }
  if (typeof node.removeInput === "function") {
    node.removeInput(slot);
  } else {
    node.inputs.splice(slot, 1);
  }
}

function findInputSlot(node, name) {
  if (typeof node.findInputSlot === "function") {
    return node.findInputSlot(name);
  }
  return node.inputs?.findIndex((input) => input?.name === name) ?? -1;
}

function migrateInputName(node, fromName, toName, localizedName = toName) {
  let fromSlot = findInputSlot(node, fromName);
  let toSlot = findInputSlot(node, toName);
  if (fromSlot < 0) {
    return false;
  }

  if (toSlot < 0) {
    renameInput(node.inputs[fromSlot], toName, null, localizedName);
    return true;
  }

  const fromInput = node.inputs[fromSlot];
  const toInput = node.inputs[toSlot];
  const fromHasLink = hasLink(fromInput);
  const toHasLink = hasLink(toInput);

  if (fromHasLink && !toHasLink) {
    removeInput(node, toSlot);
    fromSlot = findInputSlot(node, fromName);
    if (fromSlot >= 0) {
      renameInput(node.inputs[fromSlot], toName, toInput, localizedName);
    }
  } else {
    if (fromHasLink && toHasLink) {
      console.warn(
        `[ComfyUI-LLM-Session] Both ${fromName} and ${toName} inputs had links; keeping ${toName} and removing ${fromName}.`,
        node,
      );
    }
    removeInput(node, fromSlot);
  }
  return true;
}

function migrateLegacyMediaInputs(node, usesAutogrow) {
  if (!node || !SESSION_CHAT_NODE_TYPES.has(node.comfyClass || node.type)) {
    return;
  }
  if (!Array.isArray(node.inputs)) {
    return;
  }

  let changed = migrateInputName(node, "image", "media");
  if (usesAutogrow) {
    changed = migrateInputName(
      node,
      "media",
      "media_inputs.media_0",
      "media_0",
    ) || changed;
    changed = normalizeAutogrowMediaLabels(node) || changed;
  }
  if (changed) {
    node.setDirtyCanvas?.(true, true);
  }
}

function hasMediaAutogrow(nodeData) {
  const inputGroups = [nodeData?.input?.required, nodeData?.input?.optional];
  return inputGroups.some((group) =>
    Object.values(group ?? {}).some((spec) => spec?.[0] === "COMFY_AUTOGROW_V3"),
  );
}

function notifyAutogrowOfMigratedLink(node) {
  const notify = () => {
    const slot = findInputSlot(node, "media_inputs.media_0");
    const input = slot >= 0 ? node.inputs[slot] : null;
    const graphLinks = node.graph?.links;
    const link = input?.link != null
      ? (graphLinks?.get?.(input.link) ?? graphLinks?.[input.link])
      : null;
    if (input && link) {
      node.onConnectionsChange?.(LITEGRAPH_INPUT, slot, true, link, input);
    }
  };
  if (typeof requestAnimationFrame === "function") {
    requestAnimationFrame(notify);
  } else {
    notify();
  }
}

app.registerExtension({
  name: "ComfyUI-LLM-Session.MediaInputMigration",
  async beforeRegisterNodeDef(nodeType, nodeData) {
    if (!SESSION_CHAT_NODE_TYPES.has(nodeData.name)) {
      return;
    }
    const usesAutogrow = hasMediaAutogrow(nodeData);
    nodeType.prototype.__llmSessionUsesMediaAutogrow = usesAutogrow;

    const originalOnNodeCreated = nodeType.prototype.onNodeCreated;
    nodeType.prototype.onNodeCreated = function (...args) {
      const result = originalOnNodeCreated?.apply(this, args);
      migrateLegacyMediaInputs(this, usesAutogrow);
      return result;
    };

    const originalConfigure = nodeType.prototype.configure;
    nodeType.prototype.configure = function (...args) {
      const result = originalConfigure?.apply(this, args);
      migrateLegacyMediaInputs(this, usesAutogrow);
      if (usesAutogrow) {
        notifyAutogrowOfMigratedLink(this);
      }
      return result;
    };
  },
  loadedGraphNode(node) {
    const usesAutogrow = Boolean(node.__llmSessionUsesMediaAutogrow);
    migrateLegacyMediaInputs(node, usesAutogrow);
    if (usesAutogrow) {
      notifyAutogrowOfMigratedLink(node);
    }
  },
});
