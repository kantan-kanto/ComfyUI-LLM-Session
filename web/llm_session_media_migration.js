import { app } from "../../scripts/app.js";

const SESSION_CHAT_NODE_TYPES = new Set([
  "LLMSessionChatNode",
  "LLMSessionChatSimpleNode",
]);

function hasLink(input) {
  if (!input) {
    return false;
  }
  if (input.link != null) {
    return true;
  }
  return Array.isArray(input.links) && input.links.length > 0;
}

function renameInputToMedia(input) {
  input.name = "media";
  input.localized_name = "media";
  input.type = "*";
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

function migrateLegacyImageInput(node) {
  if (!node || !SESSION_CHAT_NODE_TYPES.has(node.comfyClass || node.type)) {
    return;
  }
  if (!Array.isArray(node.inputs)) {
    return;
  }

  let imageSlot = findInputSlot(node, "image");
  let mediaSlot = findInputSlot(node, "media");
  if (imageSlot < 0) {
    return;
  }

  if (mediaSlot < 0) {
    renameInputToMedia(node.inputs[imageSlot]);
    node.setDirtyCanvas?.(true, true);
    return;
  }

  const imageInput = node.inputs[imageSlot];
  const mediaInput = node.inputs[mediaSlot];
  const imageHasLink = hasLink(imageInput);
  const mediaHasLink = hasLink(mediaInput);

  if (imageHasLink && !mediaHasLink) {
    removeInput(node, mediaSlot);
    imageSlot = findInputSlot(node, "image");
    if (imageSlot >= 0) {
      renameInputToMedia(node.inputs[imageSlot]);
    }
  } else {
    if (imageHasLink && mediaHasLink) {
      console.warn(
        "[ComfyUI-LLM-Session] Both legacy image and media inputs had links; keeping media and removing image.",
        node,
      );
    }
    removeInput(node, imageSlot);
    mediaSlot = findInputSlot(node, "media");
    if (mediaSlot >= 0) {
      renameInputToMedia(node.inputs[mediaSlot]);
    }
  }

  node.setDirtyCanvas?.(true, true);
}

app.registerExtension({
  name: "ComfyUI-LLM-Session.MediaInputMigration",
  async beforeRegisterNodeDef(nodeType, nodeData) {
    if (!SESSION_CHAT_NODE_TYPES.has(nodeData.name)) {
      return;
    }

    const originalOnNodeCreated = nodeType.prototype.onNodeCreated;
    nodeType.prototype.onNodeCreated = function (...args) {
      const result = originalOnNodeCreated?.apply(this, args);
      migrateLegacyImageInput(this);
      return result;
    };

    const originalConfigure = nodeType.prototype.configure;
    nodeType.prototype.configure = function (...args) {
      const result = originalConfigure?.apply(this, args);
      migrateLegacyImageInput(this);
      return result;
    };
  },
  loadedGraphNode(node) {
    migrateLegacyImageInput(node);
  },
});
