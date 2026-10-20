import { Buffer } from "node:buffer";
import {
  API_VERSION,
  BASE_URL,
  DEFAULT_MODEL,
  HttpError,
  fixCors,
  generateId,
  makeHeaders,
  parseStream,
  parseStreamFlush,
  safetySettings,
} from "./worker.mjs";

const stopReasonsMap = { //https://ai.google.dev/api/rest/v1/GenerateContentResponse#finishreason
  "STOP": "end_turn",
  "MAX_TOKENS": "max_tokens",
  "SAFETY": "refusal",
  "RECITATION": "end_turn",
};

const parseImg = async (url) => {
  let mimeType, data;
  if (url.startsWith("http://") || url.startsWith("https://")) {
    try {
      const response = await fetch(url);
      if (!response.ok) {
        throw new Error(`${response.status} ${response.statusText} (${url})`);
      }
      mimeType = response.headers.get("content-type");
      data = Buffer.from(await response.arrayBuffer()).toString("base64");
    } catch (err) {
      throw new Error("Error fetching image: " + err.toString());
    }
  } else {
    const match = url.match(/^data:(?<mimeType>.*?)(;base64)?,(?<data>.*)$/);
    if (!match) {
      throw new HttpError("Invalid image data: " + url, 400);
    }
    ({ mimeType, data } = match.groups);
  }
  return { inlineData: { mimeType, data } };
};

const transformMsg = async (content, toolNames) => {
  const parts = [];
  if (typeof content === "string") {
    if (content) { parts.push({ text: content }); }
    return parts;
  }
  if (!Array.isArray(content)) {
    throw new HttpError("Invalid message content", 400);
  }
  for (const block of content) {
    switch (block.type) {
      case "text":
        if (block.text) { parts.push({ text: block.text }); }
        break;
      case "image": {
        const source = block.source;
        if (source?.type === "base64") {
          parts.push({
            inlineData: { mimeType: source.media_type, data: source.data },
          });
        } else if (source?.type === "url") {
          parts.push(await parseImg(source.url));
        } else {
          throw new HttpError(`Unsupported image source type: "${source?.type}"`, 400);
        }
        break;
      }
      case "tool_use": {
        const id = block.id ?? "toolu_" + generateId();
        toolNames.set(id, block.name);
        parts.push({
          functionCall: {
            id,
            name: block.name,
            args: block.input ?? {},
          }
        });
        break;
      }
      default:
        throw new HttpError(`Unknown content block type: "${block.type}"`, 400);
    }
  }
  return parts;
};

const transformToolResult = (block, toolNames) => {
  const { tool_use_id, content } = block;
  const name = toolNames.get(tool_use_id);
  if (!name) {
    throw new HttpError("Unknown tool_use_id: " + tool_use_id, 400);
  }
  let response;
  if (typeof content === "string") {
    try {
      response = JSON.parse(content);
    } catch (err) {
      response = { result: content };
    }
    if (typeof response === "object" && response !== null && !Array.isArray(response)) {
      response = { result: response };
    } else {
      response = { result: response };
    }
  } else if (Array.isArray(content)) {
    response = { result: content.map(el => el.type === "text" ? el.text : "").join("") };
  } else {
    response = { result: content ?? null };
  }
  return {
    functionResponse: { id: tool_use_id, name, response },
  };
};

const transformMessages = async (messages) => {
  if (!messages) { return; }
  const contents = [];
  let system_instruction;
  const toolNames = new Map();
  for (const item of messages) {
    switch (item.role) {
      case "system":
        system_instruction = { parts: await transformMsg(item.content, toolNames) };
        continue;
      case "assistant":
        contents.push({ role: "model", parts: await transformMsg(item.content, toolNames) });
        continue;
      case "user": {
        const toolResults = Array.isArray(item.content)
          ? item.content.filter(el => el.type === "tool_result")
          : [];
        const parts = await transformMsg(
          toolResults.length
            ? item.content.filter(el => el.type !== "tool_result")
            : item.content,
          toolNames
        );
        if (toolResults.length) {
          contents.push({
            role: "function", // ignored
            parts: toolResults.map(el => transformToolResult(el, toolNames)),
          });
        }
        if (parts.length) {
          contents.push({ role: "user", parts });
        }
        continue;
      }
      default:
        throw new HttpError(`Unknown message role: "${item.role}"`, 400);
    }
  }
  if (system_instruction) {
    if (!contents[0]?.parts.some(part => part.text)) {
      contents.unshift({ role: "user", parts: { text: " " } });
    }
  }
  return { system_instruction, contents };
};

const transformTools = (req) => {
  let tools, tool_config;
  if (req.tools?.length) {
    tools = [{
      function_declarations: req.tools.map(tool => ({
        name: tool.name,
        description: tool.description,
        parameters: tool.input_schema,
      }))
    }];
  }
  if (req.tool_choice) {
    const { type, name } = req.tool_choice;
    const allowed_function_names = type === "tool" ? [ name ] : undefined;
    if (allowed_function_names || typeof type === "string") {
      tool_config = {
        function_calling_config: {
          mode: allowed_function_names ? "ANY" : (type === "auto" ? "AUTO" : "ANY"),
          allowed_function_names,
        }
      };
    }
  }
  return { tools, tool_config };
};

const transformConfig = (req, isV3) => {
  const cfg = {};
  if (req.max_tokens != null) { cfg.maxOutputTokens = req.max_tokens; }
  if (req.temperature != null) { cfg.temperature = req.temperature; }
  if (req.top_p != null) { cfg.topP = req.top_p; }
  if (req.top_k != null) { cfg.topK = req.top_k; }
  if (req.stop_sequences?.length) { cfg.stopSequences = req.stop_sequences; }
  if (req.thinking?.type === "enabled") {
    cfg.thinkingConfig = isV3
      ? { thinkingLevel: (req.thinking.budget_tokens ?? 0) > 8192 ? "high" : "medium" }
      : { thinkingBudget: req.thinking.budget_tokens ?? 1024 };
  }
  return cfg;
};

const transformUsage = (data) => ({
  input_tokens: data.promptTokenCount ?? 0,
  output_tokens: (data.candidatesTokenCount ?? 0)
    + (data.thoughtsTokenCount ?? 0)
    + (data.toolUsePromptTokenCount ?? 0),
});

const processAnthropicResponse = (data, model, id) => {
  const cand = data.candidates?.[0] ?? {};
  const content = [];
  for (const part of cand.content?.parts ?? []) {
    if (typeof part.text === "string") {
      if (part.thought) {
        content.push({ type: "thinking", thinking: part.text });
        if (part.thoughtSignature) {
          content[content.length - 1].signature = part.thoughtSignature;
        }
      } else if (part.text) {
        content.push({ type: "text", text: part.text });
      }
    } else if (part.functionCall) {
      const fc = part.functionCall;
      content.push({
        type: "tool_use",
        id: fc.id ?? "toolu_" + generateId(),
        name: fc.name,
        input: fc.args ?? {},
      });
    } else {
      console.error("Unexpected part type:", JSON.stringify(part, 2));
    }
  }
  if (!content.length) { content.push({ type: "text", text: "" }); }
  let stop_reason;
  if (content.some(el => el.type === "tool_use")) {
    stop_reason = "tool_use";
  } else {
    stop_reason = stopReasonsMap[cand.finishReason] ?? "end_turn";
  }
  return JSON.stringify({
    id: data.responseId ?? id,
    type: "message",
    role: "assistant",
    model: data.modelVersion ?? model,
    content,
    stop_reason,
    stop_sequence: null,
    usage: data.usageMetadata && transformUsage(data.usageMetadata),
  });
};

const sseEvent = (type, data) => `event: ${type}\ndata: ${JSON.stringify(data)}\n\n`;

function closeBlock (controller) {
  if (this.blockType != null) {
    controller.enqueue(sseEvent("content_block_stop", {
      type: "content_block_stop",
      index: this.blockIndex,
    }));
    this.blockType = null;
  }
}

function toAnthropicStream (line, controller) {
  let data;
  try {
    data = JSON.parse(line);
    if (!data.candidates) {
      throw new Error("Invalid completion chunk object");
    }
  } catch (err) {
    console.error("Error parsing response:", err);
    if (!this.shared.is_buffers_rest) { line += "\n\n"; }
    controller.enqueue(line); // output as is
    return;
  }
  const cand = data.candidates[0];
  this.lastData = data;
  this.lastCandidate = cand;
  if (!this.started) {
    this.started = true;
    controller.enqueue(sseEvent("message_start", {
      type: "message_start",
      message: {
        id: this.id,
        type: "message",
        role: "assistant",
        model: data.modelVersion ?? this.model,
        content: [],
        stop_reason: null,
        stop_sequence: null,
        usage: { input_tokens: data.usageMetadata?.promptTokenCount ?? 0, output_tokens: 0 },
      },
    }));
  }
  for (const part of cand?.content?.parts ?? []) {
    if (typeof part.text === "string") {
      if (!part.text) { continue; }
      if (part.thought) {
        if (this.blockType !== "thinking") {
          this.closeBlock(controller);
          this.blockIndex++;
          this.blockType = "thinking";
          controller.enqueue(sseEvent("content_block_start", {
            type: "content_block_start",
            index: this.blockIndex,
            content_block: { type: "thinking", thinking: "" },
          }));
        }
        controller.enqueue(sseEvent("content_block_delta", {
          type: "content_block_delta",
          index: this.blockIndex,
          delta: { type: "thinking_delta", thinking: part.text },
        }));
      } else {
        if (this.blockType !== "text") {
          this.closeBlock(controller);
          this.blockIndex++;
          this.blockType = "text";
          controller.enqueue(sseEvent("content_block_start", {
            type: "content_block_start",
            index: this.blockIndex,
            content_block: { type: "text", text: "" },
          }));
        }
        controller.enqueue(sseEvent("content_block_delta", {
          type: "content_block_delta",
          index: this.blockIndex,
          delta: { type: "text_delta", text: part.text },
        }));
      }
    } else if (part.functionCall) {
      this.closeBlock(controller);
      const fc = part.functionCall;
      this.blockIndex++;
      this.blockType = "tool_use";
      controller.enqueue(sseEvent("content_block_start", {
        type: "content_block_start",
        index: this.blockIndex,
        content_block: {
          type: "tool_use",
          id: fc.id ?? "toolu_" + generateId(),
          name: fc.name,
          input: {},
        },
      }));
      controller.enqueue(sseEvent("content_block_delta", {
        type: "content_block_delta",
        index: this.blockIndex,
        delta: { type: "input_json_delta", partial_json: JSON.stringify(fc.args ?? {}) },
      }));
      controller.enqueue(sseEvent("content_block_stop", {
        type: "content_block_stop",
        index: this.blockIndex,
      }));
      this.blockType = null;
      this.toolUsed = true;
    } else {
      console.error("Unexpected part type:", JSON.stringify(part, 2));
    }
  }
}

function toAnthropicStreamFlush (controller) {
  if (!this.started) { return; }
  this.closeBlock(controller);
  let stop_reason;
  if (this.toolUsed) {
    stop_reason = "tool_use";
  } else {
    stop_reason = stopReasonsMap[this.lastCandidate?.finishReason] ?? "end_turn";
  }
  controller.enqueue(sseEvent("message_delta", {
    type: "message_delta",
    delta: { stop_reason, stop_sequence: null },
    usage: {
      output_tokens: this.lastData?.usageMetadata
        ? transformUsage(this.lastData.usageMetadata).output_tokens
        : 0,
    },
  }));
  controller.enqueue(sseEvent("message_stop", { type: "message_stop" }));
}

export async function handleAnthropic (req, apiKey) {
  let model = req.model;
  switch (true) {
    case typeof model !== "string":
      throw new HttpError("model is not specified", 400);
    case model.startsWith("models/"):
      model = model.substring(7);
      break;
    case model.startsWith("gemini-"):
    case model.startsWith("gemma-"):
      break;
    default:
      model = DEFAULT_MODEL;
  }
  let isV3 = model.startsWith("gemini-3");
  let body = {
    ...await transformMessages(req.messages),
    safetySettings,
    generationConfig: transformConfig(req, isV3),
    ...transformTools(req),
  };
  switch (true) {
    case model.endsWith(":search"):
      model = model.slice(0, -7);
      // eslint-disable-next-line no-fallthrough
    case req.model?.includes("-search-preview"):
      body.tools ??= [];
      body.tools.push({ googleSearch: {} });
  }
  const TASK = req.stream ? "streamGenerateContent" : "generateContent";
  let url = `${BASE_URL}/${API_VERSION}/models/${model}:${TASK}`;
  if (req.stream) { url += "?alt=sse"; }
  const response = await fetch(url, {
    method: "POST",
    headers: makeHeaders(apiKey, { "Content-Type": "application/json" }),
    body: JSON.stringify(body),
  });

  body = response.body;
  if (response.ok) {
    const id = "msg_" + generateId();
    const shared = {};
    if (req.stream) {
      body = response.body
        .pipeThrough(new TextDecoderStream())
        .pipeThrough(new TransformStream({
          transform: parseStream,
          flush: parseStreamFlush,
          buffer: "",
          shared,
        }))
        .pipeThrough(new TransformStream({
          transform: toAnthropicStream,
          flush: toAnthropicStreamFlush,
          model, id, shared, closeBlock,
          started: false,
          blockIndex: -1,
          blockType: null,
          toolUsed: false,
          lastData: null,
          lastCandidate: null,
        }))
        .pipeThrough(new TextEncoderStream());
    } else {
      body = await response.text();
      try {
        body = JSON.parse(body);
        if (!body.candidates) {
          throw new Error("Invalid completion object");
        }
      } catch (err) {
        console.error("Error parsing response:", err);
        return new Response(body, fixCors(response)); // output as is
      }
      body = processAnthropicResponse(body, model, id);
    }
  }
  return new Response(body, fixCors(response));
}