import { events } from "fetch-event-stream";

import type { LanguageModelV2StreamPart, LanguageModelV2Usage } from "@ai-sdk/provider";
import { mapWorkersAIUsage } from "./map-workersai-usage";
import { processPartialToolCalls } from "./utils";

export function getMappedStream(response: Response) {
	const chunkEvent = events(response);
	let usage: LanguageModelV2Usage = { inputTokens: 0, outputTokens: 0, totalTokens: 0 }
	const partialToolCalls: any[] = [];
	let textStarted = false;

	return new ReadableStream<LanguageModelV2StreamPart>({
		async start(controller) {
			console.log('[STREAMING] Starting stream processing');
			for await (const event of chunkEvent) {
				console.log('[STREAMING] Event received:', event);
				if (!event.data) {
					console.log('[STREAMING] No data in event, continuing');
					continue;
				}
				if (event.data === "[DONE]") {
					console.log('[STREAMING] Received [DONE], breaking');
					break;
				}
				console.log('[STREAMING] Raw event data:', event.data);
				const chunk = JSON.parse(event.data);
				console.log('[STREAMING] Parsed chunk:', chunk);
				if (chunk.usage) {
					usage = mapWorkersAIUsage(chunk);
				}
				if (chunk.tool_calls) {
					partialToolCalls.push(...chunk.tool_calls);
					continue;
				}
				if (chunk.response?.length) {
					console.log('[STREAMING] Text chunk found:', chunk.response);
					if (!textStarted) {
						console.log('[STREAMING] Starting text stream');
						controller.enqueue({
							type: "text-start",
							id: crypto.randomUUID(),
						});
						textStarted = true;
					}
					controller.enqueue({
						type: "text-delta",
						id: crypto.randomUUID(),
						delta: chunk.response,
					});
					console.log('[STREAMING] Enqueued text-delta');
				} else {
					console.log('[STREAMING] No response in chunk');
				}
			}

			if (textStarted) {
				controller.enqueue({
					type: "text-end",
					id: crypto.randomUUID(),
				});
			}

			if (partialToolCalls.length > 0) {
				const toolCalls = processPartialToolCalls(partialToolCalls);
				toolCalls.map((toolCall) => {
					controller.enqueue({
						type: "tool-call",
						...toolCall,
					});
				});
			}

			controller.enqueue({
				type: "finish",
				finishReason: "stop",
				usage: usage,
			});
			controller.close();
		},
	});
}
