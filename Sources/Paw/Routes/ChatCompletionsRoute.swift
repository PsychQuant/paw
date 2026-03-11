import Foundation
import Logging
import MLX
import MLXLLM
@preconcurrency import MLXLMCommon
@preconcurrency import Tokenizers
import Vapor

func registerChatCompletionsRoute(
    _ app: Application,
    modelManager: ModelManager,
    isVLM: Bool = false,
    promptCacheManager: PromptCacheManager? = nil
) throws {
    app.post("v1", "chat", "completions") { req async throws -> Response in
        let chatRequest = try req.content.decode(ChatCompletionRequest.self)
        let logger = req.logger
        let reqModelId = chatRequest.model
        let (modelContainer, tokenizer, loadedModelName) = try await modelManager.getModel(
            requestedModelId: reqModelId)
        guard let eosTokenId = tokenizer.eosTokenId else {
            throw ProcessingError(
                status: .internalServerError, reason: "Tokenizer EOS token ID missing",
                modelId: loadedModelName)
        }

        let userInput = processUserMessages(chatRequest, isVLM: isVLM)

        if isVLM {
            logger.info(
                "VLM: Processing request with \(userInput.images.count) images and \(userInput.videos.count) videos"
            )
        }
        let estimatedTokens = estimatePromptTokens(
            messages: chatRequest.messages, tokenizer: tokenizer)

        logger.info(
            "Received CHAT completion request for model '\(loadedModelName)', estimated prompt tokens: \(estimatedTokens)"
        )

        let maxTokens = chatRequest.maxTokens ?? GenerationDefaults.maxTokens
        let temperature = chatRequest.temperature ?? GenerationDefaults.temperature
        let topP = chatRequest.topP ?? GenerationDefaults.topP
        let streamResponse = chatRequest.stream ?? GenerationDefaults.stream
        let stopWords = chatRequest.stop ?? GenerationDefaults.stopSequences
        let stopIdSequences = stopSequencesToIds(stopWords: stopWords, tokenizer: tokenizer)
        let repetitionPenalty =
            chatRequest.repetitionPenalty ?? GenerationDefaults.repetitionPenalty
        let repetitionContextSize =
            chatRequest.repetitionContextSize ?? GenerationDefaults.repetitionContextSize

        let kvBits = chatRequest.kvBits
        let kvGroupSize = chatRequest.kvGroupSize ?? GenerationDefaults.kvGroupSize
        let quantizedKVStart =
            chatRequest.quantizedKVStart ?? GenerationDefaults.quantizedKVStart

        try KVCacheValidation.validate(
            bits: kvBits,
            groupSize: kvGroupSize,
            quantizationStart: quantizedKVStart
        )

        try await validateProcessor(modelContainer: modelContainer)

        let detokenizer = NaiveStreamingDetokenizer(tokenizer: tokenizer)

        let generationContext = ChatGenerationContext(
            modelContainer: modelContainer,
            tokenizer: tokenizer,
            eosTokenId: eosTokenId,
            userInput: userInput,
            logger: logger,
            promptCacheManager: isVLM ? nil : promptCacheManager
        )
        let generationParameters = ChatGenerationParameters(
            maxTokens: maxTokens,
            temperature: temperature,
            topP: topP,
            repetitionPenalty: repetitionPenalty,
            repetitionContextSize: repetitionContextSize,
            kvBits: kvBits,
            kvGroupSize: kvGroupSize,
            quantizedKVStart: quantizedKVStart
        )
        let responseContext = ChatResponseContext(
            loadedModelName: loadedModelName,
            stopIdSequences: stopIdSequences,
            detokenizer: detokenizer,
            estimatedPromptTokens: estimatedTokens
        )

        if streamResponse {
            return try await handleStreamingChatResponse(
                generationContext: generationContext,
                generationParameters: generationParameters,
                responseContext: responseContext
            )
        } else {
            return try await handleNonStreamingChatResponse(
                req: req,
                generationContext: generationContext,
                generationParameters: generationParameters,
                responseContext: responseContext
            )
        }
    }
}

private func handleStreamingChatResponse(
    generationContext: ChatGenerationContext,
    generationParameters: ChatGenerationParameters,
    responseContext: ChatResponseContext
) async throws -> Response {
    let headers = HTTPHeaders([
        ("Content-Type", "text/event-stream"),
        ("Cache-Control", "no-cache"),
        ("Connection", "keep-alive"),
    ])
    let response = Response(status: .ok, headers: headers)
    response.body = .init(stream: { writer in
        let chatId = "chatcmpl-\(UUID().uuidString)"
        let created = Int(Date().timeIntervalSince1970)
        let systemFingerprint: String? = nil
        var streamDetokenizer = responseContext.detokenizer

        Task {
            var generatedTokens: [Int] = []
            generatedTokens.reserveCapacity(generationParameters.maxTokens)
            var finalFinishReason: String?

            do {
                generationContext.logger.info(
                    "Starting CHAT stream generation (ID: \(chatId)) for model \(responseContext.loadedModelName)"
                )

                let initialDelta = ChatCompletionDelta(role: "assistant", content: "")
                let initialChoice = ChatCompletionChoiceDelta(
                    index: 0, delta: initialDelta, finishReason: nil)
                let initialChunk = ChatCompletionChunkResponse(
                    id: chatId, created: created,
                    model: responseContext.loadedModelName,
                    systemFingerprint: systemFingerprint, choices: [initialChoice])
                if let initialSse = encodeSSE(
                    response: initialChunk, logger: generationContext.logger)
                {
                    writer.write(.buffer(.init(string: initialSse)))
                }

                let tokenStream = try await generateChatTokenStream(
                    context: generationContext,
                    parameters: generationParameters
                )

                for try await token in tokenStream {
                    generatedTokens.append(token)
                    streamDetokenizer.append(token: token)
                    let stopCondition = checkStoppingCriteria(
                        tokens: generatedTokens,
                        stopIdSequences: responseContext.stopIdSequences,
                        eosTokenId: generationContext.eosTokenId)

                    if stopCondition.stopMet {
                        finalFinishReason = "stop"
                        break
                    }

                    if let newTextChunk = streamDetokenizer.next() {
                        let delta = ChatCompletionDelta(role: nil, content: newTextChunk)
                        let choice = ChatCompletionChoiceDelta(
                            index: 0, delta: delta, finishReason: nil)
                        let chunkResponse = ChatCompletionChunkResponse(
                            id: chatId, created: created,
                            model: responseContext.loadedModelName,
                            systemFingerprint: systemFingerprint, choices: [choice]
                        )
                        if let sseString = encodeSSE(
                            response: chunkResponse, logger: generationContext.logger)
                        {
                            writer.write(.buffer(.init(string: sseString)))
                        }
                    }
                }

                if finalFinishReason == nil {
                    finalFinishReason =
                        (generatedTokens.count >= generationParameters.maxTokens)
                        ? "length" : "stop"
                }

                let finalDelta = ChatCompletionDelta(role: nil, content: nil)
                let finalChoice = ChatCompletionChoiceDelta(
                    index: 0, delta: finalDelta, finishReason: finalFinishReason)
                let finalChunk = ChatCompletionChunkResponse(
                    id: chatId, created: created,
                    model: responseContext.loadedModelName,
                    systemFingerprint: systemFingerprint, choices: [finalChoice]
                )
                if let finalSseString = encodeSSE(
                    response: finalChunk, logger: generationContext.logger)
                {
                    _ = try writer.write(.buffer(.init(string: finalSseString)))
                }
            } catch {
                generationContext.logger.error(
                    "Chat stream error (ID: \(chatId)): \(error)")
                finalFinishReason = "error"
            }

            _ = writer.write(.buffer(.init(string: AppConstants.sseDoneMessage)))
            generationContext.logger.info(
                "CHAT Streaming response finished sending (ID: \(chatId)). Final Reason: \(finalFinishReason ?? "unknown")"
            )
            _ = writer.write(.end)
        }
    })
    return response
}

private func handleNonStreamingChatResponse(
    req: Request,
    generationContext: ChatGenerationContext,
    generationParameters: ChatGenerationParameters,
    responseContext: ChatResponseContext
) async throws -> Response {
    var generatedTokens: [Int] = []
    generatedTokens.reserveCapacity(generationParameters.maxTokens)
    var finalFinishReason = "stop"
    let responseId = "chatcmpl-\(UUID().uuidString)"
    let created = Int(Date().timeIntervalSince1970)

    do {
        generationContext.logger.info(
            "Starting non-streaming CHAT generation (ID: \(responseId)) for model \(responseContext.loadedModelName)"
        )
        let tokenStream = try await generateChatTokenStream(
            context: generationContext,
            parameters: generationParameters
        )

        for try await token in tokenStream {
            generatedTokens.append(token)
            let stopCondition = checkStoppingCriteria(
                tokens: generatedTokens,
                stopIdSequences: responseContext.stopIdSequences,
                eosTokenId: generationContext.eosTokenId)

            if stopCondition.stopMet {
                if stopCondition.trimLength > 0
                    && generatedTokens.count >= stopCondition.trimLength
                {
                    generatedTokens.removeLast(stopCondition.trimLength)
                }
                finalFinishReason = "stop"
                break
            }
        }

        if finalFinishReason != "stop" {
            finalFinishReason =
                (generatedTokens.count >= generationParameters.maxTokens) ? "length" : "stop"
        }
    } catch {
        generationContext.logger.error(
            "Non-streaming chat generation error (ID: \(responseId)): \(error)")
        throw ProcessingError(
            status: .internalServerError, reason: "Failed to generate chat completion",
            underlyingError: error)
    }

    let completionText = decodeTokens(generatedTokens, tokenizer: generationContext.tokenizer)

    let assistantMessage = ChatMessageResponseData(role: "assistant", content: completionText)
    let chatChoice = ChatCompletionChoice(
        index: 0, message: assistantMessage, finishReason: finalFinishReason)
    let usage = CompletionUsage(
        promptTokens: responseContext.estimatedPromptTokens,
        completionTokens: generatedTokens.count,
        totalTokens: responseContext.estimatedPromptTokens + generatedTokens.count
    )

    let chatResponse = ChatCompletionResponse(
        id: responseId, created: created, model: responseContext.loadedModelName,
        choices: [chatChoice], usage: usage
    )

    generationContext.logger.info(
        "Non-streaming CHAT response generated (ID: \(responseId)). Reason: \(finalFinishReason)"
    )
    return try await chatResponse.encodeResponse(for: req)
}
