import CoreImage
import Foundation
import Hub
import Logging
import MLX
import MLXLLM
@preconcurrency import MLXLMCommon
@preconcurrency import Tokenizers
import Vapor

// MARK: - Atomic Counter

final class AtomicCounter {
    private let lock = NSLock()
    private var _value = 0

    var value: Int {
        lock.lock()
        defer { lock.unlock() }
        return _value
    }

    func increment() {
        lock.lock()
        defer { lock.unlock() }
        _value += 1
    }
}

// MARK: - Context Types

struct ChatGenerationContext {
    let modelContainer: ModelContainer
    let tokenizer: Tokenizer
    let eosTokenId: Int
    let userInput: UserInput
    let logger: Logger
    let promptCacheManager: PromptCacheManager?
}

struct ChatGenerationParameters {
    let maxTokens: Int
    let temperature: Float
    let topP: Float
    let repetitionPenalty: Float
    let repetitionContextSize: Int
    let kvBits: Int?
    let kvGroupSize: Int
    let quantizedKVStart: Int
}

struct ChatResponseContext {
    let loadedModelName: String
    let stopIdSequences: [[Int]]
    let detokenizer: NaiveStreamingDetokenizer
    let estimatedPromptTokens: Int
}

// MARK: - Message Processing Keys

private enum MessageProcessingKeys {
    static let role = "role"
    static let content = "content"
    static let type = "type"
    static let text = "text"
    static let imageType = "image"
    static let videoType = "video"
}

// MARK: - Message Processing

func processTextOnlyMessages(_ chatRequest: ChatCompletionRequest) -> UserInput {
    let messages: [[String: Any]] = chatRequest.messages.map {
        [
            MessageProcessingKeys.role: $0.role,
            MessageProcessingKeys.content: $0.content.asString ?? "",
        ]
    }
    return UserInput(messages: messages)
}

func processVLMMessages(_ chatRequest: ChatCompletionRequest) -> UserInput {
    var allImages: [UserInput.Image] = []
    var allVideos: [UserInput.Video] = []

    let processedMessages: [[String: Any]] = chatRequest.messages.map {
        message -> [String: Any] in
        switch message.content {
        case .text(let textContent):
            return [
                MessageProcessingKeys.role: message.role,
                MessageProcessingKeys.content: textContent,
            ]

        case .fragments(let fragments):
            let imageFragments = fragments.filter {
                $0.type == MessageProcessingKeys.imageType
            }
            let videoFragments = fragments.filter {
                $0.type == MessageProcessingKeys.videoType
            }

            let images = imageFragments.compactMap { fragment in
                fragment.imageUrl.map { UserInput.Image.url($0) }
            }
            allImages.append(contentsOf: images)

            let videos = videoFragments.compactMap { fragment in
                fragment.videoUrl.map { UserInput.Video.url($0) }
            }
            allVideos.append(contentsOf: videos)

            if !images.isEmpty || !videos.isEmpty {
                var contentFragments: [[String: Any]] = []

                fragments.forEach { fragment in
                    if fragment.type == MessageProcessingKeys.text, let text = fragment.text {
                        contentFragments.append([
                            MessageProcessingKeys.type: MessageProcessingKeys.text,
                            MessageProcessingKeys.text: text,
                        ])
                    }
                }

                contentFragments.append(
                    contentsOf: imageFragments.map { _ in
                        [MessageProcessingKeys.type: MessageProcessingKeys.imageType]
                    })
                contentFragments.append(
                    contentsOf: videoFragments.map { _ in
                        [MessageProcessingKeys.type: MessageProcessingKeys.videoType]
                    })

                return [
                    MessageProcessingKeys.role: message.role,
                    MessageProcessingKeys.content: contentFragments,
                ]
            } else {
                return [
                    MessageProcessingKeys.role: message.role,
                    MessageProcessingKeys.content: message.content.asString ?? "",
                ]
            }

        case .none:
            return [
                MessageProcessingKeys.role: message.role,
                MessageProcessingKeys.content: "",
            ]
        }
    }

    var userInput = UserInput(
        messages: processedMessages, images: allImages, videos: allVideos)

    if let resize = chatRequest.resize, !resize.isEmpty {
        let size: CGSize
        if resize.count == 1 {
            let value = resize[0]
            size = CGSize(width: value, height: value)
        } else if resize.count >= 2 {
            let v0 = resize[0]
            let v1 = resize[1]
            size = CGSize(width: v0, height: v1)
        } else {
            size = .zero
        }

        if size != .zero {
            userInput.processing.resize = size
        }
    }

    return userInput
}

func processUserMessages(_ chatRequest: ChatCompletionRequest, isVLM: Bool) -> UserInput {
    if isVLM {
        return processVLMMessages(chatRequest)
    } else {
        return processTextOnlyMessages(chatRequest)
    }
}

func estimatePromptTokens(messages: [ChatMessageRequestData], tokenizer: Tokenizer) -> Int {
    let combinedContent = messages.compactMap { $0.content.asString }.joined(separator: "\n")
    return tokenizer.encode(text: combinedContent).count
}

func validateProcessor(modelContainer: ModelContainer) async throws {
    _ = await modelContainer.perform { context in
        _ = context.processor
    }
}

// MARK: - Chat Token Stream Generation

func generateChatTokenStream(
    context: ChatGenerationContext,
    parameters: ChatGenerationParameters
) async throws -> AsyncStream<Int> {
    return AsyncStream { continuation in
        Task {
            let tokenCount = AtomicCounter()
            do {
                let generateParameters = GenerateParameters(
                    kvBits: parameters.kvBits,
                    kvGroupSize: parameters.kvGroupSize,
                    quantizedKVStart: parameters.quantizedKVStart,
                    temperature: parameters.temperature,
                    topP: parameters.topP,
                    repetitionPenalty: parameters.repetitionPenalty,
                    repetitionContextSize: parameters.repetitionContextSize
                )

                // Store original parameters for cache manager
                let originalGenerateParameters = generateParameters

                _ = try await context.modelContainer.perform { modelContext in
                    let lmInput: LMInput = try await modelContext.processor.prepare(
                        input: context.userInput)

                    let promptTokens = lmInput.text.tokens.asArray(Int.self)

                    var tokensToProcess: [Int] = []
                    var existingCache: [KVCache]?
                    if let cacheManager = context.promptCacheManager {
                        tokensToProcess = promptTokens
                        let modelKey = await context.modelContainer.configuration.name
                        let cacheResult = await cacheManager.getCachedState(
                            modelKey: modelKey,
                            tokens: promptTokens,
                            parameters: generateParameters,
                            model: modelContext.model
                        )
                        tokensToProcess = cacheResult.tokensToProcess
                        existingCache = cacheResult.cache

                        if existingCache != nil {
                            context.logger.info(
                                "Using cached prompt prefix, processing \(tokensToProcess.count) new tokens"
                            )
                        }
                    }

                    let inputForGeneration =
                        tokensToProcess.isEmpty
                        ? lmInput : LMInput(tokens: MLXArray(tokensToProcess))

                    let cache =
                        existingCache
                        ?? modelContext.model.newCache(parameters: generateParameters)

                    // Create modified parameters to prevent TokenIterator quantization
                    var iteratorParameters = generateParameters
                    iteratorParameters.quantizedKVStart = Int.max

                    let iterator = try TokenIterator(
                        input: inputForGeneration,
                        model: modelContext.model,
                        cache: cache,
                        parameters: iteratorParameters
                    )

                    var allGeneratedTokens: [Int] = []

                    for token in iterator {
                        if token == context.eosTokenId { break }
                        if tokenCount.value >= parameters.maxTokens { break }

                        if token == context.tokenizer.unknownTokenId {
                            context.logger.warning(
                                "Generated unknown token ID \(token). Skipping.")
                        } else {
                            continuation.yield(token)
                            tokenCount.increment()
                            allGeneratedTokens.append(token)
                        }
                    }

                    if let cacheManager = context.promptCacheManager {
                        let fullTokens = promptTokens + allGeneratedTokens
                        await cacheManager.updateCache(
                            modelKey: context.modelContainer.configuration.name,
                            tokens: fullTokens,
                            kvCaches: cache,
                            parameters: originalGenerateParameters,
                            model: modelContext.model
                        )
                    }
                }
                continuation.finish()
            } catch {
                context.logger.error("Chat token stream error: \(error)")
                continuation.finish()
            }
        }
    }
}
