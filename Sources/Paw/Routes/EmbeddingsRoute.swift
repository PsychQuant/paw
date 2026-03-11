import Foundation
import MLX
import MLXEmbedders
import Tokenizers
import Vapor

func registerEmbeddingsRoute(
    _ app: Application, embeddingModelManager: EmbeddingModelManager
) {
    app.post("v1", "embeddings") { req async throws -> EmbeddingResponse in
        let embeddingRequest = try req.content.decode(EmbeddingRequest.self)
        let logger = req.logger
        let embeddingReqId = "emb-\(UUID().uuidString)"

        logger.info(
            "Received embedding request (ID: \(embeddingReqId)) for model: \(embeddingRequest.model ?? "Default")"
        )

        let (modelContainer, loadedModelName) = try await embeddingModelManager.getModel(
            requestedModelId: embeddingRequest.model
        )

        let texts = embeddingRequest.input.values
        guard !texts.isEmpty, texts.allSatisfy({ !$0.isEmpty }) else {
            logger.error(
                "Embedding request (ID: \(embeddingReqId)) input is empty or contains empty strings."
            )
            throw Abort(.badRequest, reason: "Input text(s) cannot be empty.")
        }

        let encodingFormat = embeddingRequest.encoding_format ?? "float"
        let batchSize = embeddingRequest.batch_size ?? texts.count

        logger.debug(
            "Processing \(texts.count) text(s) for embedding (ID: \(embeddingReqId)) with model \(loadedModelName). Batch size: \(batchSize), Format: \(encodingFormat)"
        )

        return modelContainer.perform { model, tokenizer, pooling in
            var allData: [EmbeddingData] = []
            var promptTokens = 0
            var index = 0

            for batchStart in stride(from: 0, to: texts.count, by: batchSize) {
                let batchEnd = min(batchStart + batchSize, texts.count)
                let batchTexts = Array(texts[batchStart..<batchEnd])
                logger.trace(
                    "Processing batch \(batchStart / batchSize + 1) for \(embeddingReqId), indices \(batchStart)..<\(batchEnd)"
                )

                let tokenized = batchTexts.map {
                    tokenizer.encode(text: $0, addSpecialTokens: true)
                }
                let currentBatchTokens = tokenized.reduce(0) { $0 + $1.count }
                promptTokens += currentBatchTokens
                logger.trace(
                    "Batch tokens for \(embeddingReqId): \(currentBatchTokens)")

                let maxLength = tokenized.map { $0.count }.max() ?? 16
                let padId = tokenizer.eosTokenId ?? 0

                let paddedArrays = tokenized.map { elem in
                    MLXArray(
                        elem
                            + Array(
                                repeating: padId,
                                count: maxLength - elem.count))
                }

                guard !paddedArrays.isEmpty else { continue }

                let padded = MLX.stacked(paddedArrays)
                let attentionMask = padded .!= MLXArray(padId)
                let tokenTypeIds = MLXArray.zeros(like: padded)

                let output = model(
                    padded, positionIds: nil, tokenTypeIds: tokenTypeIds,
                    attentionMask: attentionMask)

                // Apply pooling to get final embeddings
                let embeddings = pooling(output, mask: attentionMask, normalize: true)

                switch encodingFormat {
                case "base64":
                    for i in 0..<embeddings.shape[0] {
                        let arr = embeddings[i].asArray(Float.self)
                        let data = arr.withUnsafeBufferPointer { Data(buffer: $0) }
                        let base64 = data.base64EncodedString()
                        allData.append(
                            EmbeddingData(embedding: .base64(base64), index: index))
                        index += 1
                    }
                default:
                    for i in 0..<embeddings.shape[0] {
                        let arr = embeddings[i].asArray(Float.self)
                        allData.append(
                            EmbeddingData(embedding: .floats(arr), index: index))
                        index += 1
                    }
                }
                logger.trace(
                    "Finished processing batch \(batchStart / batchSize + 1) for \(embeddingReqId). Total embeddings so far: \(index)"
                )
            }
            let usage = UsageData(
                prompt_tokens: promptTokens, total_tokens: promptTokens)
            logger.info(
                "Embedding generation complete (ID: \(embeddingReqId)) for model \(loadedModelName). Total tokens: \(promptTokens)"
            )

            return EmbeddingResponse(
                data: allData, model: loadedModelName, usage: usage)
        }
    }
}
