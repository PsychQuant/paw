import Foundation
import Logging
import MLX
import MLXEmbedders
import MLXLMCommon
import Tokenizers

/// Manager for embedding models with caching and reuse.
actor EmbeddingModelManager {
    private var loadedModel: LoadedEmbeddingModel?
    private let defaultModelId: String?
    private let logger: Logger

    init(defaultModelId: String?, logger: Logger) {
        self.defaultModelId = defaultModelId
        self.logger = logger
    }

    func getModel(requestedModelId: String? = nil) async throws -> (
        MLXEmbedders.ModelContainer, String
    ) {
        let modelIdToLoad =
            requestedModelId ?? defaultModelId ?? "mlx-community/snowflake-arctic-embed-m-v1.5"

        if let loaded = loadedModel, loaded.modelId == modelIdToLoad {
            return (loaded.container, loaded.modelId)
        }

        if loadedModel != nil {
            logger.info("Clearing previously loaded embedding model")
            loadedModel = nil
        }

        logger.info("Loading embedding model: \(modelIdToLoad)")
        let container = try await loadEmbeddingModel(modelId: modelIdToLoad)
        loadedModel = LoadedEmbeddingModel(modelId: modelIdToLoad, container: container)

        return (container, modelIdToLoad)
    }

    private func loadEmbeddingModel(modelId: String) async throws
        -> MLXEmbedders.ModelContainer
    {
        do {
            let configuration = ModelConfiguration(id: modelId)
            let container = try await MLXEmbedders.loadModelContainer(
                configuration: configuration)

            MLX.GPU.set(cacheLimit: 20 * 1_024 * 1_024)

            logger.info("Embedding model loaded successfully: \(modelId)")
            return container
        } catch {
            logger.error("Failed to load embedding model \(modelId): \(error)")
            throw error
        }
    }
}

private struct LoadedEmbeddingModel {
    let modelId: String
    let container: MLXEmbedders.ModelContainer
}
