import Foundation
import Logging
import Vapor

func registerModelsRoute(_ app: Application, modelManager: ModelManager) {
    // GET /v1/models - list all available models
    app.get("v1", "models") { req async throws -> ModelListResponse in
        req.logger.info("Handling /v1/models request")
        let modelIds = await modelManager.getAvailableModelIDs()
        let modelInfos = modelIds.map { ModelInfo(id: $0) }
        return ModelListResponse(data: modelInfos)
    }

    // GET /v1/models/:model - get single model info
    app.get("v1", "models", ":model") { req async throws -> ModelInfo in
        guard let modelId = req.parameters.get("model") else {
            throw Abort(.badRequest, reason: "Model ID is required")
        }

        req.logger.info("Handling /v1/models/\(modelId) request")

        let availableIds = await modelManager.getAvailableModelIDs()
        guard availableIds.contains(modelId) else {
            throw Abort(.notFound, reason: "Model '\(modelId)' not found")
        }

        return ModelInfo(id: modelId)
    }

    // GET /health - health check
    app.get("health") { req async throws -> HealthResponse in
        req.logger.debug("Handling /health request")
        return HealthResponse(status: "ok", version: "0.1.0")
    }
}
